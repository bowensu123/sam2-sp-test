# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sequence (temporal) parallel utilities for SAM2 training.

Splits the image-encoder forward (the heaviest HxW activation consumer) across
GPUs within an SP (sequence / time) group, so the heaviest module's activations
no longer have to fit on a single GPU. resolution / num_frames / batch_size are
left untouched.

Layout (2D device mesh, world_size = sp_size * dp_size):

    global rank r  ->  (dp_rank = r // sp_size, sp_rank = r % sp_size)

  * SP group (row of the mesh): ranks sharing the same `dp_rank` process the
    SAME video. The image encoder forward is sharded across them along the
    (T*B) dimension; `all_gather` then reconstructs the full feature set on
    every SP rank so memory attention / SAM heads run on the full features.
  * DP group (column of the mesh): ranks sharing the same `sp_rank` process
    DIFFERENT videos. DDP averages gradients across the DP group only.

Gradient correctness
---------------------
* Non-image-encoder params: the forward runs on the full gathered features,
  so every SP rank produces the same gradient. DDP's DP-group average already
  gives the correct value; nothing extra is needed.
* Image-encoder params: each SP rank only sees its frame chunk, so its local
  gradient is partial. After DDP (DP-group average) the value on rank (i, j)
  is (1/dp_size) * sum_i' partial_{i'j}. `sync_image_encoder_grads_for_sp`
  then all-reduces (SUM) across the SP group, recovering
  (1/dp_size) * sum_i' sum_j' partial_{i'j'} = correct full gradient.
"""

import logging
from typing import List, Optional

import torch
import torch.distributed as dist


_SP_GROUP: Optional[dist.ProcessGroup] = None
_DP_GROUP: Optional[dist.ProcessGroup] = None
_SP_SIZE: int = 1
_DP_SIZE: int = 1


def init_sequence_parallel(sp_size: int) -> None:
    """Create SP and DP process groups forming a 2D mesh [dp_size, sp_size].

    Must be called after `torch.distributed.init_process_group` and before
    DDP wrapping / dataloader construction. Calling with sp_size <= 1 is a
    no-op (sequence parallelism disabled).
    """
    global _SP_GROUP, _DP_GROUP, _SP_SIZE, _DP_SIZE

    if sp_size <= 1:
        _SP_GROUP = None
        _DP_GROUP = None
        _SP_SIZE = 1
        _DP_SIZE = 1
        return

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "torch.distributed must be initialized before calling "
            "init_sequence_parallel()."
        )

    world_size = dist.get_world_size()
    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sp_size "
            f"({sp_size})."
        )
    dp_size = world_size // sp_size
    rank = dist.get_rank()
    dp_rank = rank // sp_size
    sp_rank = rank % sp_size

    # `new_group` is collective: every rank must call it with the same ranks
    # list. We loop over all SP/DP groups and keep only the one containing
    # this rank.
    sp_group_local: Optional[dist.ProcessGroup] = None
    for i in range(dp_size):
        sp_ranks = list(range(i * sp_size, (i + 1) * sp_size))
        group = dist.new_group(ranks=sp_ranks)
        if i == dp_rank:
            sp_group_local = group

    dp_group_local: Optional[dist.ProcessGroup] = None
    for j in range(sp_size):
        dp_ranks = list(range(j, world_size, sp_size))
        group = dist.new_group(ranks=dp_ranks)
        if j == sp_rank:
            dp_group_local = group

    _SP_GROUP = sp_group_local
    _DP_GROUP = dp_group_local
    _SP_SIZE = sp_size
    _DP_SIZE = dp_size
    logging.info(
        f"[SP] Sequence parallel initialized: world_size={world_size} "
        f"sp_size={sp_size} dp_size={dp_size} global_rank={rank} "
        f"sp_rank={sp_rank} dp_rank={dp_rank}"
    )


def is_sp_enabled() -> bool:
    return _SP_SIZE > 1


def get_sp_group() -> Optional[dist.ProcessGroup]:
    return _SP_GROUP


def get_dp_group() -> Optional[dist.ProcessGroup]:
    return _DP_GROUP


def get_sp_size() -> int:
    return _SP_SIZE


def get_dp_size() -> int:
    return _DP_SIZE


def get_sp_rank() -> int:
    if not is_sp_enabled():
        return 0
    return dist.get_rank(group=_SP_GROUP)


def get_dp_rank() -> int:
    if not is_sp_enabled():
        return dist.get_rank()
    return dist.get_rank(group=_DP_GROUP)


def sp_all_gather_cat(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Autograd-aware all-gather across the SP group, then cat along `dim`.

    Forward: every SP rank provides its local chunk; the returned tensor is
    identical on all SP ranks and equals torch.cat([rank0, rank1, ...], dim).
    Backward: reduce-scatter routes gradients back to each rank's local chunk
    only, so image-encoder parameters receive partial gradients (as expected).
    """
    if not is_sp_enabled():
        return tensor
    gathered = torch.distributed.nn.functional.all_gather(
        tensor, group=_SP_GROUP
    )
    return torch.cat(gathered, dim=dim)


def sync_image_encoder_grads_for_sp(model: torch.nn.Module) -> None:
    """All-reduce (SUM) image-encoder grads across the SP group.

    Must be called AFTER `scaler.unscale_(optimizer)` (when AMP is on) and
    AFTER DDP's DP-group allreduce has run (i.e. after `loss.backward()`).
    Image-encoder grads are partial on each SP rank (only the chunk that rank
    computed); summing them across the SP group recovers the full gradient.
    Other params are already correct after DDP and are intentionally left
    untouched.
    """
    sync_sp_module_grads(model, "image_encoder")


def sync_sp_module_grads(model: torch.nn.Module, *module_names: str) -> None:
    """All-reduce (SUM) grads of the given submodules across the SP group.

    Used for modules whose forward is SP-sharded (`image_encoder`,
    `memory_attention`, `sam_mask_decoder.conv_s0`/`conv_s1`). Each SP rank's
    grad is partial (covers only this rank's spatial/temporal chunk);
    SUM-ing across the SP group recovers the full gradient. Non-sharded
    modules are already correct after DDP and must NOT be summed here (that
    would scale their grad by sp_size).

    `module_names` supports dotted paths, e.g.
    ``"sam_mask_decoder.conv_s0"`` to reach nested submodules.
    """
    if not is_sp_enabled():
        return

    # Unwrap DDP (and SAM2Train) to reach the submodules by name.
    base = model
    while hasattr(base, "module"):
        base = base.module

    def _get_nested_module(root, dotted_name):
        """Traverse a dotted path (e.g. 'sam_mask_decoder.conv_s0')."""
        obj = root
        for part in dotted_name.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    grads: List[torch.Tensor] = []
    for name in module_names:
        mod = _get_nested_module(base, name)
        if mod is None:
            logging.warning(f"[SP] model has no `{name}` submodule; skipping.")
            continue
        if not isinstance(mod, torch.nn.Module):
            logging.warning(
                f"[SP] `{name}` is not an nn.Module ({type(mod)}); skipping."
            )
            continue
        grads.extend(p.grad for p in mod.parameters() if p.grad is not None)

    if not grads:
        return

    # Coalesce into a single buffer for one collective per step (much cheaper
    # than one all_reduce per parameter).
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=_SP_GROUP)
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n
