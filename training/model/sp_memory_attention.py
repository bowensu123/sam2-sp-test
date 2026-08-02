# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Sequence-parallel wrapper for SAM2's MemoryAttention.

Memory attention is a small transformer whose self-attention matrix
[B, num_heads, HW, HW] (HW = 64*64 = 4096 at 1024 res) is the main activation
memory consumer per layer. We shard the *spatial* (HW) dimension of `curr`
across the SP group so that:

  * the self-attention matrix becomes [B, num_heads, HW/sp_size, HW] per rank,
  * the cross-attention matrix becomes [B, num_heads, HW/sp_size, T_mem],
  * the per-token MLP / layernorm activations become [B, HW/sp_size, C],

while correctness is preserved by:
  - gathering the projected Q/K/V before RoPE in self-attention (RoPE is
    element-wise per spatial position, so applying it on the full tensor then
    slicing the query is correct),
  - using a *sliced* freqs_cis for the cross-attention query (this rank's
    spatial positions) and the *full* freqs_cis for the cross-attention keys
    (memory covers the whole grid), which requires a single-tensor RoPE helper
    since `apply_rotary_enc` assumes one freqs_cis for both q and k.

Cross-attention K/V come from `memory` (replicated, small). The final output
is all-gathered so downstream SAM heads see the full [HW, B, C] feature map.

Gradient correctness: every memory-attention parameter receives a *partial*
gradient on each SP rank (the attention output depends on this rank's spatial
chunk of queries), so `sync_sp_module_grads` SUMs them across the SP group,
exactly like for `image_encoder`.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.position_encoding import reshape_for_broadcast, apply_rotary_enc
from sam2.modeling.sam.transformer import RoPEAttention

from training.utils import sequence_parallel as sp


def _apply_rotary_enc_single(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply RoPE to a single tensor (q or k) with the given freqs_cis.

    Unlike `apply_rotary_enc` which takes one freqs_cis for both q and k,
    this lets us use *different* freqs_cis for q (sliced to this SP rank's
    spatial chunk) and k (the full grid, repeated for memory tokens).
    """
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    fc = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * fc).flatten(3)
    return x_out.type_as(x).to(x.device)


class SPMemoryAttention(torch.nn.Module):
    """Wraps a `MemoryAttention` to shard the spatial (HW) dimension across SP."""

    def __init__(self, wrapped: MemoryAttention):
        super().__init__()
        self.wrapped = wrapped

    @property
    def d_model(self):
        return self.wrapped.d_model

    def forward(
        self,
        curr,  # [HW, B, C] or list of one [HW, B, C]
        memory: Tensor,  # [T_mem, B, C_mem]
        curr_pos,  # [HW, B, C] or list
        memory_pos: Tensor,  # [T_mem, B, C_mem]
        num_obj_ptr_tokens: int = 0,
    ):
        if not sp.is_sp_enabled():
            return self.wrapped(curr, memory, curr_pos, memory_pos, num_obj_ptr_tokens)

        # Unwrap the single-element list (same as MemoryAttention.forward).
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]

        assert curr.shape[1] == memory.shape[1], "batch size mismatch"

        output = curr  # [HW, B, C] (seq-first)
        if self.wrapped.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        # Shard along the spatial (HW) dim, dim=0 in seq-first layout.
        sp_size = sp.get_sp_size()
        sp_rank = sp.get_sp_rank()
        HW = output.shape[0]
        assert HW % sp_size == 0, (
            f"[SP] memory-attention HW dim ({HW}) not divisible by sp_size "
            f"({sp_size}); check resolution / backbone_stride."
        )
        chunk = HW // sp_size
        start, end = sp_rank * chunk, (sp_rank + 1) * chunk

        output_chunk = output[start:end]  # [HW_chunk, B, C]
        curr_pos_chunk = curr_pos[start:end]  # [HW_chunk, B, C]

        batch_first = self.wrapped.batch_first
        if batch_first:
            # -> [B, HW_chunk, C] / [B, T_mem, C_mem]
            output_chunk = output_chunk.transpose(0, 1)
            curr_pos_chunk = curr_pos_chunk.transpose(0, 1)
            memory_t = memory.transpose(0, 1)
            memory_pos_t = memory_pos.transpose(0, 1)
        else:
            memory_t, memory_pos_t = memory, memory_pos

        for layer in self.wrapped.layers:
            output_chunk = self._sp_layer_forward(
                layer,
                output_chunk,
                memory_t,
                curr_pos_chunk,
                memory_pos_t,
                chunk,
                start,
                end,
                sp_rank,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )

        normed_chunk = self.wrapped.norm(output_chunk)  # [B, HW_chunk, C]

        if batch_first:
            normed_chunk = normed_chunk.transpose(0, 1)  # [HW_chunk, B, C]

        # All-gather the output back to full [HW, B, C] for the SAM heads.
        full_output = sp.sp_all_gather_cat(normed_chunk, dim=0)
        return full_output

    def _sp_layer_forward(
        self,
        layer,
        tgt_chunk: Tensor,  # [B, HW_chunk, C]
        memory: Tensor,  # [B, T_mem, C_mem]
        query_pos_chunk: Tensor,  # [B, HW_chunk, C]
        pos: Tensor,  # [B, T_mem, C_mem]
        chunk: int,
        start: int,
        end: int,
        sp_rank: int,
        num_k_exclude_rope: int = 0,
    ) -> Tensor:
        # 1) SP self-attention: sharded Q, gathered K/V, sharded attn matrix
        tgt_chunk = self._sp_forward_sa(
            layer, tgt_chunk, query_pos_chunk, chunk, start, end, sp_rank
        )
        # 2) SP cross-attention: sharded Q (sliced RoPE), replicated memory K/V
        tgt_chunk = self._sp_forward_ca(
            layer, tgt_chunk, memory, query_pos_chunk, pos,
            chunk, start, end, num_k_exclude_rope,
        )
        # 3) MLP (per-token, runs on the shard directly)
        tgt2 = layer.norm3(tgt_chunk)
        tgt2 = layer.linear2(
            layer.dropout(layer.activation(layer.linear1(tgt2)))
        )
        tgt_chunk = tgt_chunk + layer.dropout3(tgt2)
        return tgt_chunk

    def _sp_forward_sa(
        self,
        layer,
        tgt_chunk: Tensor,  # [B, HW_chunk, C]
        query_pos_chunk: Tensor,  # [B, HW_chunk, C]
        chunk: int,
        start: int,
        end: int,
        sp_rank: int,
    ) -> Tensor:
        """Self-attention with sharded queries and full keys/values.

        Project Q/K/V on the local shard, all-gather them (autograd-aware),
        apply RoPE on the FULL Q/K (freqs_cis matches HW_full), then slice Q
        back to this rank's chunk and run SDPA. The attention matrix is
        [B, num_heads, HW_chunk, HW] — sharded by sp_size on the query dim.
        """
        sa = layer.self_attn  # RoPEAttention
        normed_chunk = layer.norm1(tgt_chunk)
        q_in = normed_chunk + query_pos_chunk if layer.pos_enc_at_attn else normed_chunk
        v_in = normed_chunk  # v = normed(tgt) (without query_pos)

        # Project on the local shard (per-token linear).
        q_proj_chunk = sa.q_proj(q_in)
        k_proj_chunk = sa.k_proj(q_in)  # self-attn: k source == q source
        v_proj_chunk = sa.v_proj(v_in)

        # Gather projected Q/K/V across SP ranks to recover the full HW dim.
        # autograd-aware: backward routes grads to each rank's local projection.
        q_full = sp.sp_all_gather_cat(q_proj_chunk, dim=1)  # [B, HW, internal_dim]
        k_full = sp.sp_all_gather_cat(k_proj_chunk, dim=1)
        v_full = sp.sp_all_gather_cat(v_proj_chunk, dim=1)

        q_h_full = sa._separate_heads(q_full, sa.num_heads)
        k_h_full = sa._separate_heads(k_full, sa.num_heads)
        v_h_full = sa._separate_heads(v_full, sa.num_heads)

        # Apply RoPE on the FULL Q and K. freqs_cis is pre-computed for the
        # full [H, W] grid and matches q_h_full.shape[-2] == HW, so the
        # existing RoPE path works unchanged.
        sa.freqs_cis = sa.freqs_cis.to(q_h_full.device)
        if sa.freqs_cis.shape[0] != q_h_full.shape[-2]:
            import math
            w = h = math.sqrt(q_h_full.shape[-2])
            sa.freqs_cis = sa.compute_cis(end_x=w, end_y=h).to(q_h_full.device)
        # num_k_exclude_rope == 0 for self-attention.
        q_h_full, k_h_full = apply_rotary_enc(
            q_h_full, k_h_full, freqs_cis=sa.freqs_cis,
            repeat_freqs_k=sa.rope_k_repeat,
        )

        # Slice the ROPE'd Q to this rank's spatial chunk — this is what makes
        # the attention matrix [HW_chunk, HW] instead of [HW, HW].
        q_h_chunk = q_h_full[:, :, start:end, :]

        dropout_p = sa.dropout_p if sa.training else 0.0
        out_h_chunk = F.scaled_dot_product_attention(
            q_h_chunk, k_h_full, v_h_full, dropout_p=dropout_p
        )  # [B, nh, HW_chunk, hd]

        out_chunk = sa._recombine_heads(out_h_chunk)
        out_chunk = sa.out_proj(out_chunk)
        return tgt_chunk + layer.dropout1(out_chunk)

    def _sp_forward_ca(
        self,
        layer,
        tgt_chunk: Tensor,  # [B, HW_chunk, C]
        memory: Tensor,  # [B, T_mem, C_mem]
        query_pos_chunk: Tensor,  # [B, HW_chunk, C]
        pos: Tensor,  # [B, T_mem, C_mem]
        chunk: int,
        start: int,
        end: int,
        num_k_exclude_rope: int = 0,
    ) -> Tensor:
        """Cross-attention with sharded queries and replicated memory K/V.

        Q is this rank's spatial chunk; K/V come from `memory` (replicated).
        RoPE needs *different* freqs_cis for q (sliced to this rank's spatial
        positions) and k (the full grid, repeated for memory tokens), so we
        use `_apply_rotary_enc_single` instead of `apply_rotary_enc`.
        The attention matrix is [B, nh, HW_chunk, T_mem] — sharded.
        """
        ca = layer.cross_attn_image  # RoPEAttention
        normed_chunk = layer.norm2(tgt_chunk)
        q_in = normed_chunk + query_pos_chunk if layer.pos_enc_at_cross_attn_queries else normed_chunk
        k_in = memory + pos if layer.pos_enc_at_cross_attn_keys else memory
        v_in = memory

        # Project: q on the shard, k/v on replicated memory.
        q_proj = ca.q_proj(q_in)  # [B, HW_chunk, internal_dim]
        k_proj = ca.k_proj(k_in)  # [B, T_mem, internal_dim] (full, replicated)
        v_proj = ca.v_proj(v_in)

        q_h = ca._separate_heads(q_proj, ca.num_heads)  # [B, nh, HW_chunk, hd]
        k_h = ca._separate_heads(k_proj, ca.num_heads)  # [B, nh, T_mem, hd]
        v_h = ca._separate_heads(v_proj, ca.num_heads)

        # RoPE for q: sliced freqs_cis for this rank's spatial positions.
        ca.freqs_cis = ca.freqs_cis.to(q_h.device)
        freqs_cis_q = ca.freqs_cis[start:end]  # [HW_chunk, ...]
        q_h = _apply_rotary_enc_single(q_h, freqs_cis_q)

        # RoPE for k: full freqs_cis, repeated to num_k_rope (spatial memory
        # covers the whole grid; object pointers at the end get no RoPE).
        num_k_rope = k_h.size(-2) - num_k_exclude_rope
        if num_k_rope > 0:
            k_rope_part = k_h[:, :, :num_k_rope]
            fc_k = ca.freqs_cis  # [HW, ...]
            if ca.rope_k_repeat:
                r = num_k_rope // fc_k.shape[0]
                if r > 1:
                    fc_k = fc_k.repeat(*([1] * (fc_k.ndim - 2)), r, 1)
            k_h = k_h.clone()
            k_h[:, :, :num_k_rope] = _apply_rotary_enc_single(k_rope_part, fc_k)

        dropout_p = ca.dropout_p if ca.training else 0.0
        out_h = F.scaled_dot_product_attention(
            q_h, k_h, v_h, dropout_p=dropout_p
        )  # [B, nh, HW_chunk, hd]

        out = ca._recombine_heads(out_h)
        out = ca.out_proj(out)
        return tgt_chunk + layer.dropout2(out)
