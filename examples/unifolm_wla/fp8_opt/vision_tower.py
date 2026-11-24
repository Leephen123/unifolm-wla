"""BF16 Vision Tower Optimization: Pure C++ `cublasLt` + Persistent QKV RoPE Bwd + 4-Warp `LayerNorm(1024)`."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from .cuda_ops import (
    _as_bf16_contig,
    _fa4_raw_bwd,
    _fa4_raw_fwd,
    layernorm_1024_bwd_cuda,
    layernorm_1024_fwd_cuda,
    lt_bf16_dw_dbias,
    lt_bf16_dx,
    lt_bf16_fwd,
    sum_rows_bf16_cuda,
    try_begin_direct_ipg_block,
    vis_qkv_rope_raw_bwd,
    vis_qkv_rope_raw_fwd,
)

def _build_1cam_vision_buffers(vis: nn.Module, device: torch.device):
    # Exact 1-camera (t=1, h=20, w=28) bilinear interpolation indices & weights + rotary cos/sin
    h, w = 20, 28
    num_grid = vis.num_grid_per_side
    h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32)
    w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32)
    h_floor = h_idxs.int()
    w_floor = w_idxs.int()
    h_ceil = (h_floor + 1).clip(max=num_grid - 1)
    w_ceil = (w_floor + 1).clip(max=num_grid - 1)
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor
    base_h = h_floor * num_grid
    base_h_ceil = h_ceil * num_grid
    idx_1cam = torch.stack([
        (base_h[:, None] + w_floor[None, :]).flatten(),
        (base_h[:, None] + w_ceil[None, :]).flatten(),
        (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
        (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
    ], dim=0).to(dtype=torch.long, device=device)  # (4, 560)
    w_1cam = torch.stack([
        ((1.0 - dh)[:, None] * (1.0 - dw)[None, :]).flatten(),
        ((1.0 - dh)[:, None] * dw[None, :]).flatten(),
        (dh[:, None] * (1.0 - dw)[None, :]).flatten(),
        (dh[:, None] * dw[None, :]).flatten(),
    ], dim=0).unsqueeze(-1).to(dtype=vis.pos_embed.weight.dtype, device=device)  # (4, 560, 1)

    with torch.no_grad():
        one_grid = torch.tensor([[1, 20, 28]], dtype=torch.long, device=device)
        rot_1cam = vis.rot_pos_emb(one_grid)  # (560, 36)
        emb = torch.cat((rot_1cam, rot_1cam), dim=-1).float()
        cos_1cam = emb.cos().contiguous()  # (560, 72) float32
        sin_1cam = emb.sin().contiguous()  # (560, 72) float32
        dummy_cu = torch.tensor([0, 560], dtype=torch.int32, device=device)
    return idx_1cam, w_1cam, cos_1cam, sin_1cam, dummy_cu

def _vision_block_infer_raw(
    x_2d: torch.Tensor,
    ln1_w: torch.Tensor, ln1_b: torch.Tensor,
    qkv_w: torch.Tensor, qkv_b: torch.Tensor,
    proj_w: torch.Tensor, proj_b: torch.Tensor,
    ln2_w: torch.Tensor, ln2_b: torch.Tensor,
    fc1_w: torch.Tensor, fc1_b: torch.Tensor,
    fc2_w: torch.Tensor, fc2_b: torch.Tensor,
    cos_f32: torch.Tensor, sin_f32: torch.Tensor,
    num_heads: int, scaling: float, eps1: float, eps2: float,
    save_attn: bool = False,
):
    M = x_2d.shape[0]
    y_ln1, _, _ = layernorm_1024_fwd_cuda(x_2d, ln1_w, ln1_b, eps1, save_stats=False)
    qkv, _ = lt_bf16_fwd(y_ln1, qkv_w, bias=qkv_b)
    del y_ln1
    q, k, v = vis_qkv_rope_raw_fwd(qkv, cos_f32, sin_f32, 560, num_heads)
    del qkv
    attn_y, lse = _fa4_raw_fwd(q, k, v, causal=False, scale=scaling, need_lse=save_attn)
    del q, k, v
    h_mid, _ = lt_bf16_fwd(attn_y.reshape(M, -1), proj_w, bias=proj_b, c_res=x_2d)
    if not save_attn:
        del attn_y
    y_ln2, _, _ = layernorm_1024_fwd_cuda(h_mid, ln2_w, ln2_b, eps2, save_stats=False)
    h_mlp, _ = lt_bf16_fwd(y_ln2, fc1_w, bias=fc1_b, gelu=True, save_aux=False)
    del y_ln2
    out, _ = lt_bf16_fwd(h_mlp, fc2_w, bias=fc2_b, c_res=h_mid)
    del h_mlp, h_mid
    return out, attn_y if save_attn else None, lse


class _VisionBlockCkptFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_in: torch.Tensor,
        ln1_w: torch.Tensor, ln1_b: torch.Tensor,
        qkv_w: torch.Tensor, qkv_b: torch.Tensor,
        proj_w: torch.Tensor, proj_b: torch.Tensor,
        ln2_w: torch.Tensor, ln2_b: torch.Tensor,
        fc1_w: torch.Tensor, fc1_b: torch.Tensor,
        fc2_w: torch.Tensor, fc2_b: torch.Tensor,
        cos_f32: torch.Tensor, sin_f32: torch.Tensor,
        num_heads: int, scaling: float, eps1: float, eps2: float,
        use_gc: bool, save_attn: bool,
    ):
        ctx.set_materialize_grads(False)
        ctx.orig_dtype = x_in.dtype
        ctx.num_heads = num_heads
        ctx.scaling = scaling
        ctx.eps1 = eps1
        ctx.eps2 = eps2
        ctx.use_gc = use_gc
        ctx.save_attn = save_attn
        ctx.all_params = (ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b)
        ctx.dw_params = (fc2_w, fc1_w, proj_w, qkv_w)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            x_2d = _as_bf16_contig(x_in)
            M = x_2d.shape[0]
            if use_gc:
                out, attn_y, lse = _vision_block_infer_raw(
                    x_2d, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                    ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b,
                    cos_f32, sin_f32, num_heads, scaling, eps1, eps2, save_attn=save_attn,
                )
                if save_attn:
                    ctx.save_for_backward(
                        x_2d, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                        ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b, cos_f32, sin_f32, attn_y, lse,
                    )
                else:
                    ctx.save_for_backward(
                        x_2d, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                        ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b, cos_f32, sin_f32,
                    )
                return out.to(x_in.dtype)

            # Uncheckpointed training path (when vis_ckpt_blocks < 24)
            y_ln1, mean1, rstd1 = layernorm_1024_fwd_cuda(x_2d, ln1_w, ln1_b, eps1, save_stats=True)
            qkv, _ = lt_bf16_fwd(y_ln1, qkv_w, bias=qkv_b)
            q, k, v = vis_qkv_rope_raw_fwd(qkv, cos_f32, sin_f32, 560, num_heads)
            del qkv
            attn_y, lse = _fa4_raw_fwd(q, k, v, causal=False, scale=scaling, need_lse=True)
            h_mid, _ = lt_bf16_fwd(attn_y.reshape(M, -1), proj_w, bias=proj_b, c_res=x_2d)
            y_ln2, mean2, rstd2 = layernorm_1024_fwd_cuda(h_mid, ln2_w, ln2_b, eps2, save_stats=True)
            h_mlp, z_aux = lt_bf16_fwd(y_ln2, fc1_w, bias=fc1_b, gelu=True, save_aux=True)
            out, _ = lt_bf16_fwd(h_mlp, fc2_w, bias=fc2_b, c_res=h_mid)
            ctx.save_for_backward(
                x_2d, ln1_w, qkv_w, proj_w, ln2_w, fc1_w, fc2_w, cos_f32, sin_f32,
                y_ln1, mean1, rstd1, q, k, v, attn_y, lse, h_mid, y_ln2, mean2, rstd2, z_aux, h_mlp,
            )
            return out.to(x_in.dtype)

    @staticmethod
    def backward(ctx, dout_in):
        if dout_in is None:
            return (None,) * 21
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            dout = _as_bf16_contig(dout_in)
            M = dout.shape[0]
            if ctx.use_gc:
                saved = ctx.saved_tensors
                (
                    x_2d, ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b,
                    ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b, cos_f32, sin_f32,
                ) = saved[:15]
                y_ln1, mean1, rstd1 = layernorm_1024_fwd_cuda(x_2d, ln1_w, ln1_b, ctx.eps1, save_stats=True)
                qkv, _ = lt_bf16_fwd(y_ln1, qkv_w, bias=qkv_b)
                q, k, v = vis_qkv_rope_raw_fwd(qkv, cos_f32, sin_f32, 560, ctx.num_heads)
                del qkv
                if ctx.save_attn:
                    attn_y, lse = saved[15], saved[16]
                else:
                    attn_y, lse = _fa4_raw_fwd(q, k, v, causal=False, scale=ctx.scaling, need_lse=True)
                h_mid, _ = lt_bf16_fwd(attn_y.reshape(M, -1), proj_w, bias=proj_b, c_res=x_2d)
                y_ln2, mean2, rstd2 = layernorm_1024_fwd_cuda(h_mid, ln2_w, ln2_b, ctx.eps2, save_stats=True)
                h_mlp, z_aux = lt_bf16_fwd(y_ln2, fc1_w, bias=fc1_b, gelu=True, save_aux=True)
            else:
                (
                    x_2d, ln1_w, qkv_w, proj_w, ln2_w, fc1_w, fc2_w, cos_f32, sin_f32,
                    y_ln1, mean1, rstd1, q, k, v, attn_y, lse, h_mid, y_ln2, mean2, rstd2, z_aux, h_mlp,
                ) = ctx.saved_tensors

            ipg_alloc = try_begin_direct_ipg_block(dout.device, ctx.all_params, ctx.dw_params) if ctx.use_gc else None
            d_fc2_w, _ = lt_bf16_dw_dbias(dout, h_mlp, need_dbias=False, out=ipg_alloc(fc2_w) if ipg_alloc is not None else None)
            del h_mlp
            dz = lt_bf16_dx(dout, fc2_w, z_aux=z_aux, need_dgelu_bgrad=False)
            del z_aux
            d_fc1_b = sum_rows_bf16_cuda(dz)
            d_fc1_w, _ = lt_bf16_dw_dbias(dz, y_ln2, need_dbias=False, out=ipg_alloc(fc1_w) if ipg_alloc is not None else None)
            del y_ln2
            d_ln2 = lt_bf16_dx(dz, fc1_w)
            del dz
            d_h_mid, d_ln2_w, d_ln2_b, d_fc2_b = layernorm_1024_bwd_cuda(
                d_ln2, h_mid, ln2_w, mean2, rstd2, d_res_in=dout, need_dres_sum=True
            )
            del d_ln2, h_mid, mean2, rstd2, dout

            attn_2d = attn_y.reshape(M, -1)
            d_proj_w, _ = lt_bf16_dw_dbias(d_h_mid, attn_2d, need_dbias=False, out=ipg_alloc(proj_w) if ipg_alloc is not None else None)
            d_attn_2d = lt_bf16_dx(d_h_mid, proj_w)
            dq, dk, dv = _fa4_raw_bwd(
                q, k, v, attn_y, d_attn_2d.view_as(attn_y), lse, causal=False, scale=ctx.scaling
            )
            del q, k, v, attn_y, lse, d_attn_2d
            dqkv, d_qkv_b = vis_qkv_rope_raw_bwd(dq, dk, dv, cos_f32, sin_f32, 560, ctx.num_heads, need_dbias=True)
            del dq, dk, dv
            d_qkv_w, _ = lt_bf16_dw_dbias(dqkv, y_ln1, need_dbias=False, out=ipg_alloc(qkv_w) if ipg_alloc is not None else None)
            if ipg_alloc is not None:
                d_fc2_w = d_fc1_w = d_proj_w = d_qkv_w = None
            del y_ln1
            d_ln1 = lt_bf16_dx(dqkv, qkv_w)
            del dqkv
            dx, d_ln1_w, d_ln1_b, d_proj_b = layernorm_1024_bwd_cuda(
                d_ln1, x_2d, ln1_w, mean1, rstd1, d_res_in=d_h_mid, need_dres_sum=True
            )
            del d_ln1, d_h_mid, mean1, rstd1

            return (
                dx.to(ctx.orig_dtype),
                d_ln1_w, d_ln1_b,
                d_qkv_w, d_qkv_b,
                d_proj_w, d_proj_b,
                d_ln2_w, d_ln2_b,
                d_fc1_w, d_fc1_b,
                d_fc2_w, d_fc2_b,
                None, None, None, None, None, None, None, None,
            )

def run_vision_block(
    blk, hidden_states: torch.Tensor, *, cos_f32: torch.Tensor, sin_f32: torch.Tensor,
    use_gc: bool = True, save_attn: bool = False,
) -> torch.Tensor:
    num_heads = int(blk.attn.num_heads)
    scaling = float(getattr(blk.attn, "scaling", 64 ** -0.5))
    eps1 = float(getattr(blk.norm1, "eps", 1e-6))
    eps2 = float(getattr(blk.norm2, "eps", 1e-6))

    if not torch.is_grad_enabled():
        with torch.amp.autocast("cuda", enabled=False):
            out, _, _ = _vision_block_infer_raw(
                _as_bf16_contig(hidden_states),
                blk.norm1.weight, blk.norm1.bias,
                blk.attn.qkv.weight, blk.attn.qkv.bias,
                blk.attn.proj.weight, blk.attn.proj.bias,
                blk.norm2.weight, blk.norm2.bias,
                blk.mlp.linear_fc1.weight, blk.mlp.linear_fc1.bias,
                blk.mlp.linear_fc2.weight, blk.mlp.linear_fc2.bias,
                cos_f32, sin_f32, num_heads, scaling, eps1, eps2, save_attn=False,
            )
            return out.to(hidden_states.dtype)

    return _VisionBlockCkptFn.apply(
        hidden_states,
        blk.norm1.weight, blk.norm1.bias,
        blk.attn.qkv.weight, blk.attn.qkv.bias,
        blk.attn.proj.weight, blk.attn.proj.bias,
        blk.norm2.weight, blk.norm2.bias,
        blk.mlp.linear_fc1.weight, blk.mlp.linear_fc1.bias,
        blk.mlp.linear_fc2.weight, blk.mlp.linear_fc2.bias,
        cos_f32, sin_f32, num_heads, scaling, eps1, eps2, use_gc, save_attn,
    )


def fast_vision_block_forward(
    self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor = None,
    rotary_pos_emb=None, position_embeddings=None, **kwargs
) -> torch.Tensor:
    cos_f32, sin_f32 = position_embeddings
    return run_vision_block(self, hidden_states, cos_f32=cos_f32, sin_f32=sin_f32, use_gc=False, save_attn=False)


def fast_vision_model_forward(vis_self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
    hidden_states = vis_self.patch_embed(hidden_states)
    if vis_self._1cam_idx is None or vis_self._1cam_idx.device != hidden_states.device:
        idx_1, w_1, cos_1, sin_1, d_cu = _build_1cam_vision_buffers(vis_self, hidden_states.device)
        vis_self.register_buffer("_1cam_idx", idx_1, persistent=False)
        vis_self.register_buffer("_1cam_w", w_1, persistent=False)
        vis_self.register_buffer("_1cam_cos", cos_1, persistent=False)
        vis_self.register_buffer("_1cam_sin", sin_1, persistent=False)
        vis_self.register_buffer("_dummy_cu", d_cu, persistent=False)

    ms = vis_self.config.spatial_merge_size
    pos_raw = (vis_self.pos_embed(vis_self._1cam_idx) * vis_self._1cam_w.to(vis_self.pos_embed.weight.dtype)).sum(dim=0)
    pos_1cam = pos_raw.view(1, 20 // ms, ms, 28 // ms, ms, -1).permute(0, 1, 3, 2, 4, 5).reshape(1, 560, -1)
    d_vis = hidden_states.shape[-1]
    hidden_states = (hidden_states.view(-1, 560, d_vis) + pos_1cam.to(hidden_states.dtype)).view(-1, d_vis)

    cos_f32 = vis_self._1cam_cos
    sin_f32 = vis_self._1cam_sin
    gc_active = vis_self.gradient_checkpointing and vis_self.training
    if gc_active and not hidden_states.requires_grad:
        hidden_states = hidden_states.requires_grad_(True)

    num_ckpt = int(getattr(vis_self, "_num_ckpt_blocks", len(vis_self.blocks))) if gc_active else 0
    save_attn = bool(getattr(vis_self, "_attn_saved_sac", False))

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(vis_self.blocks):
        blk_gc = gc_active and (layer_num < num_ckpt)
        hidden_states = run_vision_block(
            blk, hidden_states, cos_f32=cos_f32, sin_f32=sin_f32,
            use_gc=blk_gc, save_attn=(blk_gc and save_attn),
        )
        if layer_num in vis_self.deepstack_visual_indexes:
            deepstack_feature = vis_self.deepstack_merger_list[vis_self.deepstack_visual_indexes.index(layer_num)](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    merged_hidden_states = vis_self.merger(hidden_states)
    return qwen.BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
        deepstack_features=deepstack_feature_lists,
    )
