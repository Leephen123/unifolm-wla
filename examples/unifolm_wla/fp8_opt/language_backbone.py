"""Qwen3-VL 36-Layer Language Backbone Optimization (`_QwenLayerCkptFn` + Pers64 Dual QKV Bwd + Zero-DtoH DeepStack)."""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

from .cuda_ops import (
    _FastDeepStackAddFn,
    _as_bf16_contig,
    _fa4_raw_bwd,
    _fa4_raw_fwd,
    dswiglu_quant_dual_slot,
    get_pool,
    lt_fp8_mm,
    quant_bhsd_to_bshd_1d_slot,
    quant_bhsd_to_bshd_dual_slot,
    quantize_1d_slot,
    quantize_dual_slot,
    qwen_qkv_norm_rope_raw_bwd,
    qwen_qkv_norm_rope_raw_fwd,
    rmsnorm_bwd_cuda,
    rmsnorm_quant_1d_slot,
    rmsnorm_quant_dual_slot,
    swiglu_quant_1d_slot,
    swiglu_quant_dual_slot,
    try_begin_direct_ipg_block,
    wait_fp8_weight_quant,
)

def _qwen_layer_infer_raw(
    x_3d: torch.Tensor,
    in_nw: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor, post_nw: torch.Tensor,
    cos_bf16: torch.Tensor, sin_bf16: torch.Tensor,
    wqkv_r: torch.Tensor, wqkv_s: torch.Tensor, slot_qkv: int,
    wo_r: torch.Tensor, wo_s: torch.Tensor, slot_o: int,
    wgu_r: torch.Tensor, wgu_s: torch.Tensor, slot_gu: int,
    wdn_r: torch.Tensor, wdn_s: torch.Tensor,
    h_q: int, h_kv: int, scaling: float, eps_in: float, eps_post: float,
    use_cudnn_sdpa: bool = True,
) -> torch.Tensor:
    B, S, D = x_3d.shape
    M = B * S
    x_2d = _as_bf16_contig(x_3d).view(M, D)
    _, y1_e4, y1_s = rmsnorm_quant_1d_slot(x_2d, in_nw, eps_in, slot_qkv * 5 + 0)
    qkv_2d = lt_fp8_mm(y1_e4, wqkv_r, y1_s, wqkv_s)
    del y1_e4
    q, k, v, _ = qwen_qkv_norm_rope_raw_fwd(
        qkv_2d.view(B, S, -1), qw, kw, cos_bf16, sin_bf16, h_q, h_kv, save_rstd=False, bhsd=False
    )
    del qkv_2d
    if use_cudnn_sdpa:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            attn_bh = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, scale=scaling, enable_gqa=True,
            )
        del q, k, v
        attn_t = attn_bh.transpose(1, 2)
        if attn_t.is_contiguous():
            af_e4, af_s = quantize_1d_slot(attn_t.view(M, -1), slot_o * 5 + 0, torch.float8_e4m3fn)
        else:
            af_e4, af_s = quant_bhsd_to_bshd_1d_slot(attn_bh, slot_o * 5 + 0)
        del attn_bh, attn_t
    else:
        attn_y, _ = _fa4_raw_fwd(q, k, v, causal=True, scale=scaling, need_lse=False)
        del q, k, v
        af_e4, af_s = quantize_1d_slot(attn_y.reshape(M, -1), slot_o * 5 + 0, torch.float8_e4m3fn)
        del attn_y
    h_mid = lt_fp8_mm(af_e4, wo_r, af_s, wo_s, c_res=x_2d)
    del af_e4
    _, y2_e4, y2_s = rmsnorm_quant_1d_slot(h_mid, post_nw, eps_post, slot_gu * 5 + 0)
    gu = lt_fp8_mm(y2_e4, wgu_r, y2_s, wgu_s)
    del y2_e4
    h_e4, h_s = swiglu_quant_1d_slot(gu, slot_gu * 5 + 2)
    del gu
    out_2d = lt_fp8_mm(h_e4, wdn_r, h_s, wdn_s, c_res=h_mid)
    del h_e4, h_mid
    return out_2d.view(B, S, D)


class _QwenLayerCkptFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_3d: torch.Tensor,
        in_nw: torch.Tensor,
        wqkv_param: torch.Tensor,
        qw: torch.Tensor,
        kw: torch.Tensor,
        wo_param: torch.Tensor,
        post_nw: torch.Tensor,
        wgu_param: torch.Tensor,
        wdn_param: torch.Tensor,
        cos_bf16: torch.Tensor,
        sin_bf16: torch.Tensor,
        wqkv_r: torch.Tensor, wqkv_t: torch.Tensor, wqkv_s: torch.Tensor, slot_qkv: int,
        wo_r: torch.Tensor, wo_t: torch.Tensor, wo_s: torch.Tensor, slot_o: int,
        wgu_r: torch.Tensor, wgu_t: torch.Tensor, wgu_s: torch.Tensor, slot_gu: int,
        wdn_r: torch.Tensor, wdn_t: torch.Tensor, wdn_s: torch.Tensor, slot_dn: int,
        h_q: int, h_kv: int, scaling: float, eps_in: float, eps_post: float,
        bwd_fast_accum: bool, use_cudnn_sdpa: bool, prev_dn_slot: int = -1,
    ):
        ctx.set_materialize_grads(False)
        ctx.prev_dn_slot = prev_dn_slot
        ctx.all_params = (in_nw, wqkv_param, qw, kw, wo_param, post_nw, wgu_param, wdn_param)
        ctx.dw_params = (wdn_param, wgu_param, wo_param, wqkv_param)
        ctx.orig_shape = x_3d.shape
        ctx.orig_dtype = x_3d.dtype
        ctx.slot_qkv = slot_qkv
        ctx.slot_o = slot_o
        ctx.slot_gu = slot_gu
        ctx.slot_dn = slot_dn
        ctx.h_q = h_q
        ctx.h_kv = h_kv
        ctx.scaling = scaling
        ctx.eps_in = eps_in
        ctx.eps_post = eps_post
        ctx.bwd_fast_accum = bwd_fast_accum
        ctx.use_cudnn_sdpa = use_cudnn_sdpa

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            x_c = _as_bf16_contig(x_3d)
            out_3d = _qwen_layer_infer_raw(
                x_c, in_nw, qw, kw, post_nw, cos_bf16, sin_bf16,
                wqkv_r, wqkv_s, slot_qkv,
                wo_r, wo_s, slot_o,
                wgu_r, wgu_s, slot_gu,
                wdn_r, wdn_s,
                h_q, h_kv, scaling, eps_in, eps_post, use_cudnn_sdpa=use_cudnn_sdpa,
            )
            ctx.save_for_backward(
                x_c, in_nw, qw, kw, post_nw, cos_bf16, sin_bf16,
                wqkv_r, wqkv_t, wqkv_s,
                wo_r, wo_t, wo_s,
                wgu_r, wgu_t, wgu_s,
                wdn_r, wdn_t, wdn_s,
            )
            return out_3d.to(x_3d.dtype)

    @staticmethod
    def backward(ctx, dout_3d):
        if dout_3d is None:
            return (None,) * 35
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            (
                x_c, in_nw, qw, kw, post_nw, cos_bf16, sin_bf16,
                wqkv_r, wqkv_t, wqkv_s,
                wo_r, wo_t, wo_s,
                wgu_r, wgu_t, wgu_s,
                wdn_r, wdn_t, wdn_s,
            ) = ctx.saved_tensors
            B, S, D = ctx.orig_shape
            M = B * S
            x_2d = x_c.view(M, D)
            bfa = ctx.bwd_fast_accum

            # --- Recompute Forward #2 ---
            _, y1_e4, y1_t, rstd1, y1_s = rmsnorm_quant_dual_slot(x_2d, in_nw, ctx.eps_in, ctx.slot_qkv * 5 + 0)
            qkv_2d = lt_fp8_mm(y1_e4, wqkv_r, y1_s, wqkv_s)
            del y1_e4
            qkv_3d = qkv_2d.view(B, S, -1)
            q, k, v, qkv_rstd = qwen_qkv_norm_rope_raw_fwd(
                qkv_3d, qw, kw, cos_bf16, sin_bf16, ctx.h_q, ctx.h_kv, save_rstd=True, bhsd=False
            )

            if ctx.use_cudnn_sdpa:
                q_t = q.transpose(1, 2).detach().requires_grad_(True)
                k_t = k.transpose(1, 2).detach().requires_grad_(True)
                v_t = v.transpose(1, 2).detach().requires_grad_(True)
                del q, k, v
                with torch.enable_grad(), sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    attn_bh = F.scaled_dot_product_attention(
                        q_t, k_t, v_t, is_causal=True, scale=ctx.scaling, enable_gqa=True
                    )
                attn_det_t = attn_bh.detach().transpose(1, 2)
                if attn_det_t.is_contiguous():
                    af_e4, af_t, af_s = quantize_dual_slot(attn_det_t.view(M, -1), ctx.slot_o * 5 + 0, torch.float8_e4m3fn)
                else:
                    af_e4, af_t, af_s = quant_bhsd_to_bshd_dual_slot(attn_bh.detach(), ctx.slot_o * 5 + 0)
                del attn_det_t
            else:
                attn_y, lse = _fa4_raw_fwd(q, k, v, causal=True, scale=ctx.scaling, need_lse=True)
                af_e4, af_t, af_s = quantize_dual_slot(attn_y.reshape(M, -1), ctx.slot_o * 5 + 0, torch.float8_e4m3fn)

            h_mid = lt_fp8_mm(af_e4, wo_r, af_s, wo_s, c_res=x_2d)
            del af_e4
            _, y2_e4, y2_t, rstd2, y2_s = rmsnorm_quant_dual_slot(h_mid, post_nw, ctx.eps_post, ctx.slot_gu * 5 + 0)
            gu = lt_fp8_mm(y2_e4, wgu_r, y2_s, wgu_s)
            del y2_e4
            _, h_t, h_s = swiglu_quant_dual_slot(gu, ctx.slot_gu * 5 + 2, need_row=False)

            # --- Backward Pass ---
            pool_q = get_pool(dout_3d.device)
            wdn_p, wgu_p, wo_p, wqkv_p = ctx.dw_params
            ipg_alloc = try_begin_direct_ipg_block(dout_3d.device, ctx.all_params, ctx.dw_params)
            dout_2d = _as_bf16_contig(dout_3d).view(M, D)
            stashed = getattr(pool_q, "_stashed_qwen_dout", None)
            if stashed is not None and stashed[0] == dout_2d.data_ptr() and stashed[1] == ctx.slot_dn * 5 + 1:
                _, _, dout_row, dout_t, dout_s = stashed
                pool_q._stashed_qwen_dout = None
            else:
                pool_q._stashed_qwen_dout = None
                dout_row, dout_t, dout_s = quantize_dual_slot(dout_2d, ctx.slot_dn * 5 + 1, torch.float8_e5m2)
            dh = lt_fp8_mm(dout_row, wdn_t, dout_s, wdn_s, fast_accum=bfa, is_bwd=True)
            d_wdn = lt_fp8_mm(dout_t, h_t, dout_s, h_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wdn_p) if ipg_alloc is not None else None)
            del dout_row, dout_t, h_t

            dgu_row, dgu_t, dgu_s = dswiglu_quant_dual_slot(dh, gu, ctx.slot_gu * 5 + 3)
            del dh, gu
            d_norm2 = lt_fp8_mm(dgu_row, wgu_t, dgu_s, wgu_s, fast_accum=bfa, is_bwd=True)
            d_wgu = lt_fp8_mm(dgu_t, y2_t, dgu_s, y2_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wgu_p) if ipg_alloc is not None else None)
            del dgu_row, dgu_t, y2_t

            d_h_mid, dhm_row, dhm_t, dhm_s, d_post_nw = rmsnorm_bwd_cuda(
                d_norm2, h_mid, post_nw, rstd2, d_res_in=dout_2d, quant_slot_id=ctx.slot_o * 5 + 1
            )
            del d_norm2, h_mid, rstd2, dout_2d

            d_af = lt_fp8_mm(dhm_row, wo_t, dhm_s, wo_s, fast_accum=bfa, is_bwd=True)
            d_wo = lt_fp8_mm(dhm_t, af_t, dhm_s, af_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wo_p) if ipg_alloc is not None else None)
            del dhm_row, dhm_t, af_t

            dq_is_bhsd = False
            if ctx.use_cudnn_sdpa:
                dq_bh, dk_bh, dv_bh = torch.autograd.grad(
                    attn_bh, (q_t, k_t, v_t), d_af.view(B, S, ctx.h_q, 128).transpose(1, 2)
                )
                del attn_bh, q_t, k_t, v_t, d_af
                if dq_bh.transpose(1, 2).is_contiguous() and dk_bh.transpose(1, 2).is_contiguous() and dv_bh.transpose(1, 2).is_contiguous():
                    dq, dk, dv = dq_bh.transpose(1, 2), dk_bh.transpose(1, 2), dv_bh.transpose(1, 2)
                else:
                    dq, dk, dv = dq_bh, dk_bh, dv_bh
                    dq_is_bhsd = True
            else:
                dq, dk, dv = _fa4_raw_bwd(
                    q, k, v, attn_y, d_af.view(B, S, ctx.h_q, 128), lse, causal=True, scale=ctx.scaling
                )
                del q, k, v, attn_y, lse, d_af

            dqkv_row, dqkv_t, dqkv_s, d_qw, d_kw = qwen_qkv_norm_rope_raw_bwd(
                dq, dk, dv, qkv_3d, qw, kw, cos_bf16, sin_bf16, qkv_rstd, ctx.h_q, ctx.h_kv,
                bhsd=dq_is_bhsd, quant_slot_id=ctx.slot_qkv * 5 + 1,
            )
            del dq, dk, dv, qkv_3d, qkv_rstd
            d_norm1 = lt_fp8_mm(dqkv_row, wqkv_t, dqkv_s, wqkv_s, fast_accum=bfa, is_bwd=True)
            d_wqkv = lt_fp8_mm(dqkv_t, y1_t, dqkv_s, y1_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wqkv_p) if ipg_alloc is not None else None)
            del dqkv_row, dqkv_t, y1_t
            if ipg_alloc is not None:
                d_wdn = d_wgu = d_wo = d_wqkv = None

            if ctx.prev_dn_slot >= 0 and pool_q.warmed_up == 1:
                dx_2d, dx_row, dx_t, dx_s, d_in_nw = rmsnorm_bwd_cuda(
                    d_norm1, x_2d, in_nw, rstd1, d_res_in=d_h_mid, quant_slot_id=ctx.prev_dn_slot
                )
                pool_q._stashed_qwen_dout = (dx_2d.data_ptr(), ctx.prev_dn_slot, dx_row, dx_t, dx_s)
            else:
                dx_2d, d_in_nw = rmsnorm_bwd_cuda(d_norm1, x_2d, in_nw, rstd1, d_res_in=d_h_mid)
            del d_norm1, d_h_mid, rstd1

            return (
                dx_2d.view(B, S, D).to(ctx.orig_dtype),
                d_in_nw,
                d_wqkv,
                d_qw,
                d_kw,
                d_wo,
                d_post_nw,
                d_wgu,
                d_wdn,
            ) + (None,) * 26

def run_qwen_decoder_layer(
    layer,
    hidden_states: torch.Tensor,
    *,
    cos_bf16: torch.Tensor,
    sin_bf16: torch.Tensor,
    use_gc: bool = True,
    use_cudnn_sdpa: bool = True,
) -> torch.Tensor:
    attn = layer.self_attn
    mlp = layer.mlp
    qkv_mod = attn.qkv_proj
    o_mod = attn.o_proj
    gu_mod = mlp.gate_up_proj
    dn_mod = mlp.down_proj

    wqkv_r, wqkv_t, wqkv_s = qkv_mod.get_or_quant_weight()
    wo_r, wo_t, wo_s = o_mod.get_or_quant_weight()
    wgu_r, wgu_t, wgu_s = gu_mod.get_or_quant_weight()
    wdn_r, wdn_t, wdn_s = dn_mod.get_or_quant_weight()

    h_q = attn.qkv_sizes[0] // attn.head_dim
    h_kv = attn.qkv_sizes[1] // attn.head_dim
    scaling = float(attn.scaling)
    eps_in = float(layer.input_layernorm.variance_epsilon)
    eps_post = float(layer.post_attention_layernorm.variance_epsilon)

    if not torch.is_grad_enabled():
        with torch.amp.autocast("cuda", enabled=False):
            return _qwen_layer_infer_raw(
                hidden_states,
                layer.input_layernorm.weight, attn.q_norm.weight, attn.k_norm.weight, layer.post_attention_layernorm.weight,
                cos_bf16, sin_bf16,
                wqkv_r, wqkv_s, qkv_mod.slot_id,
                wo_r, wo_s, o_mod.slot_id,
                wgu_r, wgu_s, gu_mod.slot_id,
                wdn_r, wdn_s,
                h_q, h_kv, scaling, eps_in, eps_post, use_cudnn_sdpa=use_cudnn_sdpa,
            ).to(hidden_states.dtype)

    return _QwenLayerCkptFn.apply(
        hidden_states,
        layer.input_layernorm.weight,
        qkv_mod.weight,
        attn.q_norm.weight,
        attn.k_norm.weight,
        o_mod.weight,
        layer.post_attention_layernorm.weight,
        gu_mod.weight,
        dn_mod.weight,
        cos_bf16, sin_bf16,
        wqkv_r, wqkv_t, wqkv_s, qkv_mod.slot_id,
        wo_r, wo_t, wo_s, o_mod.slot_id,
        wgu_r, wgu_t, wgu_s, gu_mod.slot_id,
        wdn_r, wdn_t, wdn_s, dn_mod.slot_id,
        h_q, h_kv, scaling, eps_in, eps_post,
        qkv_mod.bwd_fast_accum, use_cudnn_sdpa, getattr(layer, "_prev_dn_slot", -1),
    )


def fast_qwen_decoder_layer_forward(
    self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None, **kwargs
) -> torch.Tensor:
    cos, sin = position_embeddings
    return run_qwen_decoder_layer(
        self, hidden_states, cos_bf16=_as_bf16_contig(cos), sin_bf16=_as_bf16_contig(sin),
        use_gc=False, use_cudnn_sdpa=(attention_mask is None and getattr(self, "_use_cudnn_sdpa", True)),
    )


def fast_language_model_forward(
    lang_self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    visual_pos_masks=None,
    deepstack_visual_embeds=None,
    **kwargs,
):
    if inputs_embeds is None:
        inputs_embeds = lang_self.embed_tokens(input_ids)

    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        rope_pos_ids = position_ids[1:]
    else:
        text_position_ids = None
        rope_pos_ids = position_ids

    if attention_mask is not None:
        attention_mask = qwen.create_causal_mask(
            config=lang_self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

    wait_fp8_weight_quant(inputs_embeds.device)
    hidden_states = inputs_embeds
    cos, sin = lang_self.rotary_emb(hidden_states, rope_pos_ids)
    cos_bf16 = _as_bf16_contig(cos)
    sin_bf16 = _as_bf16_contig(sin)

    use_gc = lang_self.gradient_checkpointing and lang_self.training
    if use_gc and not hidden_states.requires_grad:
        hidden_states = hidden_states.requires_grad_(True)
    use_cudnn = (attention_mask is None) and getattr(lang_self, "_use_cudnn_sdpa", True)
    vis_idx_i32 = kwargs.pop("visual_pos_idx_i32", None)
    if vis_idx_i32 is None and deepstack_visual_embeds is not None and visual_pos_masks is not None and hidden_states.shape[-1] == 2560:
        vis_idx_i32 = torch.nonzero(visual_pos_masks.reshape(-1), as_tuple=False).squeeze(-1).to(device=hidden_states.device, dtype=torch.int32).contiguous()

    if not getattr(lang_self, "_prev_dn_slots_wired", False):
        use_cross_dout = (os.environ.get("WLA_CROSS_LAYER_DOUT", "0") == "1")
        for li, dl in enumerate(lang_self.layers):
            dl._prev_dn_slot = (lang_self.layers[li - 1].mlp.down_proj.slot_id * 5 + 1) if (use_cross_dout and li > 0) else -1
        lang_self._prev_dn_slots_wired = True

    for layer_idx, decoder_layer in enumerate(lang_self.layers):
        hidden_states = run_qwen_decoder_layer(
            decoder_layer, hidden_states, cos_bf16=cos_bf16, sin_bf16=sin_bf16,
            use_gc=use_gc, use_cudnn_sdpa=use_cudnn,
        )
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            if vis_idx_i32 is not None and hidden_states.shape[-1] == 2560:
                hidden_states = _FastDeepStackAddFn.apply(
                    hidden_states, deepstack_visual_embeds[layer_idx], vis_idx_i32
                )
            else:
                hidden_states = lang_self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

    hidden_states = lang_self.norm(hidden_states)
    return qwen.BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)
