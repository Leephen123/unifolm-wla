"""MMDiT 16-Block Action Expert Optimization (`_MMDiTBlockCkptFn` + Zero-Copy BHSD Pers64 Dual E5M2 QKV Bwd)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .cuda_ops import (
    _as_bf16_contig,
    _fa4_raw_bwd as _base_fa4_bwd,
    _fa4_raw_fwd as _base_fa4_fwd,
    _mmdit_joint_qkv_48_bwd_e5_dual,
    _mmdit_joint_qkv_raw_bwd,
    _mmdit_joint_qkv_raw_fwd,
    _mmdit_split_48_bwd_bhsd,
    _mmdit_split_quant_fwd,
    _mmdit_split_unpad_raw_bwd,
    _mmdit_split_unpad_raw_fwd,
    _ptr,
    _stream,
    adaln_modulate_bwd_1536_cuda,
    adaln_modulate_quant_1536_slot,
    dgelu_quant_dual_slot,
    gated_add_fwd_1536_cuda,
    gated_bwd_quant_e5_1536_slot,
    gelu_bias_quant_1d_slot,
    gelu_bias_quant_dual_slot,
    get_cuda_lib,
    get_pool,
    lt_fp8_mm,
    quantize_1d_slot,
    quantize_dual_slot,
    try_begin_direct_ipg_block,
    wait_fp8_weight_quant,
)


def _gather_tokens(x, order):
    return x.gather(1, order[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])).contiguous()

def _mask_order(mask):
    order = torch.argsort((~mask).long(), dim=1, stable=True)
    inverse = torch.argsort(order, dim=1)
    return order, inverse, mask.sum(-1, dtype=torch.int32).contiguous()

def _fa4_raw_fwd(q,k,v,causal,scale,need_lse,seqused=None):
    if seqused is None or seqused.ndim == 1:
        return _base_fa4_fwd(q,k,v,causal,scale,need_lse,seqused)
    order,inverse,lengths=_mask_order(seqused)
    out,lse=_base_fa4_fwd(*[_gather_tokens(x,order) for x in (q,k,v)],causal,scale,need_lse,lengths)
    out=_gather_tokens(out,inverse).masked_fill(~seqused[:,:,None,None],0)
    return out,lse

def _fa4_raw_bwd(q,k,v,out,dout,lse,causal,scale,seqused=None):
    if seqused is None or seqused.ndim == 1:
        return _base_fa4_bwd(q,k,v,out,dout,lse,causal,scale,seqused)
    order,inverse,lengths=_mask_order(seqused)
    grads=_base_fa4_bwd(*[_gather_tokens(x,order) for x in (q,k,v,out,dout)],lse,causal,scale,lengths)
    return tuple(_gather_tokens(x,inverse).masked_fill(~seqused[:,:,None,None],0) for x in grads)

def _mmdit_block_infer_raw(
    act_c: torch.Tensor, txt_c: torch.Tensor, act_mod_3d: torch.Tensor, txt_mod_3d: torch.Tensor,
    nq_w: torch.Tensor, nk_w: torch.Tensor, naq_w: torch.Tensor, nak_w: torch.Tensor,
    act_cos: torch.Tensor, act_sin: torch.Tensor, txt_cos: torch.Tensor, txt_sin: torch.Tensor,
    seqused: torch.Tensor | None,
    wqkv_r, wqkv_s, bqkv, slot_qkv: int,
    wadd_r, wadd_s, badd, slot_add: int,
    wo_r, wo_s, bo, slot_o: int,
    wao_r, wao_s, bao, slot_ao: int,
    wa1_r, wa1_s, ba1, slot_a1: int,
    wa2_r, wa2_s, ba2,
    wt1_r, wt1_s, bt1, slot_t1: int,
    wt2_r, wt2_s, bt2,
    H: int, eps: float, is_last: bool,
):
    B, S_act, _ = act_c.shape
    S_txt = txt_c.shape[1]

    am1_e4, _, _, _, am1_s = adaln_modulate_quant_1536_slot(act_c, act_mod_3d, 0, eps, slot_qkv * 5 + 0, dual=False, save_stats=False)
    tm1_e4, _, _, _, tm1_s = adaln_modulate_quant_1536_slot(txt_c, txt_mod_3d, 0, eps, slot_add * 5 + 0, dual=False, save_stats=False)

    act_qkv = lt_fp8_mm(am1_e4, wqkv_r, am1_s, wqkv_s, bias=bqkv)
    txt_qkv = lt_fp8_mm(tm1_e4, wadd_r, tm1_s, wadd_s, bias=badd)
    del am1_e4, tm1_e4

    use_hd48 = (seqused is None)
    jq, jk, jv, _ = _mmdit_joint_qkv_raw_fwd(
        act_qkv, txt_qkv, nq_w, nk_w, naq_w, nak_w, act_cos, act_sin, txt_cos, txt_sin,
        B, S_act, S_txt, H, eps, save_rstd=False, hd48=use_hd48,
    )
    del act_qkv, txt_qkv

    if use_hd48:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            j_raw = F.scaled_dot_product_attention(
                jq.transpose(1, 2), jk.transpose(1, 2), jv.transpose(1, 2), scale=48 ** -0.5
            )
        del jq, jk, jv
    else:
        j_raw, _ = _fa4_raw_fwd(jq, jk, jv, causal=False, scale=48 ** -0.5, need_lse=False, seqused=seqused)
        del jq, jk, jv

    aa_e4, _, aa_s, ta_e4, _, ta_s = _mmdit_split_quant_fwd(
        j_raw, B, S_act, S_txt, is_last, slot_o * 5 + 0, slot_ao * 5 + 0, dual=False, hd48=use_hd48
    )
    del j_raw

    act_attn_out = lt_fp8_mm(aa_e4, wo_r, aa_s, wo_s, bias=bo)
    del aa_e4
    act_mid = gated_add_fwd_1536_cuda(act_c, act_attn_out, act_mod_3d, 0)
    del act_attn_out

    am2_e4, _, _, _, am2_s = adaln_modulate_quant_1536_slot(act_mid, act_mod_3d, 4608, eps, slot_a1 * 5 + 0, dual=False, save_stats=False)
    za = lt_fp8_mm(am2_e4, wa1_r, am2_s, wa1_s)
    del am2_e4
    ha_e4, ha_s = gelu_bias_quant_1d_slot(za, ba1, slot_a1 * 5 + 2)
    del za
    act_mlp_out = lt_fp8_mm(ha_e4, wa2_r, ha_s, wa2_s, bias=ba2)
    del ha_e4
    act_out = gated_add_fwd_1536_cuda(act_mid, act_mlp_out, act_mod_3d, 4608)
    del act_mid, act_mlp_out

    if is_last:
        return None, act_out
    txt_attn_out = lt_fp8_mm(ta_e4, wao_r, ta_s, wao_s, bias=bao)
    del ta_e4
    txt_mid = gated_add_fwd_1536_cuda(txt_c, txt_attn_out, txt_mod_3d, 0)
    del txt_attn_out

    tm2_e4, _, _, _, tm2_s = adaln_modulate_quant_1536_slot(txt_mid, txt_mod_3d, 4608, eps, slot_t1 * 5 + 0, dual=False, save_stats=False)
    zt = lt_fp8_mm(tm2_e4, wt1_r, tm2_s, wt1_s)
    del tm2_e4
    ht_e4, ht_s = gelu_bias_quant_1d_slot(zt, bt1, slot_t1 * 5 + 2)
    del zt
    txt_mlp_out = lt_fp8_mm(ht_e4, wt2_r, ht_s, wt2_s, bias=bt2)
    del ht_e4
    txt_out = gated_add_fwd_1536_cuda(txt_mid, txt_mlp_out, txt_mod_3d, 4608)
    del txt_mid, txt_mlp_out
    return txt_out, act_out


class _MMDiTBlockCkptFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        act_in: torch.Tensor,
        txt_in: torch.Tensor,
        act_mod_in: torch.Tensor,
        txt_mod_in: torch.Tensor,
        # Trainable parameters (24 tensors; on last block wao/bao/wt1/bt1/wt2/bt2 are raw dead params)
        wqkv_p, bqkv_p,
        wadd_p, badd_p,
        nq_p, nk_p, naq_p, nak_p,
        wo_p, bo_p,
        wao_p, bao_p,
        wa1_p, ba1_p, wa2_p, ba2_p,
        wt1_p, bt1_p, wt2_p, bt2_p,
        # Non-diff RoPE & seqused & cached FP8 weights + slot meta
        act_cos, act_sin, txt_cos, txt_sin, seqused,
        wqkv_r, wqkv_t, wqkv_s, slot_qkv: int,
        wadd_r, wadd_t, wadd_s, slot_add: int,
        wo_r, wo_t, wo_s, slot_o: int,
        wao_r, wao_t, wao_s, slot_ao: int,
        wa1_r, wa1_t, wa1_s, slot_a1: int,
        wa2_r, wa2_t, wa2_s, slot_a2: int,
        wt1_r, wt1_t, wt1_s, slot_t1: int,
        wt2_r, wt2_t, wt2_s, slot_t2: int,
        H: int, eps: float, bwd_fast_accum: bool, is_last: bool,
    ):
        ctx.set_materialize_grads(False)
        ctx.act_dtype = act_in.dtype
        ctx.txt_dtype = txt_in.dtype
        ctx.act_mod_shape = act_mod_in.shape
        ctx.txt_mod_shape = txt_mod_in.shape
        ctx.act_mod_dtype = act_mod_in.dtype
        ctx.txt_mod_dtype = txt_mod_in.dtype
        ctx.nq_dt = nq_p.dtype
        ctx.slots = (slot_qkv, slot_add, slot_o, slot_ao, slot_a1, slot_a2, slot_t1, slot_t2)
        ctx.H = H
        ctx.eps = eps
        ctx.bfa = bwd_fast_accum
        ctx.is_last = is_last
        ctx.has_seqused = seqused is not None
        ctx.all_params = (
            wqkv_p, bqkv_p, wadd_p, badd_p, nq_p, nk_p, naq_p, nak_p,
            wo_p, bo_p, wao_p, bao_p, wa1_p, ba1_p, wa2_p, ba2_p, wt1_p, bt1_p, wt2_p, bt2_p,
        )
        ctx.dw_params = (
            (wa2_p, wa1_p, wo_p, wqkv_p, wadd_p)
            if is_last
            else (wa2_p, wa1_p, wt2_p, wt1_p, wo_p, wao_p, wqkv_p, wadd_p)
        )

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            act_c = _as_bf16_contig(act_in)
            txt_c = _as_bf16_contig(txt_in)
            B = act_c.shape[0]
            act_mod_3d = _as_bf16_contig(act_mod_in).view(B, -1, 9216)
            txt_mod_3d = _as_bf16_contig(txt_mod_in).view(B, -1, 9216)
            nq_c, nk_c = _as_bf16_contig(nq_p), _as_bf16_contig(nk_p)
            naq_c, nak_c = _as_bf16_contig(naq_p), _as_bf16_contig(nak_p)
            bqkv_c, badd_c = _as_bf16_contig(bqkv_p), _as_bf16_contig(badd_p)
            bo_c = _as_bf16_contig(bo_p)
            ba1_c, ba2_c = _as_bf16_contig(ba1_p), _as_bf16_contig(ba2_p)
            bao_c = None if is_last else _as_bf16_contig(bao_p)
            bt1_c = None if is_last else _as_bf16_contig(bt1_p)
            bt2_c = None if is_last else _as_bf16_contig(bt2_p)

            txt_out, act_out = _mmdit_block_infer_raw(
                act_c, txt_c, act_mod_3d, txt_mod_3d,
                nq_c, nk_c, naq_c, nak_c,
                act_cos, act_sin, txt_cos, txt_sin, seqused,
                wqkv_r, wqkv_s, bqkv_c, slot_qkv,
                wadd_r, wadd_s, badd_c, slot_add,
                wo_r, wo_s, bo_c, slot_o,
                wao_r, wao_s, bao_c, slot_ao,
                wa1_r, wa1_s, ba1_c, slot_a1,
                wa2_r, wa2_s, ba2_c,
                wt1_r, wt1_s, bt1_c, slot_t1,
                wt2_r, wt2_s, bt2_c,
                H, eps, is_last,
            )

            saved = [
                act_c, txt_c, act_mod_3d, txt_mod_3d,
                nq_c, nk_c, naq_c, nak_c,
                act_cos, act_sin, txt_cos, txt_sin,
                bqkv_c, badd_c, bo_c, ba1_c, ba2_c,
                wqkv_r, wqkv_t, wqkv_s,
                wadd_r, wadd_t, wadd_s,
                wo_r, wo_t, wo_s,
                wa1_r, wa1_t, wa1_s,
                wa2_r, wa2_t, wa2_s,
            ]
            if seqused is not None:
                saved.append(seqused)
            if is_last:
                saved.extend([wao_p, bao_p, wt1_p, bt1_p, wt2_p, bt2_p])
            else:
                saved.extend([
                    bao_c, bt1_c, bt2_c,
                    wao_r, wao_t, wao_s,
                    wt1_r, wt1_t, wt1_s,
                    wt2_r, wt2_t, wt2_s,
                ])
            ctx.save_for_backward(*saved)

            if is_last:
                dummy_txt = act_out.new_empty((0,))
                ctx.mark_non_differentiable(dummy_txt)
                return dummy_txt, act_out.to(ctx.act_dtype)
            return txt_out.to(ctx.txt_dtype), act_out.to(ctx.act_dtype)

    @staticmethod
    def backward(ctx, d_txt_in, d_act_in):
        if d_act_in is None and d_txt_in is None:
            return (None,) * 65
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            st = list(ctx.saved_tensors)
            (
                act_c, txt_c, act_mod_3d, txt_mod_3d,
                nq_c, nk_c, naq_c, nak_c,
                act_cos, act_sin, txt_cos, txt_sin,
                bqkv_c, badd_c, bo_c, ba1_c, ba2_c,
                wqkv_r, wqkv_t, wqkv_s,
                wadd_r, wadd_t, wadd_s,
                wo_r, wo_t, wo_s,
                wa1_r, wa1_t, wa1_s,
                wa2_r, wa2_t, wa2_s,
            ) = st[:32]
            idx = 32
            seqused = None
            if ctx.has_seqused:
                seqused = st[idx]
                idx += 1
            is_last = ctx.is_last
            if is_last:
                wao_p, bao_p, wt1_p, bt1_p, wt2_p, bt2_p = st[idx:]
            else:
                (
                    bao_c, bt1_c, bt2_c,
                    wao_r, wao_t, wao_s,
                    wt1_r, wt1_t, wt1_s,
                    wt2_r, wt2_t, wt2_s,
                ) = st[idx:]

            slot_qkv, slot_add, slot_o, slot_ao, slot_a1, slot_a2, slot_t1, slot_t2 = ctx.slots
            B, S_act, _ = act_c.shape
            S_txt = txt_c.shape[1]
            dev = act_c.device
            bfa = ctx.bfa
            eps = ctx.eps
            H = ctx.H

            # --- Recompute Forward #2 ---
            am1_e4, am1_t, am1_mean, am1_rstd, am1_s = adaln_modulate_quant_1536_slot(
                act_c, act_mod_3d, 0, eps, slot_qkv * 5 + 0, dual=True, save_stats=True
            )
            tm1_e4, tm1_t, tm1_mean, tm1_rstd, tm1_s = adaln_modulate_quant_1536_slot(
                txt_c, txt_mod_3d, 0, eps, slot_add * 5 + 0, dual=True, save_stats=True
            )
            act_qkv = lt_fp8_mm(am1_e4, wqkv_r, am1_s, wqkv_s, bias=bqkv_c)
            txt_qkv = lt_fp8_mm(tm1_e4, wadd_r, tm1_s, wadd_s, bias=badd_c)
            del am1_e4, tm1_e4

            use_hd48 = (seqused is None)
            jq, jk, jv, jqkv_rstd = _mmdit_joint_qkv_raw_fwd(
                act_qkv, txt_qkv, nq_c, nk_c, naq_c, nak_c,
                act_cos, act_sin, txt_cos, txt_sin, B, S_act, S_txt, H, eps, save_rstd=True, hd48=use_hd48,
            )
            if use_hd48:
                jq_t = jq.transpose(1, 2).detach().requires_grad_(True)
                jk_t = jk.transpose(1, 2).detach().requires_grad_(True)
                jv_t = jv.transpose(1, 2).detach().requires_grad_(True)
                del jq, jk, jv
                with torch.enable_grad(), sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    j_bh = F.scaled_dot_product_attention(jq_t, jk_t, jv_t, scale=48 ** -0.5)
                aa_e4, aa_t, aa_s, ta_e4, ta_t, ta_s = _mmdit_split_quant_fwd(
                    j_bh.detach(), B, S_act, S_txt, is_last, slot_o * 5 + 0, slot_ao * 5 + 0, dual=True, hd48=True
                )
            else:
                j_out, lse = _fa4_raw_fwd(jq, jk, jv, causal=False, scale=48 ** -0.5, need_lse=True, seqused=seqused)
                aa_e4, aa_t, aa_s, ta_e4, ta_t, ta_s = _mmdit_split_quant_fwd(
                    j_out, B, S_act, S_txt, is_last, slot_o * 5 + 0, slot_ao * 5 + 0, dual=True, hd48=False
                )

            act_attn_out = lt_fp8_mm(aa_e4, wo_r, aa_s, wo_s, bias=bo_c)
            del aa_e4
            act_mid = gated_add_fwd_1536_cuda(act_c, act_attn_out, act_mod_3d, 0)

            am2_e4, am2_t, am2_mean, am2_rstd, am2_s = adaln_modulate_quant_1536_slot(
                act_mid, act_mod_3d, 4608, eps, slot_a1 * 5 + 0, dual=True, save_stats=True
            )
            za = lt_fp8_mm(am2_e4, wa1_r, am2_s, wa1_s)
            del am2_e4
            zba, ha_e4, ha_t, ha_s = gelu_bias_quant_dual_slot(za, ba1_c, slot_a1 * 5 + 2)
            del za
            act_mlp_out = lt_fp8_mm(ha_e4, wa2_r, ha_s, wa2_s, bias=ba2_c)
            del ha_e4

            if not is_last:
                txt_attn_out = lt_fp8_mm(ta_e4, wao_r, ta_s, wao_s, bias=bao_c)
                del ta_e4
                txt_mid = gated_add_fwd_1536_cuda(txt_c, txt_attn_out, txt_mod_3d, 0)

                tm2_e4, tm2_t, tm2_mean, tm2_rstd, tm2_s = adaln_modulate_quant_1536_slot(
                    txt_mid, txt_mod_3d, 4608, eps, slot_t1 * 5 + 0, dual=True, save_stats=True
                )
                zt = lt_fp8_mm(tm2_e4, wt1_r, tm2_s, wt1_s)
                del tm2_e4
                zbt, ht_e4, ht_t, ht_s = gelu_bias_quant_dual_slot(zt, bt1_c, slot_t1 * 5 + 2)
                del zt
                txt_mlp_out = lt_fp8_mm(ht_e4, wt2_r, ht_s, wt2_s, bias=bt2_c)
                del ht_e4

            # --- Backward Pass ---
            ipg_alloc = try_begin_direct_ipg_block(dev, ctx.all_params, ctx.dw_params)
            if is_last:
                wa2_p_d, wa1_p_d, wo_p_d, wqkv_p_d, wadd_p_d = ctx.dw_params
                wt2_p_d = wt1_p_d = wao_p_d = None
            else:
                wa2_p_d, wa1_p_d, wt2_p_d, wt1_p_d, wo_p_d, wao_p_d, wqkv_p_d, wadd_p_d = ctx.dw_params

            d_act_mod_f32 = torch.zeros_like(act_mod_3d, dtype=torch.float32)
            d_txt_mod_f32 = torch.zeros_like(txt_mod_3d, dtype=torch.float32)

            d_act_c = _as_bf16_contig(d_act_in) if d_act_in is not None else torch.zeros_like(act_c)
            # 1. Action MLP & norm2 backward
            dam_row, dam_t, dam_s, d_ba2 = gated_bwd_quant_e5_1536_slot(
                d_act_c, act_mlp_out, act_mod_3d, 4608, d_act_mod_f32, slot_a2 * 5 + 1, need_dbias=True
            )
            del act_mlp_out
            dha = lt_fp8_mm(dam_row, wa2_t, dam_s, wa2_s, fast_accum=bfa, is_bwd=True)
            d_wa2 = lt_fp8_mm(dam_t, ha_t, dam_s, ha_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wa2_p_d) if ipg_alloc is not None else None)
            del dam_row, dam_t, ha_t
            dza_row, dza_t, d_ba1, dza_s = dgelu_quant_dual_slot(dha, zba, torch.bfloat16, slot_a1 * 5 + 3)
            del dha, zba
            d_am2 = lt_fp8_mm(dza_row, wa1_t, dza_s, wa1_s, fast_accum=bfa, is_bwd=True)
            d_wa1 = lt_fp8_mm(dza_t, am2_t, dza_s, am2_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wa1_p_d) if ipg_alloc is not None else None)
            del dza_row, dza_t, am2_t

            d_act_mid = adaln_modulate_bwd_1536_cuda(
                d_am2, d_act_c, act_mid, act_mod_3d, 4608, am2_mean, am2_rstd, d_act_mod_f32
            )
            del d_am2, d_act_c, act_mid, am2_mean, am2_rstd

            # 2. Text MLP & norm2 backward (if not last block)
            if not is_last:
                d_txt_c = _as_bf16_contig(d_txt_in) if d_txt_in is not None else torch.zeros_like(txt_c)
                dtm_row, dtm_t, dtm_s, d_bt2 = gated_bwd_quant_e5_1536_slot(
                    d_txt_c, txt_mlp_out, txt_mod_3d, 4608, d_txt_mod_f32, slot_t2 * 5 + 1, need_dbias=True
                )
                del txt_mlp_out
                dht = lt_fp8_mm(dtm_row, wt2_t, dtm_s, wt2_s, fast_accum=bfa, is_bwd=True)
                d_wt2 = lt_fp8_mm(dtm_t, ht_t, dtm_s, ht_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wt2_p_d) if ipg_alloc is not None else None)
                del dtm_row, dtm_t, ht_t
                dzt_row, dzt_t, d_bt1, dzt_s = dgelu_quant_dual_slot(dht, zbt, torch.bfloat16, slot_t1 * 5 + 3)
                del dht, zbt
                d_tm2 = lt_fp8_mm(dzt_row, wt1_t, dzt_s, wt1_s, fast_accum=bfa, is_bwd=True)
                d_wt1 = lt_fp8_mm(dzt_t, tm2_t, dzt_s, tm2_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wt1_p_d) if ipg_alloc is not None else None)
                del dzt_row, dzt_t, tm2_t

                d_txt_mid = adaln_modulate_bwd_1536_cuda(
                    d_tm2, d_txt_c, txt_mid, txt_mod_3d, 4608, tm2_mean, tm2_rstd, d_txt_mod_f32
                )
                del d_tm2, d_txt_c, txt_mid, tm2_mean, tm2_rstd
            else:
                d_wao = torch.zeros_like(wao_p)
                d_bao = torch.zeros_like(bao_p)
                d_wt1 = torch.zeros_like(wt1_p)
                d_bt1 = torch.zeros_like(bt1_p)
                d_wt2 = torch.zeros_like(wt2_p)
                d_bt2 = torch.zeros_like(bt2_p)

            # 3. Output projections (`to_out[0]` & `to_add_out`) backward
            dao_row, dao_t, dao_s, d_bo = gated_bwd_quant_e5_1536_slot(
                d_act_mid, act_attn_out, act_mod_3d, 0, d_act_mod_f32, slot_o * 5 + 1, need_dbias=True
            )
            del act_attn_out
            d_act_attn = lt_fp8_mm(dao_row, wo_t, dao_s, wo_s, fast_accum=bfa, is_bwd=True)
            d_wo = lt_fp8_mm(dao_t, aa_t, dao_s, aa_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wo_p_d) if ipg_alloc is not None else None)
            del dao_row, dao_t, aa_t

            if not is_last:
                dtao_row, dtao_t, dtao_s, d_bao = gated_bwd_quant_e5_1536_slot(
                    d_txt_mid, txt_attn_out, txt_mod_3d, 0, d_txt_mod_f32, slot_ao * 5 + 1, need_dbias=True
                )
                del txt_attn_out
                d_txt_attn = lt_fp8_mm(dtao_row, wao_t, dtao_s, wao_s, fast_accum=bfa, is_bwd=True)
                d_wao = lt_fp8_mm(dtao_t, ta_t, dtao_s, ta_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wao_p_d) if ipg_alloc is not None else None)
                del dtao_row, dtao_t, ta_t
            else:
                d_txt_attn = None

            # 4. Split/Unpad + Attention + Joint QKV backward
            pool_qkv = get_pool(dev)
            if use_hd48 and pool_qkv.warmed_up == 1:
                dj_bh = _mmdit_split_48_bwd_bhsd(d_act_attn, d_txt_attn, B, S_act, S_txt, H)
                del d_act_attn, d_txt_attn
                djq_bh, djk_bh, djv_bh = torch.autograd.grad(
                    j_bh, (jq_t, jk_t, jv_t), dj_bh
                )
                del j_bh, jq_t, jk_t, jv_t, dj_bh
                jq_bhsd = (djq_bh.is_contiguous() and djk_bh.is_contiguous() and djv_bh.is_contiguous())
                if not jq_bhsd:
                    djq_bh, djk_bh, djv_bh = djq_bh.transpose(1, 2), djk_bh.transpose(1, 2), djv_bh.transpose(1, 2)
                (
                    daq_row, daq_t, daq_s, d_bqkv,
                    dtq_row, dtq_t, dtq_s, d_badd,
                    dnq, dnk, dnaq, dnak,
                ) = _mmdit_joint_qkv_48_bwd_e5_dual(
                    djq_bh, djk_bh, djv_bh, act_qkv, txt_qkv, nq_c, nk_c, naq_c, nak_c,
                    act_cos, act_sin, txt_cos, txt_sin, jqkv_rstd, B, S_act, S_txt, H,
                    slot_qkv * 5 + 1, slot_add * 5 + 1, bhsd=jq_bhsd,
                )
                del djq_bh, djk_bh, djv_bh, act_qkv, txt_qkv, jqkv_rstd
            else:
                if use_hd48:
                    dj_bh = _mmdit_split_48_bwd_bhsd(d_act_attn, d_txt_attn, B, S_act, S_txt, H)
                    del d_act_attn, d_txt_attn
                    djq_bh, djk_bh, djv_bh = torch.autograd.grad(
                        j_bh, (jq_t, jk_t, jv_t), dj_bh
                    )
                    del j_bh, jq_t, jk_t, jv_t, dj_bh
                    dj_q, dj_k, dj_v = djq_bh.transpose(1, 2), djk_bh.transpose(1, 2), djv_bh.transpose(1, 2)
                else:
                    dj_out = _mmdit_split_unpad_raw_bwd(d_act_attn, d_txt_attn, B, S_act, S_txt, H, hd48=False)
                    del d_act_attn, d_txt_attn
                    dj_q, dj_k, dj_v = _fa4_raw_bwd(jq, jk, jv, j_out, dj_out, lse, causal=False, scale=48 ** -0.5, seqused=seqused)
                    del jq, jk, jv, j_out, lse, dj_out

                d_act_qkv, d_txt_qkv, dnq, dnk, dnaq, dnak = _mmdit_joint_qkv_raw_bwd(
                    dj_q, dj_k, dj_v, act_qkv, txt_qkv, nq_c, nk_c, naq_c, nak_c,
                    act_cos, act_sin, txt_cos, txt_sin, jqkv_rstd, B, S_act, S_txt, H, hd48=use_hd48,
                )
                del dj_q, dj_k, dj_v, act_qkv, txt_qkv, jqkv_rstd
                daq_row, daq_t, daq_s, d_bqkv = quantize_dual_slot(d_act_qkv, slot_qkv * 5 + 1, torch.float8_e5m2, need_dbias=True)
                del d_act_qkv
                dtq_row, dtq_t, dtq_s, d_badd = quantize_dual_slot(d_txt_qkv, slot_add * 5 + 1, torch.float8_e5m2, need_dbias=True)
                del d_txt_qkv

            d_am1 = lt_fp8_mm(daq_row, wqkv_t, daq_s, wqkv_s, fast_accum=bfa, is_bwd=True)
            d_wqkv = lt_fp8_mm(daq_t, am1_t, daq_s, am1_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wqkv_p_d) if ipg_alloc is not None else None)
            del daq_row, daq_t, am1_t
            d_tm1 = lt_fp8_mm(dtq_row, wadd_t, dtq_s, wadd_s, fast_accum=bfa, is_bwd=True)
            d_wadd = lt_fp8_mm(dtq_t, tm1_t, dtq_s, tm1_s, fast_accum=bfa, is_bwd=True, out=ipg_alloc(wadd_p_d) if ipg_alloc is not None else None)
            del dtq_row, dtq_t, tm1_t
            if ipg_alloc is not None:
                d_wa2 = d_wa1 = d_wo = d_wqkv = d_wadd = None
                if not is_last:
                    d_wt2 = d_wt1 = d_wao = None

            # 5. norm1 + residual backward
            d_act_in_out = adaln_modulate_bwd_1536_cuda(
                d_am1, d_act_mid, act_c, act_mod_3d, 0, am1_mean, am1_rstd, d_act_mod_f32
            )
            del d_am1, d_act_mid, am1_mean, am1_rstd

            d_txt_in_out = adaln_modulate_bwd_1536_cuda(
                d_tm1, None if is_last else d_txt_mid, txt_c, txt_mod_3d, 0, tm1_mean, tm1_rstd, d_txt_mod_f32
            )
            del d_tm1, tm1_mean, tm1_rstd

            d_act_mod_bf16 = torch.empty_like(act_mod_3d)
            d_txt_mod_bf16 = torch.empty_like(txt_mod_3d)
            get_cuda_lib().launch_cast_f32_to_bf16(_ptr(d_act_mod_f32), _ptr(d_act_mod_bf16), d_act_mod_f32.numel(), _stream(dev))
            get_cuda_lib().launch_cast_f32_to_bf16(_ptr(d_txt_mod_f32), _ptr(d_txt_mod_bf16), d_txt_mod_f32.numel(), _stream(dev))

            dt_n = ctx.nq_dt
            return (
                d_act_in_out.to(ctx.act_dtype),
                d_txt_in_out.to(ctx.txt_dtype),
                d_act_mod_bf16.view(ctx.act_mod_shape).to(ctx.act_mod_dtype),
                d_txt_mod_bf16.view(ctx.txt_mod_shape).to(ctx.txt_mod_dtype),
                d_wqkv, d_bqkv,
                d_wadd, d_badd,
                dnq.to(dt_n), dnk.to(dt_n), dnaq.to(dt_n), dnak.to(dt_n),
                d_wo, d_bo,
                d_wao, d_bao,
                d_wa1, d_ba1, d_wa2, d_ba2,
                d_wt1, d_bt1, d_wt2, d_bt2,
            ) + (None,) * 41


def run_mmdit_block(
    blk, act: torch.Tensor, txt: torch.Tensor, act_mod: torch.Tensor, txt_mod: torch.Tensor,
    rope_4tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    seqused: torch.Tensor | None, *, use_gc: bool, is_last: bool,
):
    a = blk.attn
    act_cos, act_sin, txt_cos, txt_sin = rope_4tuple
    H = a.heads
    eps = float(getattr(blk.action_norm1, "eps", 1e-6))

    wqkv_r, wqkv_t, wqkv_s = a.to_qkv.get_or_quant_weight()
    wadd_r, wadd_t, wadd_s = a.add_qkv.get_or_quant_weight()
    wo_r, wo_t, wo_s = a.to_out[0].get_or_quant_weight()
    wa1 = blk.action_mlp.net[0].proj
    wa2 = blk.action_mlp.net[2]
    wa1_r, wa1_t, wa1_s = wa1.get_or_quant_weight()
    wa2_r, wa2_t, wa2_s = wa2.get_or_quant_weight()

    if is_last:
        wao_p, bao_p = a.to_add_out.weight, a.to_add_out.bias
        wt1_p, bt1_p = blk.txt_mlp.net[0].proj.weight, blk.txt_mlp.net[0].proj.bias
        wt2_p, bt2_p = blk.txt_mlp.net[2].weight, blk.txt_mlp.net[2].bias
        wao_r = wao_t = wao_s = wt1_r = wt1_t = wt1_s = wt2_r = wt2_t = wt2_s = None
        slot_ao = slot_t1 = slot_t2 = 0
    else:
        wao_p, bao_p = a.to_add_out.weight, a.to_add_out.bias
        wt1 = blk.txt_mlp.net[0].proj
        wt2 = blk.txt_mlp.net[2]
        wt1_p, bt1_p = wt1.weight, wt1.bias
        wt2_p, bt2_p = wt2.weight, wt2.bias
        wao_r, wao_t, wao_s = a.to_add_out.get_or_quant_weight()
        wt1_r, wt1_t, wt1_s = wt1.get_or_quant_weight()
        wt2_r, wt2_t, wt2_s = wt2.get_or_quant_weight()
        slot_ao, slot_t1, slot_t2 = a.to_add_out.slot_id, wt1.slot_id, wt2.slot_id

    if not torch.is_grad_enabled():
        B = act.shape[0]
        txt_o, act_o = _mmdit_block_infer_raw(
            _as_bf16_contig(act), _as_bf16_contig(txt),
            _as_bf16_contig(act_mod).view(B, -1, 9216), _as_bf16_contig(txt_mod).view(B, -1, 9216),
            _as_bf16_contig(a.norm_q.weight), _as_bf16_contig(a.norm_k.weight),
            _as_bf16_contig(a.norm_added_q.weight), _as_bf16_contig(a.norm_added_k.weight),
            act_cos, act_sin, txt_cos, txt_sin, seqused,
            wqkv_r, wqkv_s, _as_bf16_contig(a.to_qkv.bias), a.to_qkv.slot_id,
            wadd_r, wadd_s, _as_bf16_contig(a.add_qkv.bias), a.add_qkv.slot_id,
            wo_r, wo_s, _as_bf16_contig(a.to_out[0].bias), a.to_out[0].slot_id,
            wao_r, wao_s, None if is_last else _as_bf16_contig(bao_p), slot_ao,
            wa1_r, wa1_s, _as_bf16_contig(wa1.bias), wa1.slot_id,
            wa2_r, wa2_s, _as_bf16_contig(wa2.bias),
            wt1_r, wt1_s, None if is_last else _as_bf16_contig(bt1_p), slot_t1,
            wt2_r, wt2_s, None if is_last else _as_bf16_contig(bt2_p),
            H, eps, is_last,
        )
        return (None, act_o.to(act.dtype)) if is_last else (txt_o.to(txt.dtype), act_o.to(act.dtype))

    txt_o, act_o = _MMDiTBlockCkptFn.apply(
        act, txt, act_mod, txt_mod,
        a.to_qkv.weight, a.to_qkv.bias,
        a.add_qkv.weight, a.add_qkv.bias,
        a.norm_q.weight, a.norm_k.weight, a.norm_added_q.weight, a.norm_added_k.weight,
        a.to_out[0].weight, a.to_out[0].bias,
        wao_p, bao_p,
        wa1.weight, wa1.bias, wa2.weight, wa2.bias,
        wt1_p, bt1_p, wt2_p, bt2_p,
        act_cos, act_sin, txt_cos, txt_sin, seqused,
        wqkv_r, wqkv_t, wqkv_s, a.to_qkv.slot_id,
        wadd_r, wadd_t, wadd_s, a.add_qkv.slot_id,
        wo_r, wo_t, wo_s, a.to_out[0].slot_id,
        wao_r, wao_t, wao_s, slot_ao,
        wa1_r, wa1_t, wa1_s, wa1.slot_id,
        wa2_r, wa2_t, wa2_s, wa2.slot_id,
        wt1_r, wt1_t, wt1_s, slot_t1,
        wt2_r, wt2_t, wt2_s, slot_t2,
        H, eps, a.to_qkv.bwd_fast_accum, is_last,
    )
    return (None, act_o) if is_last else (txt_o, act_o)


def fast_mmdit_model_forward(
    dit_self, hidden_states, encoder_hidden_states, timestep=None,
    encoder_attention_mask=None, return_all_hidden_states=False, vlm_intermediate_hidden_states=None, action_attention_mask=None
):
    wait_fp8_weight_quant(hidden_states.device)
    temb = dit_self.timestep_encoder(timestep)
    txt_temb = temb[:, -1] if temb.dim() == 3 else temb

    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = dit_self.txt_in(dit_self.txt_norm(encoder_hidden_states.contiguous()))
    seq_txt = encoder_hidden_states.shape[1]
    seq_action = hidden_states.shape[1]
    fa, ft = dit_self.rope(seq_txt, seq_action, hidden_states.device)
    rope_4tuple = (
        fa[0].float().contiguous(), fa[1].float().contiguous(),
        ft[0].float().contiguous(), ft[1].float().contiguous(),
    )

    seqused = None
    if encoder_attention_mask is not None:
        seqused = (seq_action + encoder_attention_mask.sum(-1, dtype=torch.int32)).contiguous()

    if action_attention_mask is not None:
        text_valid = torch.ones(encoder_hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool) if encoder_attention_mask is None else encoder_attention_mask.bool()
        seqused = torch.cat((action_attention_mask, text_valid), dim=1).contiguous()

    act_silu = F.silu(temb)
    txt_silu = F.silu(txt_temb)
    act_mods = [blk.action_mod[1](act_silu) for blk in dit_self.transformer_blocks]
    txt_mods = [blk.txt_mod[1](txt_silu) for blk in dit_self.transformer_blocks]

    all_hidden_states = [hidden_states] if return_all_hidden_states else None
    use_gc = dit_self.training and dit_self.gradient_checkpointing
    if use_gc and not hidden_states.requires_grad:
        hidden_states = hidden_states.requires_grad_(True)
    num_blks = len(dit_self.transformer_blocks)

    for idx, block in enumerate(dit_self.transformer_blocks):
        is_last_skip = getattr(block, "_single_out", False) and (idx == num_blks - 1)
        encoder_hidden_states, hidden_states = run_mmdit_block(
            block, hidden_states, encoder_hidden_states, act_mods[idx], txt_mods[idx],
            rope_4tuple, seqused, use_gc=use_gc, is_last=is_last_skip,
        )
        if return_all_hidden_states:
            all_hidden_states.append(hidden_states)

    shift, scale = dit_self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
    if scale.dim() == 2:
        shift, scale = shift[:, None], scale[:, None]
    hidden_states = dit_self.norm_out(hidden_states) * (1 + scale) + shift
    out = dit_self.proj_out_2(hidden_states)
    return (out, all_hidden_states) if return_all_hidden_states else out
