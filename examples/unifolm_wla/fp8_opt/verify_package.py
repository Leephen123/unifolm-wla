"""Unit + v9-vs-v8 Block Timing + 2-Step Full-Model 4.15B Verifier for fp8_opt."""
import json
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from accelerate.utils import send_to_device

SHM_DIR = Path("/dev/shm/wla_equiv_v9_160153")


def run_unit_checks():
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def fast_fa4_dense_v9(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False, softmax_scale: float | None = None) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal, scale=softmax_scale
        ).transpose(1, 2)

    from examples.unifolm_wla.fp8_opt import StepCachedFP8Linear
    from examples.unifolm_wla.fp8_opt.cuda_ops import (
        begin_fp8_step,
        reset_fp8_pool,
        sum_rows_bf16_cuda,
    )
    from examples.unifolm_wla.fp8_opt.vision_tower import run_vision_block as _v9_vision_block_run
    from examples.unifolm_wla.fp8_opt.language_backbone import run_qwen_decoder_layer as _v9_qwen_decoder_layer_run

    torch.manual_seed(42)
    dev = torch.device("cuda:0")

    # 1. Verify sum_rows_bf16_cuda on odd & vec64 shapes
    for M, N in ((3360, 4096), (3360, 3072), (587, 1540)):
        x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        s = sum_rows_bf16_cuda(x)
        sr = x.float().sum(0).bfloat16()
        assert float((s.float() - sr.float()).norm() / sr.float().norm()) < 0.002

    # 2. Vision Block Zero-BgradAB_v2 + Mode 5 + sum_rows_bf16_cuda vs Canonical BF16 Reference
    class _DummyVisAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 16
            self.scaling = 64 ** -0.5
            self.qkv = nn.Linear(1024, 3072, bias=True)
            self.proj = nn.Linear(1024, 1024, bias=True)

    class _DummyVisMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = nn.Linear(1024, 4096, bias=True)
            self.linear_fc2 = nn.Linear(4096, 1024, bias=True)

    class _DummyVisBlk(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(1024, eps=1e-6)
            self.attn = _DummyVisAttn()
            self.norm2 = nn.LayerNorm(1024, eps=1e-6)
            self.mlp = _DummyVisMLP()

    M_vis = 24 * 560  # 1/8th of BS64 (8 cams * 3) to fit comfortably on overlapped GPU
    vblk = _DummyVisBlk().to(dev).bfloat16()
    vcos = torch.randn(560, 64, device=dev, dtype=torch.float32)
    vsin = torch.randn(560, 64, device=dev, dtype=torch.float32)
    vx = torch.randn(M_vis, 1024, device=dev, dtype=torch.bfloat16, requires_grad=True)
    vdout = torch.randn(M_vis, 1024, device=dev, dtype=torch.bfloat16)

    vx_ref = vx.detach().clone().requires_grad_(True)
    vblk_ref = _DummyVisBlk().to(dev).bfloat16()
    vblk_ref.load_state_dict(vblk.state_dict())
    vqkv_r = vblk_ref.attn.qkv(vblk_ref.norm1(vx_ref))
    vqr, vkr, vvr = vqkv_r.view(24, 560, 3, 16, 64).unbind(2)
    vqr = ((vqr.float() * vcos[None, :, None, :]) + (_rotate_half(vqr.float()) * vsin[None, :, None, :])).bfloat16()
    vkr = ((vkr.float() * vcos[None, :, None, :]) + (_rotate_half(vkr.float()) * vsin[None, :, None, :])).bfloat16()
    va_r = fast_fa4_dense_v9(vqr, vkr, vvr, causal=False, softmax_scale=0.125).reshape(M_vis, 1024)
    vh_mid_r = vx_ref + vblk_ref.attn.proj(va_r)
    vout_r = vh_mid_r + vblk_ref.mlp.linear_fc2(F.gelu(vblk_ref.mlp.linear_fc1(vblk_ref.norm2(vh_mid_r)), approximate="tanh"))
    vout_r.backward(vdout)

    vblk.zero_grad(set_to_none=True)
    vx_t = vx.detach().clone().requires_grad_(True)
    vout_t = _v9_vision_block_run(vblk, vx_t, cos_f32=vcos, sin_f32=vsin, use_gc=True, save_attn=False)
    vout_t.backward(vdout)
    for (n, p), (_, pr) in zip(vblk.named_parameters(), vblk_ref.named_parameters()):
        rel = float((p.grad.float() - pr.grad.float()).norm() / pr.grad.float().norm())
        assert rel < 0.02, (n, rel)

    # 3. Qwen Layer End-to-End Check with cuDNN 9.12 zero-copy bshd transpose(1,2) across steps 1 & 2
    class _DummyQwenAttn(nn.Module):
        def __init__(self, s0, s1):
            super().__init__()
            self.head_dim = 128
            self.scaling = 128 ** -0.5
            self.qkv_sizes = [4096, 1024, 1024]
            self.qkv_proj = StepCachedFP8Linear(2560, 6144, slot_id=s0)
            self.q_norm = qwen.Qwen3VLTextRMSNorm(128, eps=1e-6)
            self.k_norm = qwen.Qwen3VLTextRMSNorm(128, eps=1e-6)
            self.o_proj = StepCachedFP8Linear(4096, 2560, slot_id=s1)

    class _DummyQwenMLP(nn.Module):
        def __init__(self, s2, s3):
            super().__init__()
            self.gate_up_proj = StepCachedFP8Linear(2560, 2 * 9728, slot_id=s2)
            self.down_proj = StepCachedFP8Linear(9728, 2560, slot_id=s3)

    class _DummyQwenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = qwen.Qwen3VLTextRMSNorm(2560, eps=1e-6)
            self.self_attn = _DummyQwenAttn(0, 1)
            self.post_attention_layernorm = qwen.Qwen3VLTextRMSNorm(2560, eps=1e-6)
            self.mlp = _DummyQwenMLP(2, 3)

    B, S = 8, 587
    qlayer = _DummyQwenLayer().to(dev).bfloat16()
    for m in qlayer.modules():
        if isinstance(m, StepCachedFP8Linear):
            nn.init.normal_(m.weight, std=0.02)
    reset_fp8_pool(dev)
    qmeta = [(s * 5 + i, 448.0 if i in (0, 2, 4) else 57344.0) for s in range(6) for i in range(5)]
    qx = torch.randn(B, S, 2560, device=dev, dtype=torch.bfloat16, requires_grad=True)
    qcos = torch.randn(B, S, 128, device=dev, dtype=torch.bfloat16)
    qsin = torch.randn(B, S, 128, device=dev, dtype=torch.bfloat16)
    qdout = torch.randn(B, S, 2560, device=dev, dtype=torch.bfloat16)

    qx_ref = qx.detach().clone().requires_grad_(True)
    qkv_r = F.linear(qlayer.input_layernorm(qx_ref), qlayer.self_attn.qkv_proj.weight.detach())
    qr, kr, vr = qkv_r.split([4096, 1024, 1024], dim=-1)
    qr = qlayer.self_attn.q_norm(qr.view(B, S, 32, 128)).transpose(1, 2)
    kr = qlayer.self_attn.k_norm(kr.view(B, S, 8, 128)).transpose(1, 2)
    vr = vr.view(B, S, 8, 128).transpose(1, 2)
    qr, kr = qwen.apply_rotary_pos_emb(qr, kr, qcos, qsin)
    ar = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True, scale=128 ** -0.5, enable_gqa=True).transpose(1, 2).reshape(B, S, 4096)
    hm_r = qx_ref + F.linear(ar, qlayer.self_attn.o_proj.weight.detach())
    gu_r = F.linear(qlayer.post_attention_layernorm(hm_r), qlayer.mlp.gate_up_proj.weight.detach())
    g_r, u_r = gu_r.chunk(2, dim=-1)
    qout_r = hm_r + F.linear(F.silu(g_r) * u_r, qlayer.mlp.down_proj.weight.detach())
    qout_r.backward(qdout)

    for step in (1, 2):
        begin_fp8_step(dev, qmeta if step == 1 else None)
        qlayer.zero_grad(set_to_none=True)
        for m in qlayer.modules():
            if isinstance(m, StepCachedFP8Linear):
                m.cache_step_weight()
        qx_t = qx.detach().clone().requires_grad_(True)
        with torch.no_grad():
            qo_inf = _v9_qwen_decoder_layer_run(qlayer, qx_t, cos_bf16=qcos, sin_bf16=qsin, use_gc=True, use_cudnn_sdpa=True)
        qo_tr = _v9_qwen_decoder_layer_run(qlayer, qx_t, cos_bf16=qcos, sin_bf16=qsin, use_gc=True, use_cudnn_sdpa=True)
        qo_tr.backward(qdout)
        assert float((qo_inf.float() - qo_tr.detach().float()).norm() / qo_tr.detach().float().norm()) < 1e-4
        err_qo = float((qo_tr.detach().float() - qout_r.detach().float()).norm() / qout_r.detach().float().norm())
        err_qdx = float((qx_t.grad.float() - qx_ref.grad.float()).norm() / qx_ref.grad.float().norm())
        assert err_qo < 0.09 and err_qdx < 0.16, (step, err_qo, err_qdx)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in qlayer.parameters())

    # 4. Verify v10 C++ _MMDiTBlockCkptFn (both full block & last block, RTC L=50 act_mod + L=1 txt_mod) vs BF16 Ref
    import copy
    from unifolm_wla.model.modules.action_model.DiT_modules.mmdit import QwenImageTransformerBlock, QwenActionTextRotaryEmbed
    from examples.unifolm_wla.fp8_opt import StepCachedFP8Linear
    from examples.unifolm_wla.fp8_opt.cuda_ops import FP8_E4M3_MAX, FP8_E5M2_MAX, begin_fp8_step, reset_fp8_pool
    from examples.unifolm_wla.fp8_opt.mmdit_expert import run_mmdit_block as _v10_mmdit_block_run
    reset_fp8_pool(dev)
    rope_gen = QwenActionTextRotaryEmbed(head_dim=48, theta=10000).to(dev)
    fa_r, ft_r = rope_gen(587, 50, dev)
    rope_4t = (fa_r[0].float().contiguous(), fa_r[1].float().contiguous(), ft_r[0].float().contiguous(), ft_r[1].float().contiguous())

    for is_last_blk in (False, True):
        ref_blk = QwenImageTransformerBlock(dim=1536, num_attention_heads=32, attention_head_dim=48).to(dev).bfloat16()
        opt_blk = copy.deepcopy(ref_blk)
        slots_meta = []
        sc = [0]
        def _ns():
            s = sc[0]; sc[0] += 1
            slots_meta.extend([(s*5+0, FP8_E4M3_MAX), (s*5+1, FP8_E5M2_MAX), (s*5+2, FP8_E4M3_MAX), (s*5+3, FP8_E5M2_MAX), (s*5+4, FP8_E4M3_MAX)])
            return s
        a = opt_blk.attn
        a.to_qkv = StepCachedFP8Linear.from_linears([a.to_q, a.to_k, a.to_v], _ns())
        del a.to_q, a.to_k, a.to_v
        a.add_qkv = StepCachedFP8Linear.from_linears([a.add_q_proj, a.add_k_proj, a.add_v_proj], _ns())
        del a.add_q_proj, a.add_k_proj, a.add_v_proj
        a.to_out[0] = StepCachedFP8Linear.from_linears([a.to_out[0]], _ns())
        if not is_last_blk:
            a.to_add_out = StepCachedFP8Linear.from_linears([a.to_add_out], _ns())
        opt_blk.action_mlp.net[0].proj = StepCachedFP8Linear.from_linears([opt_blk.action_mlp.net[0].proj], _ns())
        opt_blk.action_mlp.net[2] = StepCachedFP8Linear.from_linears([opt_blk.action_mlp.net[2]], _ns())
        if not is_last_blk:
            opt_blk.txt_mlp.net[0].proj = StepCachedFP8Linear.from_linears([opt_blk.txt_mlp.net[0].proj], _ns())
            opt_blk.txt_mlp.net[2] = StepCachedFP8Linear.from_linears([opt_blk.txt_mlp.net[2]], _ns())

        B_t = 4
        act0 = torch.randn(B_t, 50, 1536, device=dev, dtype=torch.bfloat16)
        txt0 = torch.randn(B_t, 587, 1536, device=dev, dtype=torch.bfloat16)
        temb0 = torch.randn(B_t, 50, 1536, device=dev, dtype=torch.bfloat16)
        ttemb0 = temb0[:, -1]

        # Ref forward/backward
        ar = act0.clone().requires_grad_(True)
        tr = txt0.clone().requires_grad_(True)
        amod_r = ref_blk.action_mod(temb0).detach().requires_grad_(True)
        tmod_r = ref_blk.txt_mod(ttemb0).detach().requires_grad_(True)
        am1, am2 = amod_r.chunk(2, dim=-1)
        tm1, tm2 = tmod_r.chunk(2, dim=-1)
        am_in, ag1 = ref_blk._modulate(ref_blk.action_norm1(ar), am1)
        tm_in, tg1 = ref_blk._modulate(ref_blk.txt_norm1(tr), tm1)
        aa_o, ta_o = ref_blk.attn(hidden_states=am_in, encoder_hidden_states=tm_in, image_rotary_emb=(fa_r, ft_r))
        ar_m = ar + ag1 * aa_o
        am_in2, ag2 = ref_blk._modulate(ref_blk.action_norm2(ar_m), am2)
        ar_o = ar_m + ag2 * ref_blk.action_mlp(am_in2)
        if not is_last_blk:
            tr_m = tr + tg1 * ta_o
            tm_in2, tg2 = ref_blk._modulate(ref_blk.txt_norm2(tr_m), tm2)
            tr_o = tr_m + tg2 * ref_blk.txt_mlp(tm_in2)
            ga = torch.randn_like(ar_o)
            gt = torch.randn_like(tr_o)
            loss_r = (ar_o * ga).sum() + (tr_o * gt).sum()
        else:
            ga = torch.randn_like(ar_o)
            loss_r = (ar_o * ga).sum()
        loss_r.backward()

        for step_t in (1, 2):
            opt_blk.zero_grad(set_to_none=True)
            reset_fp8_pool(dev) if step_t == 1 else None
            begin_fp8_step(dev, slots_meta)
            for m in opt_blk.modules():
                if isinstance(m, StepCachedFP8Linear):
                    m.cache_step_weight()
            ao = act0.clone().requires_grad_(True)
            to = txt0.clone().requires_grad_(True)
            amod_o = amod_r.detach().clone().requires_grad_(True)
            tmod_o = tmod_r.detach().clone().requires_grad_(True)
            to_out, ao_out = _v10_mmdit_block_run(
                opt_blk, ao, to, amod_o, tmod_o, rope_4t, None, use_gc=True, is_last=is_last_blk
            )
            loss_o = (ao_out * ga).sum() if is_last_blk else ((ao_out * ga).sum() + (to_out * gt).sum())
            loss_o.backward()
            err_act = float((ao_out.detach().float() - ar_o.detach().float()).norm() / ar_o.detach().float().norm())
            err_dact = float((ao.grad.float() - ar.grad.float()).norm() / ar.grad.float().norm())
            err_dtxt = float((to.grad.float() - tr.grad.float()).norm() / tr.grad.float().norm())
            err_damod = float((amod_o.grad.float() - amod_r.grad.float()).norm() / amod_r.grad.float().norm())
            assert err_act < 0.06 and err_dact < 0.08 and err_dtxt < (0.14 if is_last_blk else 0.08) and err_damod < 0.08, (is_last_blk, step_t, err_act, err_dact, err_dtxt, err_damod)

    del vblk, vblk_ref, qlayer, ref_blk, opt_blk
    reset_fp8_pool(dev)
    torch.cuda.empty_cache()
    print("[PASS] All v13 Vision + Qwen + MMDiT C++ _MMDiTBlockCkptFn (step1 & step2, full & last) checks OK!", flush=True)


def model_worker(rank: int):
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    from examples.unifolm_wla.pytree_cycle_fix import install_cycle_free_pytree
    install_cycle_free_pytree()

    from unifolm_wla.dataloader.multi_source_dataset.single_source_dataset import SingleSourceDataset
    from unifolm_wla.model.framework.base_framework import build_framework
    from examples.unifolm_wla.profile_and_hold import load_base
    from examples.unifolm_wla.training_worker_inputs import CPUInputsCollator, use_worker_inputs
    from examples.unifolm_wla.training_fusion import apply_candidate
    from examples.unifolm_wla.fp8_opt import (
        FastWLAInputsCollator,
        cache_all_fp8_weights,
        install_fp8_training_optimizations,
        install_zero_sync_forward,
    )

    cfg = OmegaConf.load("/storage/wangdonglinLab/weizhenyu/kuimouyu/research/unifoLM-WLA-fp8/runs/fp8-mfu-opt-20261118/base-config.yaml")
    cfg.datasets.vla_data.per_device_batch_size = 16
    ds = SingleSourceDataset(cfg.datasets.vla_data)
    raw_examples = [ds[i * 100] for i in range(16)]

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    m = build_framework(cfg)
    load_base(m, cfg.trainer.pretrained_checkpoint)
    m.to(dev, dtype=torch.bfloat16)
    m.qwen_vl_interface.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.qwen_vl_interface.model.config.use_cache = False
    m.action_model.model.enable_gradient_checkpointing()

    if rank == 0:
        apply_candidate(m, "dense", attention_backend="fa4")
        use_worker_inputs(m)
        batch_gpu = send_to_device(CPUInputsCollator(cfg)(raw_examples), dev)
        m.train()
        torch.manual_seed(7777); torch.cuda.manual_seed_all(7777)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(batch_gpu)["action_loss"]
        loss.backward()
        grads = {
            n: (p.grad.detach() if p.grad is not None else torch.zeros_like(p.detach())).cpu().bfloat16()
            for n, p in m.named_parameters()
        }
        loss_val = float(loss.detach().cpu())
        torch.save({"loss": loss_val, "grads": grads, "keys": len(m.state_dict())}, SHM_DIR / "bf16_ref.pt")
        print(f"[bf16_ref] loss={loss_val:.6f} keys={len(m.state_dict())}", flush=True)
        del m, batch_gpu, grads
        torch.cuda.empty_cache()
        from examples.unifolm_wla.fp8_opt import install_fp8_training_optimizations, verify_state_dict_roundtrip
        m_opt_rt = build_framework(cfg)
        load_base(m_opt_rt, cfg.trainer.pretrained_checkpoint)
        install_fp8_training_optimizations(m_opt_rt, compile_blocks=False)
        m_fresh = build_framework(cfg)
        rt_info = verify_state_dict_roundtrip(m_opt_rt, m_fresh)
        print(f"[Roundtrip Gate] {rt_info}", flush=True)
        del m_opt_rt, m_fresh
    else:
        rep = install_fp8_training_optimizations(
            m,
            bwd_fast_accum=True,
            use_cudnn_qwen_sdpa=True,
            skip_last_txt_mlp=True,
            compile_blocks=False,
        )
        install_zero_sync_forward(m)
        sd = m.state_dict()
        assert len(sd) == 1256, len(sd)
        batch_gpu = send_to_device(FastWLAInputsCollator(cfg, tight_seq=True, gpu_patch_unpack=True)(raw_examples), dev)
        m.train()

        for step_idx, tag in ((1, "fp8_opt_step1_calib"), (2, "fp8_opt_step2_delayed")):
            m.zero_grad(set_to_none=True)
            cache_all_fp8_weights(m)
            torch.manual_seed(7777); torch.cuda.manual_seed_all(7777)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(batch_gpu)["action_loss"]
            loss.backward()
            raw_grads = {
                n: (p.grad.detach() if p.grad is not None else torch.zeros_like(p.detach()))
                for n, p in m.named_parameters()
            }
            all_fin = all(bool(torch.isfinite(g).all()) for g in raw_grads.values())
            canon_grads = {}
            for k, g in raw_grads.items():
                if k in m._canonical_keys and k not in m._canonical_mapping:
                    canon_grads[k] = g.cpu().bfloat16()
            for new_k, (old_ks, sizes) in m._canonical_mapping.items():
                vals = raw_grads[new_k].split(sizes, dim=0) if sizes is not None else [raw_grads[new_k]]
                for ok, vg in zip(old_ks, vals):
                    canon_grads[ok] = vg.cpu().bfloat16()
            canon_grads["qwen_vl_interface.model.lm_head.weight"] = canon_grads["qwen_vl_interface.model.model.language_model.embed_tokens.weight"]
            assert len(canon_grads) == 1256, len(canon_grads)
            loss_val = float(loss.detach().cpu())
            torch.save(
                {"loss": loss_val, "grads": canon_grads, "all_finite": all_fin, "report": rep},
                SHM_DIR / f"{tag}.pt",
            )
            print(f"[{tag}] loss={loss_val:.6f} all_finite={all_fin}", flush=True)


def main():
    run_unit_checks()
    if "--unit-only" in sys.argv:
        return

    if SHM_DIR.exists():
        shutil.rmtree(SHM_DIR)
    SHM_DIR.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=model_worker, args=(r,)) for r in (0, 1)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0, f"Worker failed with exitcode {p.exitcode}"

    ref = torch.load(SHM_DIR / "bf16_ref.pt", map_location="cpu")
    summary = {"bf16_ref_loss": ref["loss"], "canonical_state_dict_keys": ref["keys"]}
    for tag in ("fp8_opt_step1_calib", "fp8_opt_step2_delayed"):
        cand = torch.load(SHM_DIR / f"{tag}.pt", map_location="cpu")
        diff_sq, ref_sq = 0.0, 0.0
        for k, rg in ref["grads"].items():
            cg = cand["grads"][k].float()
            rf = rg.float()
            diff_sq += float((cg - rf).pow(2).sum())
            ref_sq += float(rf.pow(2).sum())
        rel_l2 = (diff_sq ** 0.5) / (ref_sq ** 0.5)
        summary[tag] = {
            "loss": cand["loss"],
            "loss_abs_diff_vs_bf16": abs(cand["loss"] - ref["loss"]),
            "grad_rel_l2_vs_bf16": round(rel_l2, 5),
            "all_grads_finite": cand["all_finite"],
        }
        assert cand["all_finite"], f"Non-finite grads in {tag}!"
        assert rel_l2 < 0.38, f"Grad rel L2 too high in {tag}: {rel_l2}"

    out_path = Path("/storage/wangdonglinLab/weizhenyu/kuimouyu/research/unifoLM-WLA-fp8/runs/fp8-mfu-opt-20261118/equivalence-report-fp8-opt.json")
    out_path.write_text(json.dumps(summary, indent=2))
    shutil.rmtree(SHM_DIR, ignore_errors=True)
    print("=== EQUIVALENCE_FP8_OPT_SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
