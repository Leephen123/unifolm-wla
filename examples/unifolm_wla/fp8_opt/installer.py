"""Unified Model Installer & Lossless 1,256-Key Canonical `state_dict` Adapter."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import types
import torch
import torch._dynamo
from torch import nn
import torch.nn.functional as F

from .cuda_ops import FP8_E4M3_MAX, FP8_E5M2_MAX, MAX_SLOTS, _cuda_device, get_cuda_lib, wait_fp8_weight_quant
from .data_collator import install_zero_sync_forward
from .delayed_scaling import StepCachedFP8Linear, clear_all_fp8_weights
from .language_backbone import fast_language_model_forward, fast_qwen_decoder_layer_forward
from .mmdit_expert import fast_mmdit_model_forward
from .vision_tower import fast_vision_block_forward, fast_vision_model_forward


@dataclass
class WLAFP8Config:
    """Configuration for SM90 Native FP8 + BF16 Hybrid UnifoLM-WLA Training."""

    bwd_fast_accum: bool = True
    use_cudnn_qwen_sdpa: bool = True
    skip_last_txt_mlp: bool = True
    compile_mergers: bool = True
    install_zero_sync: bool = False


def install_wla_fp8_training(model: nn.Module, config: WLAFP8Config | None = None) -> dict:
    """Install SM90 FP8 Linear fusion, flat C++ full-GC block runners, and canonical `state_dict` hooks.

    Guarantees:
    - 100% Full Gradient Checkpointing preserved across all Vision (24), Qwen (36), and MMDiT (16) blocks (ZERO SAC).
    - Vision tower, robot_state_projector, action heads, norms, modulations, and attention stay strictly BF16.
    - `model.state_dict()` and `load_state_dict` remain 100% lossless with the 1,256 canonical keys.
    """
    if config is None:
        config = WLAFP8Config(
            use_cudnn_qwen_sdpa=(os.environ.get("WLA_QWEN_CUDNN_SDPA", "1") == "1"),
        )
    get_cuda_lib()

    original_keys = set(model.state_dict().keys())
    original_numel = sum(p.numel() for p in model.parameters())
    mapping: dict[str, tuple[list[str], list[int] | None]] = {}
    slot_counter = [0]
    slot_meta: list[tuple[int, float]] = []

    def next_slot() -> int:
        s = slot_counter[0]
        slot_counter[0] += 1
        assert s * 5 + 4 < MAX_SLOTS, f"Exceeded MAX_SLOTS={MAX_SLOTS}: {s}"
        slot_meta.extend([
            (s * 5 + 0, FP8_E4M3_MAX),
            (s * 5 + 1, FP8_E5M2_MAX),
            (s * 5 + 2, FP8_E4M3_MAX),
            (s * 5 + 3, FP8_E5M2_MAX),
            (s * 5 + 4, FP8_E4M3_MAX),
        ])
        return s

    def record(new_prefix: str, old_prefixes: list[str], sizes: list[int] | None = None, has_bias: bool = False) -> None:
        mapping[new_prefix + ".weight"] = ([p + ".weight" for p in old_prefixes], sizes)
        if has_bias:
            mapping[new_prefix + ".bias"] = ([p + ".bias" for p in old_prefixes], sizes)

    bfa = config.bwd_fast_accum

    # 1. Language Backbone (36 Qwen3-VL Decoder Layers)
    lang = model.qwen_vl_interface.model.model.language_model
    lang._use_cudnn_sdpa = config.use_cudnn_qwen_sdpa
    lang._slim_gc_no_down_recompute = True
    lang.forward = types.MethodType(fast_language_model_forward, lang)
    for i, layer in enumerate(lang.layers):
        p = f"qwen_vl_interface.model.model.language_model.layers.{i}"
        a = layer.self_attn
        a.qkv_sizes = [a.q_proj.out_features, a.k_proj.out_features, a.v_proj.out_features]
        a.qkv_proj = StepCachedFP8Linear.from_linears([a.q_proj, a.k_proj, a.v_proj], next_slot(), bwd_fast_accum=bfa)
        record(f"{p}.self_attn.qkv_proj", [f"{p}.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj")], a.qkv_sizes, has_bias=(a.q_proj.bias is not None))
        del a.q_proj, a.k_proj, a.v_proj
        a.o_proj = StepCachedFP8Linear.from_linears([a.o_proj], next_slot(), bwd_fast_accum=bfa)

        m = layer.mlp
        gu_sizes = [m.gate_proj.out_features, m.up_proj.out_features]
        m.gate_up_proj = StepCachedFP8Linear.from_linears([m.gate_proj, m.up_proj], next_slot(), bwd_fast_accum=bfa)
        record(f"{p}.mlp.gate_up_proj", [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"], gu_sizes, has_bias=(m.gate_proj.bias is not None))
        del m.gate_proj, m.up_proj
        m.down_proj = StepCachedFP8Linear.from_linears([m.down_proj], next_slot(), bwd_fast_accum=bfa)
        layer._use_cudnn_sdpa = config.use_cudnn_qwen_sdpa
        layer.forward = types.MethodType(fast_qwen_decoder_layer_forward, layer)

    # 2. Vision Tower (24 BF16 Blocks, 100% Full GC ON, ZERO SAC)
    vis = model.qwen_vl_interface.model.model.visual
    for buf_name in ("_1cam_idx", "_1cam_w", "_1cam_cos", "_1cam_sin", "_dummy_cu"):
        vis.register_buffer(buf_name, None, persistent=False)
    vis._num_ckpt_blocks = len(vis.blocks)
    vis._attn_saved_sac = False
    vis._slim_gc_no_fc2_recompute = True

    def _fast_patch_embed_forward(pe_self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = pe_self.proj.weight
        return F.linear(
            hidden_states.reshape(-1, w.numel() // pe_self.embed_dim).to(dtype=w.dtype),
            w.view(pe_self.embed_dim, -1),
            pe_self.proj.bias,
        )

    vis.patch_embed.forward = types.MethodType(_fast_patch_embed_forward, vis.patch_embed)
    vis.forward = types.MethodType(fast_vision_model_forward, vis)
    for blk in vis.blocks:
        blk.forward = types.MethodType(fast_vision_block_forward, blk)

    # 3. MMDiT Action Expert (16 Joint Blocks, 100% Full GC ON, ZERO SAC)
    dit = model.action_model.model
    dit._hoisted_mod_gc = True
    dit._skip_last_txt_mlp = bool(config.skip_last_txt_mlp)
    dit.forward = types.MethodType(fast_mmdit_model_forward, dit)
    dit_blocks = dit.transformer_blocks
    num_dit = len(dit_blocks)
    for i, blk in enumerate(dit_blocks):
        p = f"action_model.model.transformer_blocks.{i}"
        is_last = config.skip_last_txt_mlp and (i == num_dit - 1)
        a = blk.attn
        qkv_parts = [a.to_q, a.to_k, a.to_v]
        a.to_qkv = StepCachedFP8Linear.from_linears(qkv_parts, next_slot(), bwd_fast_accum=bfa)
        record(f"{p}.attn.to_qkv", [f"{p}.attn.{n}" for n in ("to_q", "to_k", "to_v")], [x.out_features for x in qkv_parts], has_bias=(qkv_parts[0].bias is not None))
        del a.to_q, a.to_k, a.to_v

        add_parts = [a.add_q_proj, a.add_k_proj, a.add_v_proj]
        a.add_qkv = StepCachedFP8Linear.from_linears(add_parts, next_slot(), bwd_fast_accum=bfa)
        record(f"{p}.attn.add_qkv", [f"{p}.attn.{n}" for n in ("add_q_proj", "add_k_proj", "add_v_proj")], [x.out_features for x in add_parts], has_bias=(add_parts[0].bias is not None))
        del a.add_q_proj, a.add_k_proj, a.add_v_proj

        a.to_out[0] = StepCachedFP8Linear.from_linears([a.to_out[0]], next_slot(), bwd_fast_accum=bfa)
        if not is_last:
            a.to_add_out = StepCachedFP8Linear.from_linears([a.to_add_out], next_slot(), bwd_fast_accum=bfa)

        blk.action_mlp.net[0].proj = StepCachedFP8Linear.from_linears([blk.action_mlp.net[0].proj], next_slot(), bwd_fast_accum=bfa)
        blk.action_mlp.net[2] = StepCachedFP8Linear.from_linears([blk.action_mlp.net[2]], next_slot(), bwd_fast_accum=bfa)
        if not is_last:
            blk.txt_mlp.net[0].proj = StepCachedFP8Linear.from_linears([blk.txt_mlp.net[0].proj], next_slot(), bwd_fast_accum=bfa)
            blk.txt_mlp.net[2] = StepCachedFP8Linear.from_linears([blk.txt_mlp.net[2]], next_slot(), bwd_fast_accum=bfa)
            blk._single_out = False
        else:
            blk._single_out = True

    from transformers.utils.output_capturing import maybe_install_capturing_hooks
    maybe_install_capturing_hooks(lang)
    maybe_install_capturing_hooks(vis)
    for mod in model.modules():
        if type(mod).__name__ in ("Qwen3VLTextAttention", "Qwen3VLVisionAttention"):
            hooks = mod._forward_hooks
            mod._forward_hooks = type(hooks)((k, v) for k, v in hooks.items() if v.__name__ != "output_capturing_hook")

    if config.compile_mergers:
        torch._dynamo.config.recompile_limit = 256
        torch._dynamo.config.cache_size_limit = 512
        options = {"emulate_precision_casts": True, "fallback_random": True}
        mergers = [vis.merger] + list(getattr(vis, "deepstack_merger_list", []))
        for m in mergers:
            m.forward = torch.compile(m.forward, fullgraph=True, dynamic=False, options=options)

    if config.install_zero_sync:
        install_zero_sync_forward(model)

    assert sum(p.numel() for p in model.parameters()) == original_numel
    model._fp8_opt_slot_meta = slot_meta
    install_canonical_state_dict_hooks(model, mapping, original_keys)
    fp8_count = sum(1 for m in model.modules() if isinstance(m, StepCachedFP8Linear))
    return {
        "backend": "fp8_opt_sm90_native_v14",
        "fp8_modules": fp8_count,
        "total_slots": slot_counter[0],
        "bwd_fast_accum": bfa,
        "use_cudnn_qwen_sdpa": config.use_cudnn_qwen_sdpa,
        "full_gradient_checkpointing": True,
        "sac_enabled": False,
    }


def install_fp8_training_optimizations(
    model: nn.Module,
    *,
    bwd_fast_accum: bool = True,
    use_cudnn_qwen_sdpa: bool = True,
    skip_last_txt_mlp: bool = True,
    compile_blocks: bool = True,
) -> dict:
    """Ergonomic keyword wrapper around `install_wla_fp8_training`."""
    return install_wla_fp8_training(
        model,
        WLAFP8Config(
            bwd_fast_accum=bwd_fast_accum,
            use_cudnn_qwen_sdpa=use_cudnn_qwen_sdpa,
            skip_last_txt_mlp=skip_last_txt_mlp,
            compile_mergers=compile_blocks,
            install_zero_sync=False,
        ),
    )


_LM_KEY = "qwen_vl_interface.model.lm_head.weight"
_EMBED_KEY = "qwen_vl_interface.model.model.language_model.embed_tokens.weight"


def install_canonical_state_dict_hooks(model, mapping, canonical_keys):
    """Bind canonical split-key export and load to this model instance."""
    model._canonical_mapping = mapping
    model._canonical_keys = set(canonical_keys)

    def _canonical_state_dict(self, *args, destination=None, prefix="", keep_vars=False, **kwargs):
        if args:
            if len(args) >= 1:
                destination = args[0]
            if len(args) >= 2:
                prefix = args[1]
            if len(args) >= 3:
                keep_vars = args[2]
        if getattr(self, "_pending_non_vis_ag", None) is not None:
            raise RuntimeError(
                "Pending async ZeRO-2 non-vision weight gather detected in state_dict(). "
                "Call flush_pending_weight_gather(model) (or accelerator.get_state_dict(model) / "
                "engine.save_checkpoint(...)) across ALL ranks before exporting weights on rank 0."
            )
        dev = _cuda_device(self)
        if dev is not None:
            wait_fp8_weight_quant(dev)
        raw = nn.Module.state_dict(self, destination=None, prefix="", keep_vars=keep_vars, **kwargs)
        res = OrderedDict((k, v) for k, v in raw.items() if k in self._canonical_keys and k not in self._canonical_mapping)
        for new_k, (old_ks, sizes) in self._canonical_mapping.items():
            parts = raw[new_k].split(sizes, dim=0) if sizes is not None else [raw[new_k]]
            res.update(zip(old_ks, parts))
        for alias, source in ((_LM_KEY, _EMBED_KEY), (_EMBED_KEY, _LM_KEY)):
            if alias in self._canonical_keys and alias not in res:
                res[alias] = raw[source]
        assert set(res) == self._canonical_keys
        if prefix or destination is not None:
            out = destination if destination is not None else OrderedDict()
            out.update((prefix + k, v) for k, v in res.items())
            return out
        return res

    def _canonical_load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        dev = _cuda_device(self)
        if dev is not None:
            wait_fp8_weight_quant(dev)
        remapped = dict(state_dict)
        raw_sd = nn.Module.state_dict(self, keep_vars=True)
        partial_missing = []
        for new_k, (old_ks, sizes) in self._canonical_mapping.items():
            if new_k in remapped:
                continue
            if all(k in remapped for k in old_ks):
                parts = [remapped.pop(k) for k in old_ks]
                remapped[new_k] = torch.cat(parts, dim=0) if sizes is not None else parts[0]
            elif not strict and sizes is not None and new_k in raw_sd and any(k in remapped for k in old_ks):
                ref = next(remapped[k] for k in old_ks if k in remapped)
                parts = []
                for key, current in zip(old_ks, raw_sd[new_k].detach().split(sizes)):
                    if key in remapped:
                        parts.append(remapped.pop(key))
                    else:
                        parts.append(current.to(ref))
                        partial_missing.append(key)
                remapped[new_k] = torch.cat(parts, dim=0)

        for alias, target in ((_LM_KEY, _EMBED_KEY), (_EMBED_KEY, _LM_KEY)):
            if alias not in raw_sd and target in raw_sd and alias in remapped:
                value = remapped.pop(alias)
                remapped.setdefault(target, value)
            elif (alias in raw_sd and target in raw_sd and alias in remapped and target not in remapped
                  and raw_sd[alias].data_ptr() == raw_sd[target].data_ptr()):
                remapped[target] = remapped[alias]

        try:
            incompat = nn.Module.load_state_dict(self, remapped, strict=strict, assign=assign)
        finally:
            for attr in ("_pending_non_vis_ag", "_deferred_fp8_quant_fn"):
                if hasattr(self, attr):
                    setattr(self, attr, None)
            clear_all_fp8_weights(self, release_memory=True)
        missing = []
        for key in incompat.missing_keys:
            missing.extend(self._canonical_mapping[key][0] if key in self._canonical_mapping else [key])
        return type(incompat)(missing + partial_missing, list(incompat.unexpected_keys))

    model.state_dict = types.MethodType(_canonical_state_dict, model)
    model.load_state_dict = types.MethodType(_canonical_load_state_dict, model)


def verify_state_dict_roundtrip(opt_model: nn.Module, fresh_canonical_model: nn.Module,
                                expected_keys: int | None = 1256) -> dict:
    """Check both strict directions and canonical missing keys for a partial load."""
    def max_diff(left, right):
        assert left.keys() == right.keys()
        return max((a.detach().cpu().float() - right[k].detach().cpu().float()).abs().max().item()
                   for k, a in left.items() if a.numel())

    sd = opt_model.state_dict()
    missing, unexpected = fresh_canonical_model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    ref_sd = fresh_canonical_model.state_dict()
    assert expected_keys is None or len(sd) == expected_keys
    fwd_diff = max_diff(sd, ref_sd)
    assert fwd_diff == 0.0, fwd_diff
    with torch.no_grad():
        for param in opt_model.parameters():
            param.zero_()
    missing, unexpected = opt_model.load_state_dict(ref_sd, strict=True)
    assert not missing and not unexpected
    assert all(m._w_row is None for m in opt_model.modules() if isinstance(m, StepCachedFP8Linear))
    rev_diff = max_diff(opt_model.state_dict(), ref_sd)
    assert rev_diff == 0.0, rev_diff

    groups = [keys for keys, sizes in opt_model._canonical_mapping.values() if sizes is not None]
    fused = {k for keys in groups for k in keys}
    plain = next(k for k in ref_sd if k not in fused and k not in (_LM_KEY, _EMBED_KEY))
    dropped = {*groups[0], plain}
    partial_sd = {k: v for k, v in ref_sd.items() if k not in dropped}
    try:
        incompat = opt_model.load_state_dict(partial_sd, strict=False)
        assert set(incompat.missing_keys) == dropped and not incompat.unexpected_keys, incompat
    finally:
        opt_model.load_state_dict(ref_sd, strict=True)
    return {"canonical_keys": len(sd), "max_abs_diff": max(fwd_diff, rev_diff),
            "reverse_max_abs_diff": rev_diff, "strict_load_ok": True, "reverse_strict_load_ok": True}
