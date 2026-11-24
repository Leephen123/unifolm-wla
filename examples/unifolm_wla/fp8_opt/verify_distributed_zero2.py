"""Single-node regression suite for torchrun --nproc_per_node=8 (SM90)."""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from datetime import timedelta

_LOCAL_RANK_ENV = os.environ.get("LOCAL_RANK", "0")
_TRITON_DIR = f"/dev/shm/wla_triton_z2_{os.getpid()}_{_LOCAL_RANK_ENV}"
os.makedirs(_TRITON_DIR, exist_ok=True)
atexit.register(shutil.rmtree, _TRITON_DIR, True)
os.environ.setdefault("TRITON_CACHE_DIR", _TRITON_DIR)

import deepspeed
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch import nn

from .cuda_ops import (FP8_E4M3_MAX, FP8_E5M2_MAX, get_cuda_lib, lt_bf16_dw_dbias,
                       reset_fp8_pool, try_begin_direct_ipg_block, wait_fp8_weight_quant)
from .delayed_scaling import StepCachedFP8Linear, cache_all_fp8_weights
from .installer import install_canonical_state_dict_hooks
from .zero2_hooks import flush_pending_weight_gather, install_fast_zero2_hooks

WIDTH = 256
SEED = 431


def _linear(inputs=WIDTH, outputs=WIDTH, bias=False):
    return nn.Linear(inputs, outputs, bias=bias, dtype=torch.bfloat16)


class _DirectDecoderFn(torch.autograd.Function):
    """Exercise the same direct BF16 weight-gradient write used by the vision blocks."""
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return nn.functional.linear(x, weight)

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        alloc = try_begin_direct_ipg_block(x.device, (weight,), (weight,))
        dw, _ = lt_bf16_dw_dbias(dy, x, out=alloc(weight) if alloc else None)
        return dy @ weight, None if alloc else dw


class _SyntheticWLAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_vl_interface = nn.Module()
        qwen = self.qwen_vl_interface.model = nn.Module()
        trunk = qwen.model = nn.Module()
        trunk.visual = nn.Sequential(_linear(), nn.GELU(), _linear())
        lang = trunk.language_model = nn.Module()
        lang.embed_tokens = nn.Embedding(1024, WIDTH, dtype=torch.bfloat16)
        lang.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            attn = layer.self_attn = nn.Module()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attn, name, _linear())
            mlp = layer.mlp = nn.Module()
            mlp.gate_proj = _linear(WIDTH, 2 * WIDTH)
            mlp.up_proj = _linear(WIDTH, 2 * WIDTH)
            mlp.down_proj = _linear(2 * WIDTH, WIDTH)
            lang.layers.append(layer)
        qwen.lm_head = _linear(WIDTH, 1024)
        qwen.lm_head.weight = lang.embed_tokens.weight

        self.action_model = nn.Module()
        action = self.action_model.model = nn.Module()
        action.transformer_blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            attn = block.attn = nn.Module()
            for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
                setattr(attn, name, _linear(bias=True))
            attn.to_out = _linear()
            block.action_mlp = nn.Sequential(_linear(WIDTH, 2 * WIDTH), nn.GELU(), _linear(2 * WIDTH, WIDTH))
            action.transformer_blocks.append(block)
        self.action_model.action_decoder = _linear()
        # Keep adjacent BF16 values within the comparison tolerance after three updates.
        with torch.no_grad():
            for p in self.parameters():
                p.uniform_(-0.002, 0.002)

    def forward(self, pixels, tokens):
        trunk = self.qwen_vl_interface.model.model
        x = pixels + trunk.visual(pixels)
        wait_fp8_weight_quant(x.device)
        x = x + trunk.language_model.embed_tokens(tokens)
        for layer in trunk.language_model.layers:
            q, k, v = layer.self_attn.qkv_proj(x).chunk(3, dim=-1)
            x = x + layer.self_attn.o_proj(q + k + v)
            gate, up = layer.mlp.gate_up_proj(x).chunk(2, dim=-1)
            x = x + layer.mlp.down_proj(gate.sigmoid() * up)
        for block in self.action_model.model.transformer_blocks:
            joint = block.attn.to_qkv(x) + block.attn.add_qkv(x)
            q, k, v = joint.chunk(3, dim=-1)
            x = x + block.attn.to_out(q + k + v)
            x = x + block.action_mlp(x)
        decoder = self.action_model.action_decoder
        if getattr(self, "test_direct_ipg", False):
            return _DirectDecoderFn.apply(x, decoder.weight)
        return decoder(x)


def _patch_synthetic_wla(model):
    canonical_keys = set(model.state_dict())
    mapping = {}
    meta = []

    def fuse(path, fused_name, *names):
        parent = model.get_submodule(path)
        parts = [getattr(parent, name) for name in names]
        slot = len(meta) // 5
        fused = StepCachedFP8Linear.from_linears(parts, slot)
        meta.extend((slot * 5 + i, FP8_E5M2_MAX if i in (1, 3) else FP8_E4M3_MAX) for i in range(5))
        for name in names:
            if name != fused_name:
                delattr(parent, name)
        setattr(parent, fused_name, fused)
        if len(parts) > 1:
            sizes = [p.out_features for p in parts]
            for suffix in (("weight", "bias") if fused.bias is not None else ("weight",)):
                mapping[f"{path}.{fused_name}.{suffix}"] = ([f"{path}.{n}.{suffix}" for n in names], sizes)

    for i in range(2):
        layer = f"qwen_vl_interface.model.model.language_model.layers.{i}"
        fuse(layer + ".self_attn", "qkv_proj", "q_proj", "k_proj", "v_proj")
        fuse(layer + ".self_attn", "o_proj", "o_proj")
        fuse(layer + ".mlp", "gate_up_proj", "gate_proj", "up_proj")
        fuse(layer + ".mlp", "down_proj", "down_proj")
        block = f"action_model.model.transformer_blocks.{i}"
        fuse(block + ".attn", "to_qkv", "to_q", "to_k", "to_v")
        fuse(block + ".attn", "add_qkv", "add_q_proj", "add_k_proj", "add_v_proj")
        fuse(block + ".attn", "to_out", "to_out")
        fuse(block + ".action_mlp", "0", "0")
        fuse(block + ".action_mlp", "2", "2")
    model._fp8_opt_slot_meta = meta
    install_canonical_state_dict_hooks(model, mapping, canonical_keys)


def _engine(dev, *, direct=True, fused=True, asynchronous=True, gas=1, fp32_comm=False, hooked=True):
    reset_fp8_pool(dev)
    torch.manual_seed(SEED)
    os.environ.update(WLA_DIRECT_FLAT_PART=str(int(direct)), WLA_FUSED_ZERO2_ADAMW=str(int(fused)),
                      WLA_ASYNC_WEIGHT_AG=str(int(asynchronous)))
    if fp32_comm:
        model = _linear(32, 1)
    else:
        model = _SyntheticWLAModel()
        _patch_synthetic_wla(model)
    model.to(dev)
    if not fp32_comm:
        qwen = model.qwen_vl_interface.model
        qwen.lm_head.weight = qwen.model.language_model.embed_tokens.weight
        assert 3_000_000 <= sum(p.numel() for p in model.parameters()) <= 4_000_000
    cfg = {
        "train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": gas,
        "bf16": {"enabled": True}, "communication_data_type": "fp32" if fp32_comm else "bf16",
        "data_types": {"grad_accum_dtype": "bf16"},
        "gradient_clipping": 1.0, "gradient_predivide_factor": 1.0,
        "zero_optimization": {"stage": 2, "reduce_scatter": False, "overlap_comm": True,
                              "contiguous_gradients": True, "round_robin_gradients": False,
                              "allgather_partitions": True, "reduce_bucket_size": 1048576},
        "optimizer": {"type": "AdamW", "params": {
            "torch_adam": True, "lr": 0.1 if fp32_comm else 1e-3,
            "betas": [0.0 if fp32_comm else 0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0.0 if fp32_comm else 0.01}},
    }
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=cfg)
    hooks = install_fast_zero2_hooks(engine) if hooked else {}
    if hooked:
        assert hooks["installed"] and hooks["direct_flat_part"] == direct
        assert hooks["fused_zero2_adamw"] == fused
        assert hooks["async_two_stage_weight_ag"] == (asynchronous and not fp32_comm)
    return engine, hooks


def _step(engine, step, gas=1):
    dev = engine.device
    cache_all_fp8_weights(engine.module)
    for micro in range(gas):
        rng = torch.Generator(device=dev).manual_seed(SEED + step * 100 + micro * 8 + dist.get_rank())
        pixels = torch.randn(16, WIDTH, generator=rng, device=dev, dtype=torch.bfloat16)
        tokens = torch.randint(1024, (16,), generator=rng, device=dev)
        target = torch.randn(16, WIDTH, generator=rng, device=dev)
        pred = engine(pixels, tokens)
        engine.backward((pred.float() * target).mean())
        engine.step()
        expected_accum = os.environ["WLA_DIRECT_FLAT_PART"] == "1" and micro + 1 < gas
        assert engine.optimizer._grad_accum_has_data[0] == expected_accum
    assert not engine.optimizer._touched_part_ids


def _weights(engine):
    return torch.cat([p.detach().reshape(-1) for p in engine.module.parameters()])


def _compare(left, right, tolerance=0):
    diff = (left.float() - right.float()).abs().max().item()
    assert (diff < tolerance if tolerance else diff == 0), diff
    return diff


def _accumulation(dev):
    metrics = {}
    for gas, steps in ((1, 3), (2, 2)):
        refs = {}
        step2_masters = {}
        for direct, fused in ((False, False), (False, True), (True, False), (True, True)):
            engine, _ = _engine(dev, gas=gas, direct=direct, fused=fused, asynchronous=fused)
            for step in range(1, steps + 1):
                _step(engine, step, gas)
                expected_cache = {0: step} if fused and step > 1 else {}
                assert engine.optimizer._step_int_cache == expected_cache
                if step == 2:
                    step2_masters[(direct, fused)] = (
                        engine.optimizer.single_partition_of_fp32_groups[0].detach().clone()
                    )
            flush_pending_weight_gather(engine)
            if not direct:
                refs[fused] = _weights(engine)
            else:
                # Exact 0.0 diff against same-AdamW non-direct_flat_part reference!
                exact_diff = _compare(_weights(engine), refs[fused], 0)
                metrics[f"gas{gas}_fused{int(fused)}_exact_diff"] = exact_diff
                if fused:
                    # Step-2 FP32 master weights (before Step-3 FP8 E4M3 bin quantization) match PyTorch AdamW < 1e-5
                    m_diff = _compare(step2_masters[(True, True)], step2_masters[(False, False)], 1e-5)
                    # Final weights across 3 FP8-quantized steps stay within 1 FP8 E4M3 bin (< 5e-4)
                    w_diff = _compare(_weights(engine), refs[False], 5e-4)
                    metrics[f"gas{gas}_step2_fp32_master_diff"] = m_diff
                    metrics[f"gas{gas}_final_bf16_vs_unfused_diff"] = w_diff
            del engine
    if dist.get_rank() == 0:
        print(f"  [Test 1 metrics] {metrics}", flush=True)


def _master_export(engine):
    opt = engine.optimizer
    local = opt.single_partition_of_fp32_groups[0].detach().bfloat16()
    masters = torch.empty(local.numel() * dist.get_world_size(), dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(masters, local.contiguous(), group=opt.real_dp_process_group[0])
    if dist.get_rank() != 0:
        return
    model = engine.module
    sd = model.state_dict()
    assert set(sd) == model._canonical_keys
    offsets = {}
    pos = 0
    for p in opt.bit16_groups[0]:
        offsets[id(p)] = pos
        pos += p.numel()
    expected = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        weight = masters.narrow(0, offsets[id(p)], p.numel()).view_as(p)
        if name in model._canonical_mapping:
            keys, sizes = model._canonical_mapping[name]
            expected.update(zip(keys, weight.split(sizes)))
        else:
            expected[name] = weight
    assert expected.keys() == sd.keys()
    actual = torch.cat([sd[k].reshape(-1) for k in expected])
    _compare(actual, torch.cat([v.reshape(-1) for v in expected.values()]))


def _export(dev):
    engine, _ = _engine(dev)
    engine.module.test_direct_ipg = True
    for step in range(1, 5):
        _step(engine, step)
    assert engine.optimizer._direct_ipg_stats[0] >= 4
    model = engine.module
    pending = model._pending_non_vis_ag
    assert pending is not None
    try:
        model.state_dict()
    except RuntimeError as error:
        assert "across ALL ranks" in str(error)
    else:
        raise AssertionError("state_dict accepted a pending gather")
    assert model._pending_non_vis_ag is pending
    flush_pending_weight_gather(engine)
    assert model._pending_non_vis_ag is None
    _master_export(engine)
    dist.barrier()
    _step(engine, 5)
    assert model._pending_non_vis_ag is not None
    sd = Accelerator().get_state_dict(engine)
    assert model._pending_non_vis_ag is None and set(sd) == model._canonical_keys


def _fp32_communication(dev):
    expected_norm = 4.5 * (32 ** 0.5)
    for hooked in (False, True):
        engine, hooks = _engine(dev, fp32_comm=True, hooked=hooked, asynchronous=False)
        if hooked:
            assert hooks["nvls_deferred_div"] is False
        before = engine.optimizer.single_partition_of_fp32_groups[0].detach().clone()
        x = torch.ones(1, 32, dtype=torch.bfloat16, device=dev)
        # Step 1 exercises Python norm/unscale fallback; Step 2 exercises fused CUDA AdamW inv_scale!
        engine.backward(engine(x).float().sum() * (dist.get_rank() + 1))
        engine.step()
        norm1 = float(engine.get_global_grad_norm())
        assert abs(norm1 - expected_norm) < 1e-4, (norm1, expected_norm)
        engine.backward(engine(x).float().sum() * (dist.get_rank() + 1))
        engine.step()
        norm2 = float(engine.get_global_grad_norm())
        assert abs(norm2 - expected_norm) < 1e-4, (norm2, expected_norm)
        update = engine.optimizer.single_partition_of_fp32_groups[0].detach() - before
        assert update.abs().max() > 0
        if hooked:
            assert abs(norm1 - ref_norm1) < 1e-5 and abs(norm2 - ref_norm2) < 1e-5
            _compare(update, ref_update, 1e-5)
            _compare(_weights(engine), ref_weights, 5e-5)
        else:
            ref_norm1, ref_norm2, ref_update, ref_weights = norm1, norm2, update, _weights(engine)


def _resume(dev, directory):
    a, _ = _engine(dev)
    for step in range(1, 5):
        _step(a, step)
    assert a.module._pending_non_vis_ag is not None
    a.save_checkpoint(directory, tag="step4")
    assert a.module._pending_non_vis_ag is None

    b, _ = _engine(dev)
    # Dirty the local bookkeeping to verify that restore actually resets it.
    b.optimizer._step_int_cache[0] = 99
    b.optimizer._grad_accum_has_data[0] = True
    cached = next(m for m in b.module.modules() if isinstance(m, StepCachedFP8Linear))
    cached.cache_step_weight()
    assert cached._w_row is not None
    restored, _ = b.load_checkpoint(directory, tag="step4", load_module_strict=True)
    assert restored and b.optimizer._step_int_cache == {} and not b.optimizer._grad_accum_has_data[0]
    assert cached._w_row is None and cached._w_t is None and cached._w_scale is None
    _compare(_weights(a), _weights(b))
    pa, pb = (e.optimizer.single_partition_of_fp32_groups[0] for e in (a, b))
    _compare(pa, pb)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        _compare(a.optimizer.optimizer.state[pa][key], b.optimizer.optimizer.state[pb][key])
    # The delayed-scaling pool is per device. Calibrate both continuations from identical state.
    for engine in (a, b):
        reset_fp8_pool(dev)
        _step(engine, 5)
        flush_pending_weight_gather(engine)
    _compare(_weights(a), _weights(b))
    _compare(pa, pb)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    dev = torch.device("cuda", local_rank)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60), device_id=dev)
    rank = dist.get_rank()
    directory = None
    try:
        assert dist.get_world_size() == 8 and torch.cuda.get_device_capability(dev) == (9, 0)
        os.environ.update(WLA_WARM_RETUNE="0", WLA_ASYNC_WEIGHT_QUANT="1", WLA_DIRECT_IPG="1")
        # The loader builds the shared library on first use if it is absent.
        if rank == 0:
            get_cuda_lib()
        dist.barrier()
        get_cuda_lib().lt_set_autotune_enabled(0)
        for idx, (test, label) in enumerate((
            (_accumulation, "gradient reset and micro-batch accumulation"),
            (_export, "two-stage gather and rank-0 export"),
            (_fp32_communication, "FP32 gradient communication"),
        ), start=1):
            test(dev)
            dist.barrier()
            if rank == 0:
                print(f"[PASS] Test {idx}: {label}", flush=True)
        if rank == 0:
            directory = tempfile.mkdtemp(prefix="wla-zero2-", dir="/dev/shm")
        paths = [directory]
        dist.broadcast_object_list(paths, src=0)
        _resume(dev, paths[0])
        dist.barrier()
        if rank == 0:
            print("[PASS] Test 4: strict ZeRO-2 checkpoint resume", flush=True)
            print("=== ALL 8-GPU DISTRIBUTED ZERO-2 TESTS PASSED ===", flush=True)
    finally:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
