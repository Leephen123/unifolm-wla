"""Step-Cached Horizontally-Fused FP8 Linear Module (`StepCachedFP8Linear`) & Rank-0 Algo Sync."""
from __future__ import annotations

import ctypes
import os
import torch
import torch._dynamo
import torch.distributed as dist
from torch import nn

from .cuda_ops import (
    LT_WORKSPACE_BYTES,
    _ptr,
    begin_fp8_step,
    get_cuda_lib,
    get_pool,
    lt_fp8_mm,
    quantize_1d_slot,
    quantize_dual_slot,
    quantize_dual_slot_into,
    reset_fp8_pool,
    wait_fp8_weight_quant,
)

@torch.library.custom_op("wla_fp8_opt::linear_infer", mutates_args=())
def fused_fp8_linear_infer(
    x_2d: torch.Tensor, w_row: torch.Tensor, w_scale: torch.Tensor,
    bias_buf: torch.Tensor, has_bias: bool, slot_id: int,
) -> torch.Tensor:
    x_e4, x_s = quantize_1d_slot(x_2d, slot_id * 5 + 0, torch.float8_e4m3fn)
    return lt_fp8_mm(x_e4, w_row, x_s, w_scale, bias=bias_buf if has_bias else None)


@fused_fp8_linear_infer.register_fake
def _lin_inf_fake(x_2d, w_row, w_scale, bias_buf, has_bias, slot_id):
    return torch.empty((x_2d.shape[0], w_row.shape[0]), dtype=torch.bfloat16, device=x_2d.device)


@torch.library.custom_op("wla_fp8_opt::linear_bwd", mutates_args=())
def _fp8_linear_bwd(
    dy_2d: torch.Tensor, x_t: torch.Tensor, w_t: torch.Tensor, w_scale: torch.Tensor,
    has_bias: bool, slot_id: int, bwd_fast_accum: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_s = get_pool(dy_2d.device).curr_scale[slot_id * 5 + 0]
    if has_bias:
        dy_row, dy_t, dy_s, dbias = quantize_dual_slot(dy_2d, slot_id * 5 + 1, torch.float8_e5m2, need_dbias=True)
    else:
        dy_row, dy_t, dy_s = quantize_dual_slot(dy_2d, slot_id * 5 + 1, torch.float8_e5m2, need_dbias=False)
        dbias = torch.empty((0,), dtype=torch.bfloat16, device=dy_2d.device)
    dx = lt_fp8_mm(dy_row, w_t, dy_s, w_scale, fast_accum=bwd_fast_accum, is_bwd=True)
    dw = lt_fp8_mm(dy_t, x_t, dy_s, x_s, fast_accum=bwd_fast_accum, is_bwd=True)
    return dx, dw, dbias


@_fp8_linear_bwd.register_fake
def _lin_bwd_fake(dy_2d, x_t, w_t, w_scale, has_bias, slot_id, bwd_fast_accum):
    K, M = x_t.shape
    N = dy_2d.shape[1]
    db = torch.empty((N,), dtype=torch.bfloat16, device=dy_2d.device) if has_bias else torch.empty((0,), dtype=torch.bfloat16, device=dy_2d.device)
    return (
        torch.empty((M, K), dtype=torch.bfloat16, device=dy_2d.device),
        torch.empty((N, K), dtype=torch.bfloat16, device=dy_2d.device),
        db,
    )


@torch.library.custom_op("wla_fp8_opt::linear_train", mutates_args=())
def fused_fp8_linear_train(
    x_2d: torch.Tensor, weight: torch.Tensor, bias_buf: torch.Tensor, has_bias: bool,
    w_row: torch.Tensor, w_t: torch.Tensor, w_scale: torch.Tensor, slot_id: int, bwd_fast_accum: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_e4, x_t, x_s = quantize_dual_slot(x_2d, slot_id * 5 + 0, torch.float8_e4m3fn)
    out = lt_fp8_mm(x_e4, w_row, x_s, w_scale, bias=bias_buf if has_bias else None)
    del x_e4
    return out, x_t


@fused_fp8_linear_train.register_fake
def _lin_tr_fake(x_2d, weight, bias_buf, has_bias, w_row, w_t, w_scale, slot_id, bwd_fast_accum):
    M, K = x_2d.shape
    return (
        torch.empty((M, weight.shape[0]), dtype=torch.bfloat16, device=x_2d.device),
        torch.empty((K, M), dtype=torch.float8_e4m3fn, device=x_2d.device),
    )


def _lin_setup(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    x_2d, _, bias_buf, has_bias, _, w_t, w_scale, slot_id, bwd_fast_accum = inputs
    _, x_t = output
    ctx.save_for_backward(x_t, w_t, w_scale)
    ctx.has_bias = has_bias
    ctx.b_dtype = bias_buf.dtype
    ctx.slot_id = slot_id
    ctx.bwd_fast_accum = bwd_fast_accum
    ctx.x_dtype = x_2d.dtype
    ctx.mark_non_differentiable(x_t)


def _lin_bwd(ctx, grad_out, _gxt):
    if grad_out is None:
        return (None,) * 9
    x_t, w_t, w_scale = ctx.saved_tensors
    dx, dw, db = _fp8_linear_bwd(grad_out, x_t, w_t, w_scale, ctx.has_bias, ctx.slot_id, ctx.bwd_fast_accum)
    if dx.dtype != ctx.x_dtype:
        dx = dx.to(ctx.x_dtype)
    grad_b = (db.to(ctx.b_dtype) if db.dtype != ctx.b_dtype else db) if ctx.has_bias else None
    return dx, dw, grad_b, None, None, None, None, None, None

fused_fp8_linear_train.register_autograd(_lin_bwd, setup_context=_lin_setup)


class StepCachedFP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, slot_id: int, bias: bool = False,
                 device=None, dtype=torch.bfloat16, bwd_fast_accum: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.slot_id = int(slot_id)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.bwd_fast_accum = bwd_fast_accum
        self.register_buffer("_w_row", None, persistent=False)
        self.register_buffer("_w_t", None, persistent=False)
        self.register_buffer("_w_scale", None, persistent=False)
        self.register_buffer("_empty_bias", torch.empty((0,), device=device, dtype=dtype), persistent=False)

    @classmethod
    def from_linears(cls, parts: list[nn.Linear], slot_id: int, bwd_fast_accum: bool = True):
        has_bias = parts[0].bias is not None
        assert all((p.bias is not None) == has_bias for p in parts)
        in_f = parts[0].in_features
        out_f = sum(p.out_features for p in parts)
        mod = cls(in_f, out_f, slot_id=slot_id, bias=has_bias,
                  device=parts[0].weight.device, dtype=parts[0].weight.dtype, bwd_fast_accum=bwd_fast_accum)
        with torch.no_grad():
            mod.weight.copy_(torch.cat([p.weight for p in parts], dim=0))
            if has_bias:
                mod.bias.copy_(torch.cat([p.bias for p in parts], dim=0))
        return mod

    def get_or_quant_weight(self):
        if self._w_row is not None:
            return self._w_row, self._w_t, self._w_scale
        w_row, w_t, w_s = quantize_dual_slot(self.weight.detach(), self.slot_id * 5 + 4, torch.float8_e4m3fn)
        return w_row, w_t, w_s.clone()

    def cache_step_weight(self):
        with torch.no_grad():
            w = self.weight.detach()
            if self._w_row is None or self._w_row.device != w.device:
                self.register_buffer("_w_row", torch.empty((self.out_features, self.in_features), dtype=torch.float8_e4m3fn, device=w.device), persistent=False)
                self.register_buffer("_w_t", torch.empty((self.in_features, self.out_features), dtype=torch.float8_e4m3fn, device=w.device), persistent=False)
                self.register_buffer("_w_scale", torch.empty((), dtype=torch.float32, device=w.device), persistent=False)
            if self._empty_bias.device != w.device:
                self.register_buffer("_empty_bias", torch.empty((0,), dtype=w.dtype, device=w.device), persistent=False)
            scale = quantize_dual_slot_into(w, self._w_row, self._w_t, self.slot_id * 5 + 4)
            self._buffers["_w_scale"] = scale

    def clear_step_weight(self, release_memory: bool = False):
        if release_memory:
            self.register_buffer("_w_row", None, persistent=False)
            self.register_buffer("_w_t", None, persistent=False)
            self.register_buffer("_w_scale", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wait_fp8_weight_quant(x.device)
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        w_row, w_t, w_scale = self.get_or_quant_weight()
        has_bias = self.bias is not None
        b_buf = self.bias if has_bias else self._empty_bias
        if torch.is_grad_enabled():
            out_2d, _ = fused_fp8_linear_train(
                x_2d, self.weight, b_buf, has_bias, w_row, w_t, w_scale, self.slot_id, self.bwd_fast_accum
            )
        else:
            out_2d = fused_fp8_linear_infer(x_2d, w_row, w_scale, b_buf, has_bias, self.slot_id)
        return out_2d.view(*orig_shape[:-1], self.out_features)

def cache_all_fp8_weights(model: nn.Module):
    fp8_mods = getattr(model, "_fp8_opt_modules", None)
    if fp8_mods is None:
        fp8_mods = [m for m in model.modules() if isinstance(m, StepCachedFP8Linear)]
        model._fp8_opt_modules = fp8_mods
    if not fp8_mods:
        return
    dev = fp8_mods[0].weight.device
    slot_meta = getattr(model, "_fp8_opt_slot_meta", None)
    pool = get_pool(dev)
    def _sync_rank0_lt_algos():
        import torch.distributed as dist
        lib = get_cuda_lib()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            max_ent = 512
            eb = int(lib.lt_serialized_entry_bytes())
            buf_cpu = torch.zeros(max_ent * eb, dtype=torch.uint8)
            cnt_gpu = torch.zeros(1, dtype=torch.int32, device=dev)
            if dist.get_rank() == 0:
                n_exp = int(lib.lt_export_algos(ctypes.c_void_p(buf_cpu.data_ptr()), max_ent))
                cnt_gpu.fill_(n_exp)
            dist.broadcast(cnt_gpu, src=0)
            n_exp = int(cnt_gpu.item())
            if n_exp > 0:
                buf_gpu = buf_cpu[: n_exp * eb].to(device=dev)
                dist.broadcast(buf_gpu, src=0)
                if dist.get_rank() != 0:
                    buf_rx = buf_gpu.cpu().contiguous()
                    lib.lt_import_algos(ctypes.c_void_p(buf_rx.data_ptr()), n_exp, LT_WORKSPACE_BYTES)
            lib.lt_set_autotune_enabled(0)

    if pool.step_count == 2 and not getattr(pool, "_lt_algos_synced", False):
        pool._lt_algos_synced = True
        _sync_rank0_lt_algos()
    elif pool.step_count == 3 and not getattr(pool, "_lt_warm_retuned", False) and os.environ.get("WLA_WARM_RETUNE", "1") == "1":
        pool._lt_warm_retuned = True
        torch.cuda.synchronize()
        lib = get_cuda_lib()
        lib.lt_clear_all_caches()
        lib.lt_set_autotune_enabled(1)
    elif pool.step_count == 4 and not getattr(pool, "_lt_warm_synced", False) and os.environ.get("WLA_WARM_RETUNE", "1") == "1":
        pool._lt_warm_synced = True
        _sync_rank0_lt_algos()
    use_async = (pool.step_count >= 2 and pool.step_count != 3) and (os.environ.get("WLA_ASYNC_WEIGHT_QUANT", "1") == "1")
    if use_async and getattr(model, "_has_zero_sync_fwd", False):
        def _deferred_ag_and_quant(pre_vis_ev):
            pool.quant_stream.wait_event(pre_vis_ev)
            with torch.cuda.stream(pool.quant_stream):
                pending_ag = getattr(model, "_pending_non_vis_ag", None)
                if pending_ag is not None:
                    model._pending_non_vis_ag = None
                    pending_ag(pool.quant_stream)
                begin_fp8_step(dev, slot_meta)
                for m in fp8_mods:
                    m.cache_step_weight()
            pool.quant_pending = True

        model._deferred_fp8_quant_fn = _deferred_ag_and_quant
    elif use_async:
        curr_s = torch.cuda.current_stream(dev)
        pool.quant_stream.wait_stream(curr_s)
        with torch.cuda.stream(pool.quant_stream):
            pending_ag = getattr(model, "_pending_non_vis_ag", None)
            if pending_ag is not None:
                model._pending_non_vis_ag = None
                pending_ag(pool.quant_stream)
            begin_fp8_step(dev, slot_meta)
            for m in fp8_mods:
                m.cache_step_weight()
        pool.quant_pending = True
    else:
        pending_ag = getattr(model, "_pending_non_vis_ag", None)
        if pending_ag is not None:
            model._pending_non_vis_ag = None
            pending_ag(torch.cuda.current_stream(dev))
        begin_fp8_step(dev, slot_meta)
        for m in fp8_mods:
            m.cache_step_weight()


def clear_all_fp8_weights(model: nn.Module, release_memory: bool = False):
    if release_memory:
        modules = getattr(model, "_fp8_opt_modules", None)
        if modules is None:
            modules = [m for m in model.modules() if isinstance(m, StepCachedFP8Linear)]
        for m in modules:
            m.clear_step_weight(release_memory=True)
