"""SM90 Native CUDA C++ & CUDA 12.8 cublasLt Operator Bindings and 128 MiB Delayed-Scaling Pool."""
from __future__ import annotations

import ctypes
import os
from itertools import chain
from pathlib import Path
import subprocess
import torch
import torch.nn.functional as F

from examples.unifolm_wla.training_optimizations import DETERMINISTIC
from examples.unifolm_wla import training_fa4_attention as fa4_mod

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
MAX_SLOTS = 4096
LT_WORKSPACE_BYTES = 128 * 1024 * 1024

BF16_FWD_DEFAULT = 0
BF16_FWD_BIAS = 1
BF16_FWD_GELU_BIAS = 2
BF16_FWD_GELU_AUX_BIAS = 3
BF16_BWD_DX_DEFAULT = 4
BF16_BWD_DX_DGELU = 5
BF16_BWD_DW_BGRADB = 6
BF16_BWD_DX_DGELU_BGRAD = 7

_VP = ctypes.c_void_p
_INT = ctypes.c_int
_I64 = ctypes.c_int64
_SZT = ctypes.c_size_t
_FLT = ctypes.c_float

_LIB_SM90 = None
_POOLS: dict[int, DelayedScalingPool] = {}

def get_cuda_lib():
    global _LIB_SM90
    if _LIB_SM90 is not None:
        return _LIB_SM90
    csrc_dir = Path(__file__).parent / "csrc"
    so_path = csrc_dir / "libwla_sm90_kernels.so"
    cu_path = csrc_dir / "wla_sm90_kernels.cu"
    if not so_path.exists() and cu_path.exists():
        import nvidia.cublas
        cb_dir = Path(nvidia.cublas.__file__).parent
        subprocess.run(
            [
                "/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", "-arch=sm_90", "--use_fast_math",
                f"-I{cb_dir / 'include'}", f"-L{cb_dir / 'lib'}",
                f"-Xlinker", f"-rpath,{cb_dir / 'lib'}", "-lcublasLt",
                "--shared", "-Xcompiler", "-fPIC", str(cu_path), "-o", str(so_path),
            ],
            check=True,
        )
    lib = ctypes.CDLL(str(so_path))
    lib.launch_finalize_delayed_scales.argtypes = [_VP, _VP, _VP, _VP, _INT, _VP]
    lib.launch_sum_rows_bf16.argtypes = [_VP, _VP, _VP, _INT, _INT, _VP]
    lib.launch_transpose_fp8.argtypes = [_VP, _VP, _INT, _INT, _VP]
    lib.launch_quant_1d_bf16.argtypes = [_VP, _VP, _VP, _VP, _VP, _INT, _I64, _INT, _INT, _VP]
    lib.launch_quant_bhsd_to_bshd.argtypes = [_VP] * 6 + [_INT, _INT, _INT, _INT, _INT, _INT, _VP]
    lib.launch_quant_dual_bf16.argtypes = [_VP] * 8 + [_INT, _INT, _INT, _INT, _INT, _VP]
    lib.launch_swiglu_delayed.argtypes = [_VP] * 6 + [_INT, _INT, _INT, _INT, _INT, _VP]
    lib.launch_dswiglu_dual_delayed.argtypes = [_VP] * 7 + [_INT, _INT, _INT, _INT, _VP]
    lib.launch_gelu_bias_delayed.argtypes = [_VP] * 8 + [_INT, _INT, _INT, _INT, _INT, _VP]
    lib.launch_dgelu_dual_delayed.argtypes = [_VP] * 9 + [_INT, _INT, _INT, _INT, _VP]
    lib.launch_ln_fwd_1024.argtypes = [_VP] * 6 + [_INT, _FLT, _VP]
    lib.launch_ln_bwd_1024.argtypes = [_VP] * 11 + [_INT, _VP]
    lib.launch_qwen_qkv_norm_rope_fwd.argtypes = [_VP] * 9 + [_INT, _INT, _INT, _INT, _INT, _FLT, _VP]
    lib.launch_qwen_qkv_norm_rope_bwd.argtypes = [_VP] * 14 + [_INT] + [_VP] * 2 + [_INT, _INT, _INT, _INT, _INT, _VP]
    lib.launch_vis_qkv_rope_fwd.argtypes = [_VP] * 6 + [_INT, _INT, _INT, _VP]
    lib.launch_vis_qkv_rope_bwd.argtypes = [_VP] * 8 + [_INT, _INT, _INT, _VP]
    lib.launch_rmsnorm_quant_delayed.argtypes = [_VP] * 10 + [_INT, _INT, _INT, _FLT, _INT, _INT, _VP]
    lib.launch_rmsnorm_bwd_2560.argtypes = [_VP] * 10 + [_INT, _VP, _INT, _VP]
    lib.launch_mmdit_joint_qkv_fwd.argtypes = [_VP] * 14 + [_INT, _INT, _INT, _INT, _FLT, _VP]
    lib.launch_mmdit_joint_qkv_bwd.argtypes = [_VP] * 20 + [_INT, _INT, _INT, _INT, _VP]
    lib.launch_mmdit_split_unpad_fwd.argtypes = [_VP, _VP, _VP, _INT, _INT, _INT, _INT, _VP]
    lib.launch_mmdit_split_unpad_bwd.argtypes = [_VP, _VP, _VP, _INT, _INT, _INT, _INT, _VP]
    lib.lt_fp8_matmul.argtypes = [_VP] * 8 + [_SZT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _VP]
    lib.lt_fp8_matmul.restype = _INT
    lib.lt_bf16_matmul.argtypes = [_INT] + [_VP] * 7 + [_SZT, _INT, _INT, _INT, _INT, _VP]
    lib.lt_bf16_matmul.restype = _INT
    lib.launch_adaln_modulate_quant_1536.argtypes = [_VP] * 9 + [_INT, _INT, _INT, _INT, _INT, _FLT, _INT, _INT, _VP]
    lib.launch_gated_add_fwd_1536.argtypes = [_VP] * 4 + [_INT, _INT, _INT, _INT, _VP]
    lib.launch_gated_bwd_1536.argtypes = [_VP] * 11 + [_INT, _INT, _INT, _INT, _INT, _VP]
    lib.launch_adaln_modulate_bwd_1536.argtypes = [_VP] * 8 + [_INT, _INT, _INT, _INT, _VP]
    lib.launch_cast_f32_to_bf16.argtypes = [_VP, _VP, _INT, _VP]
    lib.launch_mmdit_joint_qkv_48_fwd.argtypes = [_VP] * 14 + [_INT, _INT, _INT, _INT, _FLT, _VP]
    lib.launch_mmdit_joint_qkv_48_bwd.argtypes = [_VP] * 20 + [_INT, _INT, _INT, _INT, _VP]
    lib.launch_mmdit_split_48_fwd.argtypes = [_VP, _VP, _VP, _INT, _INT, _INT, _VP]
    lib.launch_mmdit_split_48_bwd.argtypes = [_VP, _VP, _VP, _INT, _INT, _INT, _VP]
    lib.launch_mmdit_split_48_bwd_bhsd.argtypes = [_VP, _VP, _VP, _INT, _INT, _INT, _VP]
    lib.launch_mmdit_split_quant_sub.argtypes = [
        _VP, _VP, _VP, _VP, _VP, _VP,
        _INT, _INT, _INT, _INT, _INT, _INT, _INT, _INT, _VP,
    ]
    lib.lt_set_autotune_enabled.argtypes = [_INT]
    lib.lt_serialized_entry_bytes.restype = _INT
    lib.lt_export_algos.argtypes = [_VP, _INT]
    lib.lt_export_algos.restype = _INT
    lib.lt_import_algos.argtypes = [_VP, _INT, _SZT]
    lib.launch_bf16_sum_sq_f64.argtypes = [_VP, _VP, _I64, _INT, _VP]
    lib.launch_compute_adamw_inv_scale.argtypes = [_VP, _VP, _VP, _FLT, _FLT, _FLT, _INT, _VP]
    lib.launch_fused_adamw_bf16_io.argtypes = [
        _VP, _VP, _VP, _VP, _VP, _VP,
        _FLT, _FLT, _FLT, _FLT, _FLT, _FLT, _FLT, _I64, _VP,
    ]
    lib.launch_mmdit_joint_qkv_48_bwd_e5_dual.argtypes = (
        [_VP] * 24 + [_INT, _INT] + [_VP] * 4 + [_INT] * 5 + [_VP]
    )
    lib.launch_scatter_rows_2560_bf16.argtypes = [_VP, _VP, _VP, _INT, _VP]
    lib.launch_gather_and_zero_rows_2560_bf16.argtypes = [_VP, _VP, _VP, _INT, _INT, _VP]
    lib.launch_deepstack_add_2560_bf16.argtypes = [_VP, _VP, _VP, _INT, _VP]
    lib.lt_clear_all_caches.argtypes = []
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        lib.lt_set_autotune_enabled(1 if dist.get_rank() == 0 else 0)
    _LIB_SM90 = lib
    return lib


def _ptr(t: torch.Tensor | None):
    return _VP(t.data_ptr()) if t is not None else _VP(0)


def _stream(dev: torch.device):
    return _VP(torch.cuda.current_stream(dev).cuda_stream)


def _as_bf16_contig(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    return x if x.is_contiguous() else x.contiguous()


# -----------------------------------------------------------------------------
# 1. Per-Device Delayed-Scaling Pool + Persistent cublasLt Workspace
# -----------------------------------------------------------------------------
class DelayedScalingPool:
    def __init__(self, device: torch.device):
        self.device = device
        self.curr_scale = torch.ones(MAX_SLOTS, dtype=torch.float32, device=device)
        self.curr_inv_scale = torch.ones(MAX_SLOTS, dtype=torch.float32, device=device)
        self.next_amax_i32 = torch.zeros(MAX_SLOTS, dtype=torch.int32, device=device)
        self.fp8_max_buf = torch.full((MAX_SLOTS,), FP8_E4M3_MAX, dtype=torch.float32, device=device)
        self.lt_workspace = torch.empty(LT_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        self.dbias_f32_ws = torch.empty(32768, dtype=torch.float32, device=device)
        self.bwd_sm_target = int(os.environ.get("WLA_LT_BWD_SM_TARGET", "0"))
        self.warmed_up = 0
        self.step_count = 0
        self.quant_stream = torch.cuda.Stream(device=device)
        self.quant_pending = False


_POOLS_V9: dict[int, DelayedScalingPool] = {}


def get_pool(device: torch.device) -> DelayedScalingPool:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    pool = _POOLS_V9.get(idx)
    if pool is None:
        pool = DelayedScalingPool(device)
        _POOLS_V9[idx] = pool
    return pool


def reset_fp8_pool(device: torch.device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    _POOLS_V9.pop(idx, None)


def begin_fp8_step(device: torch.device, slot_fp8_max_cpu: list[tuple[int, float]] | None = None):
    pool = get_pool(device)
    if pool.step_count == 0 and slot_fp8_max_cpu:
        with torch.no_grad():
            arr = torch.full((MAX_SLOTS,), FP8_E4M3_MAX, dtype=torch.float32)
            for s_id, fmax in slot_fp8_max_cpu:
                arr[s_id] = fmax
            pool.fp8_max_buf.copy_(arr.to(device))
    if pool.step_count > 0:
        get_cuda_lib().launch_finalize_delayed_scales(
            _ptr(pool.next_amax_i32), _ptr(pool.fp8_max_buf),
            _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale),
            MAX_SLOTS, _stream(device),
        )
        pool.warmed_up = 1
    else:
        pool.next_amax_i32.zero_()
        pool.warmed_up = 0
    pool.step_count += 1


# -----------------------------------------------------------------------------
# 2. Low-Level CUDA Quantization, Activation, LayerNorm, & cublasLt Wrappers
# -----------------------------------------------------------------------------
_DIRECT_IPG_HOOKS: dict[int, object] = {}


def register_direct_ipg_hook(dev: torch.device, hook_fn) -> None:
    idx = -1 if dev.index is None else int(dev.index)
    if hook_fn is None:
        _DIRECT_IPG_HOOKS.pop(idx, None)
    else:
        _DIRECT_IPG_HOOKS[idx] = hook_fn


def try_begin_direct_ipg_block(dev: torch.device, all_params: tuple, dw_params: tuple):
    fn = _DIRECT_IPG_HOOKS.get(-1 if dev.index is None else int(dev.index))
    if fn is None:
        return None
    return fn(all_params, dw_params)


def lt_fp8_mm(
    a_row: torch.Tensor,
    b_row: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    c_res: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    fast_accum: bool = True,
    is_bwd: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, K = a_row.shape
    if (K & 15) != 0:
        pad_k = 16 - (K & 15)
        a_row = F.pad(a_row.view(torch.uint8), (0, pad_k)).view(a_row.dtype)
        b_row = F.pad(b_row.view(torch.uint8), (0, pad_k)).view(b_row.dtype)
        K = a_row.shape[1]
    N = b_row.shape[0]
    dev = a_row.device
    pool = get_pool(dev)
    d_out = out if out is not None else torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    c_c = _as_bf16_contig(c_res) if c_res is not None else None
    b_c = _as_bf16_contig(bias) if bias is not None else None
    a_e5 = 1 if a_row.dtype == torch.float8_e5m2 else 0
    b_e5 = 1 if b_row.dtype == torch.float8_e5m2 else 0
    smt = pool.bwd_sm_target if is_bwd else 0
    rc = get_cuda_lib().lt_fp8_matmul(
        _ptr(a_row), _ptr(b_row), _ptr(c_c), _ptr(d_out),
        _ptr(scale_a), _ptr(scale_b), _ptr(b_c),
        _ptr(pool.lt_workspace), LT_WORKSPACE_BYTES,
        M, N, K, a_e5, b_e5, 1 if fast_accum else 0, smt, _stream(dev),
    )
    if rc == 0:
        return d_out
    res = torch._scaled_mm(a_row, b_row.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16, use_fast_accum=fast_accum)
    if b_c is not None:
        res = res + b_c
    if c_c is not None:
        res = res + c_c
    if out is not None:
        out.copy_(res)
        return out
    return res


def lt_bf16_fwd(
    x_row: torch.Tensor,
    w_row: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    c_res: torch.Tensor | None = None,
    gelu: bool = False,
    save_aux: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    xb = _as_bf16_contig(x_row)
    wb = _as_bf16_contig(w_row)
    M, K = xb.shape
    N = wb.shape[0]
    dev = xb.device
    pool = get_pool(dev)
    d_out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    bb = _as_bf16_contig(bias) if bias is not None else None
    cb = _as_bf16_contig(c_res) if c_res is not None else None
    aux = torch.empty((M, N), dtype=torch.bfloat16, device=dev) if (gelu and save_aux) else None
    if gelu:
        mode = BF16_FWD_GELU_AUX_BIAS if save_aux else BF16_FWD_GELU_BIAS
    else:
        mode = BF16_FWD_BIAS if bb is not None else BF16_FWD_DEFAULT
    rc = get_cuda_lib().lt_bf16_matmul(
        mode, _ptr(xb), _ptr(wb), _ptr(cb), _ptr(d_out), _ptr(bb), _ptr(aux),
        _ptr(pool.lt_workspace), LT_WORKSPACE_BYTES, M, N, K, 0, _stream(dev),
    )
    if rc == 0:
        return d_out, aux
    z = F.linear(xb, wb, bb)
    if gelu:
        return F.gelu(z, approximate="tanh"), (z if save_aux else None)
    if cb is not None:
        z = z + cb
    return z, None


def lt_bf16_dx(
    dy_row: torch.Tensor,
    w_row: torch.Tensor,
    *,
    z_aux: torch.Tensor | None = None,
    c_res: torch.Tensor | None = None,
    need_dgelu_bgrad: bool = False,
):
    dyb = _as_bf16_contig(dy_row)
    wb = _as_bf16_contig(w_row)
    M, N = dyb.shape
    K = wb.shape[1]
    dev = dyb.device
    pool = get_pool(dev)
    dx = torch.empty((M, K), dtype=torch.bfloat16, device=dev)
    cb = _as_bf16_contig(c_res) if c_res is not None else None
    ab = _as_bf16_contig(z_aux) if z_aux is not None else None
    if need_dgelu_bgrad:
        assert ab is not None
        db = torch.empty((K,), dtype=torch.bfloat16, device=dev)
        rc = get_cuda_lib().lt_bf16_matmul(
            BF16_BWD_DX_DGELU_BGRAD, _ptr(dyb), _ptr(wb), _VP(0), _ptr(dx), _ptr(db), _ptr(ab),
            _ptr(pool.lt_workspace), LT_WORKSPACE_BYTES, M, N, K, pool.bwd_sm_target, _stream(dev),
        )
        if rc == 0:
            return dx, db
        # Fallback if heuristic unsupported
        rc2 = get_cuda_lib().lt_bf16_matmul(
            BF16_BWD_DX_DGELU, _ptr(dyb), _ptr(wb), _VP(0), _ptr(dx), _VP(0), _ptr(ab),
            _ptr(pool.lt_workspace), LT_WORKSPACE_BYTES, M, N, K, pool.bwd_sm_target, _stream(dev),
        )
        if rc2 == 0:
            return dx, dx.float().sum(dim=0).to(torch.bfloat16)

    mode = BF16_BWD_DX_DGELU if ab is not None else BF16_BWD_DX_DEFAULT
    rc = get_cuda_lib().lt_bf16_matmul(
        mode, _ptr(dyb), _ptr(wb), _ptr(cb), _ptr(dx), _VP(0), _ptr(ab),
        _ptr(pool.lt_workspace), LT_WORKSPACE_BYTES, M, N, K, pool.bwd_sm_target, _stream(dev),
    )
    if rc == 0:
        return (dx, dx.float().sum(dim=0).to(torch.bfloat16)) if need_dgelu_bgrad else dx
    out = dyb @ wb
    if ab is not None:
        z_req = ab.detach().float().requires_grad_(True)
        with torch.enable_grad():
            g = F.gelu(z_req, approximate="tanh")
        (dz,) = torch.autograd.grad(g, (z_req,), out.float())
        out = dz.to(torch.bfloat16)
    if cb is not None:
        out = out + cb
    if need_dgelu_bgrad:
        return out, out.float().sum(dim=0).to(torch.bfloat16)
    return out


def lt_bf16_dw_dbias(
    dy_row: torch.Tensor,
    x_row: torch.Tensor,
    *,
    need_dbias: bool = False,
    out: torch.Tensor | None = None,
):
    dyb = _as_bf16_contig(dy_row)
    xb = _as_bf16_contig(x_row)
    M, N = dyb.shape
    K = xb.shape[1]
    dev = dyb.device
    pool = get_pool(dev)
    dw = out if out is not None else torch.empty((N, K), dtype=torch.bfloat16, device=dev)
    db = torch.empty((N,), dtype=torch.bfloat16, device=dev) if need_dbias else None
    rc = get_cuda_lib().lt_bf16_matmul(
        BF16_BWD_DW_BGRADB, _ptr(dyb), _ptr(xb), _VP(0), _ptr(dw), _ptr(db), _VP(0),
        _ptr(pool.lt_workspace), LT_WORKSPACE_BYTES, M, N, K, pool.bwd_sm_target, _stream(dev),
    )
    if rc == 0:
        return dw, db
    dw_fb = dyb.t() @ xb
    if out is not None:
        out.copy_(dw_fb)
        dw = out
    else:
        dw = dw_fb
    if need_dbias:
        db = dyb.float().sum(dim=0).to(torch.bfloat16)
    return dw, db


def sum_rows_bf16_cuda(x_2d: torch.Tensor) -> torch.Tensor:
    xb = _as_bf16_contig(x_2d)
    M, N = xb.shape
    pool = get_pool(xb.device)
    db = torch.empty((N,), dtype=torch.bfloat16, device=xb.device)
    get_cuda_lib().launch_sum_rows_bf16(_ptr(xb), _ptr(db), _ptr(pool.dbias_f32_ws), M, N, _stream(xb.device))
    return db


def quantize_1d_slot(x_2d: torch.Tensor, slot_id: int, dtype: torch.dtype):
    xb = _as_bf16_contig(x_2d)
    pool = get_pool(xb.device)
    y = torch.empty(xb.shape, dtype=dtype, device=xb.device)
    is_e5 = 1 if dtype == torch.float8_e5m2 else 0
    get_cuda_lib().launch_quant_1d_bf16(
        _ptr(xb), _ptr(y), _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, xb.numel(), is_e5, pool.warmed_up, _stream(xb.device),
    )
    return y, pool.curr_scale[slot_id]


def quant_bhsd_to_bshd_1d_slot(x_bhsd: torch.Tensor, slot_id: int):
    xb = _as_bf16_contig(x_bhsd)
    B, H, S, D = xb.shape
    assert D == 128
    pool = get_pool(xb.device)
    y_row = torch.empty((B * S, H * 128), dtype=torch.float8_e4m3fn, device=xb.device)
    get_cuda_lib().launch_quant_bhsd_to_bshd(
        _ptr(xb), _ptr(y_row), _VP(0),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, B, H, S, 0, pool.warmed_up, _stream(xb.device),
    )
    return y_row, pool.curr_scale[slot_id]


def quant_bhsd_to_bshd_dual_slot(x_bhsd: torch.Tensor, slot_id: int):
    xb = _as_bf16_contig(x_bhsd)
    B, H, S, D = xb.shape
    assert D == 128
    pool = get_pool(xb.device)
    M = B * S
    N = H * 128
    y_row = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=xb.device)
    y_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=xb.device)
    get_cuda_lib().launch_quant_bhsd_to_bshd(
        _ptr(xb), _ptr(y_row), _ptr(y_t),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, B, H, S, 1, pool.warmed_up, _stream(xb.device),
    )
    return y_row, y_t, pool.curr_scale[slot_id]


def quantize_dual_slot_into(
    x_2d: torch.Tensor, y_row: torch.Tensor, y_t: torch.Tensor, slot_id: int,
    dbias_out: torch.Tensor | None = None,
):
    xb = _as_bf16_contig(x_2d)
    pool = get_pool(xb.device)
    M, N = xb.shape
    is_e5 = 1 if y_row.dtype == torch.float8_e5m2 else 0
    get_cuda_lib().launch_quant_dual_bf16(
        _ptr(xb), _ptr(y_row), _ptr(y_t), _ptr(dbias_out), _ptr(pool.dbias_f32_ws),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, M, N, is_e5, pool.warmed_up, _stream(xb.device),
    )
    return pool.curr_scale[slot_id]


def quantize_dual_slot(x_2d: torch.Tensor, slot_id: int, dtype: torch.dtype, need_dbias: bool = False):
    M, N = x_2d.shape
    dev = x_2d.device
    y_row = torch.empty((M, N), dtype=dtype, device=dev)
    y_t = torch.empty((N, M), dtype=dtype, device=dev)
    dbias = torch.empty((N,), dtype=torch.bfloat16, device=dev) if need_dbias else None
    scale = quantize_dual_slot_into(x_2d, y_row, y_t, slot_id, dbias_out=dbias)
    if need_dbias:
        return y_row, y_t, scale, dbias
    return y_row, y_t, scale


def swiglu_quant_1d_slot(gu_2d: torch.Tensor, slot_id: int):
    gub = _as_bf16_contig(gu_2d)
    pool = get_pool(gub.device)
    M = gub.shape[0]
    N = gub.shape[1] // 2
    h_row = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=gub.device)
    get_cuda_lib().launch_swiglu_delayed(
        _ptr(gub), _ptr(h_row), _VP(0), _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, M, N, 0, pool.warmed_up, _stream(gub.device),
    )
    return h_row, pool.curr_scale[slot_id]


def swiglu_quant_dual_slot(gu_2d: torch.Tensor, slot_id: int, need_row: bool = True):
    gub = _as_bf16_contig(gu_2d)
    pool = get_pool(gub.device)
    M = gub.shape[0]
    N = gub.shape[1] // 2
    can_col_only = (not need_row) and ((M & 3) == 0) and ((N & 3) == 0)
    h_row = None if can_col_only else torch.empty((M, N), dtype=torch.float8_e4m3fn, device=gub.device)
    h_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=gub.device)
    get_cuda_lib().launch_swiglu_delayed(
        _ptr(gub), _ptr(h_row), _ptr(h_t), _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, M, N, 1, pool.warmed_up, _stream(gub.device),
    )
    return h_row, h_t, pool.curr_scale[slot_id]


def dswiglu_quant_dual_slot(dh_2d: torch.Tensor, gu_2d: torch.Tensor, slot_id: int):
    dhb = _as_bf16_contig(dh_2d)
    gub = _as_bf16_contig(gu_2d)
    pool = get_pool(dhb.device)
    M, N = dhb.shape
    dgu_row = torch.empty((M, 2 * N), dtype=torch.float8_e5m2, device=dhb.device)
    dgu_t = torch.empty((2 * N, M), dtype=torch.float8_e5m2, device=dhb.device)
    get_cuda_lib().launch_dswiglu_dual_delayed(
        _ptr(dhb), _ptr(gub), _ptr(dgu_row), _ptr(dgu_t),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, M, N, pool.warmed_up, _stream(dhb.device),
    )
    return dgu_row, dgu_t, pool.curr_scale[slot_id]


def gelu_bias_quant_1d_slot(z_2d: torch.Tensor, b1: torch.Tensor, slot_id: int):
    zb = _as_bf16_contig(z_2d)
    bb = _as_bf16_contig(b1)
    pool = get_pool(zb.device)
    M, N = zb.shape
    h_row = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=zb.device)
    get_cuda_lib().launch_gelu_bias_delayed(
        _ptr(zb), _ptr(bb), _VP(0), _ptr(h_row), _VP(0),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, M, N, 0, pool.warmed_up, _stream(zb.device),
    )
    return h_row, pool.curr_scale[slot_id]


def gelu_bias_quant_dual_slot(z_2d: torch.Tensor, b1: torch.Tensor, slot_id: int):
    zb = _as_bf16_contig(z_2d)
    bb = _as_bf16_contig(b1)
    pool = get_pool(zb.device)
    M, N = zb.shape
    zb_out = torch.empty((M, N), dtype=torch.bfloat16, device=zb.device)
    h_row = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=zb.device)
    h_t = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=zb.device)
    get_cuda_lib().launch_gelu_bias_delayed(
        _ptr(zb), _ptr(bb), _ptr(zb_out), _ptr(h_row), _ptr(h_t),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, M, N, 1, pool.warmed_up, _stream(zb.device),
    )
    return zb_out, h_row, h_t, pool.curr_scale[slot_id]


def dgelu_quant_dual_slot(dh_2d: torch.Tensor, zb_2d: torch.Tensor, b1_dtype: torch.dtype, slot_id: int):
    dhb = _as_bf16_contig(dh_2d)
    zbb = _as_bf16_contig(zb_2d)
    pool = get_pool(dhb.device)
    M, N = dhb.shape
    dz_row = torch.empty((M, N), dtype=torch.float8_e5m2, device=dhb.device)
    dz_t = torch.empty((N, M), dtype=torch.float8_e5m2, device=dhb.device)
    db1_bf16 = torch.empty((N,), dtype=torch.bfloat16, device=dhb.device)
    get_cuda_lib().launch_dgelu_dual_delayed(
        _ptr(dhb), _ptr(zbb), _ptr(dz_row), _ptr(dz_t), _ptr(db1_bf16), _ptr(pool.dbias_f32_ws),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, M, N, pool.warmed_up, _stream(dhb.device),
    )
    db1_out = db1_bf16 if b1_dtype == torch.bfloat16 else db1_bf16.to(b1_dtype)
    return dz_row, dz_t, db1_out, pool.curr_scale[slot_id]


def rmsnorm_quant_1d_slot(x_2d: torch.Tensor, norm_w: torch.Tensor, eps: float, slot_id: int,
                          res_in: torch.Tensor | None = None):
    xb = _as_bf16_contig(x_2d)
    wb = _as_bf16_contig(norm_w)
    pool = get_pool(xb.device)
    M, D = xb.shape
    y_row = torch.empty((M, D), dtype=torch.float8_e4m3fn, device=xb.device)
    rb = _as_bf16_contig(res_in) if res_in is not None else None
    sum_out = torch.empty((M, D), dtype=torch.bfloat16, device=xb.device) if rb is not None else None
    get_cuda_lib().launch_rmsnorm_quant_delayed(
        _ptr(xb), _ptr(rb), _ptr(sum_out), _ptr(wb),
        _ptr(y_row), _VP(0), _VP(0),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, M, D, eps, 0, pool.warmed_up, _stream(xb.device),
    )
    return sum_out, y_row, pool.curr_scale[slot_id]


def rmsnorm_quant_dual_slot(x_2d: torch.Tensor, norm_w: torch.Tensor, eps: float, slot_id: int,
                            res_in: torch.Tensor | None = None):
    xb = _as_bf16_contig(x_2d)
    wb = _as_bf16_contig(norm_w)
    pool = get_pool(xb.device)
    M, D = xb.shape
    y_row = torch.empty((M, D), dtype=torch.float8_e4m3fn, device=xb.device)
    y_t = torch.empty((D, M), dtype=torch.float8_e4m3fn, device=xb.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=xb.device)
    rb = _as_bf16_contig(res_in) if res_in is not None else None
    sum_out = torch.empty((M, D), dtype=torch.bfloat16, device=xb.device) if rb is not None else None
    get_cuda_lib().launch_rmsnorm_quant_delayed(
        _ptr(xb), _ptr(rb), _ptr(sum_out), _ptr(wb),
        _ptr(y_row), _ptr(y_t), _ptr(rstd),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, M, D, eps, 1, pool.warmed_up, _stream(xb.device),
    )
    return sum_out, y_row, y_t, rstd, pool.curr_scale[slot_id]


def rmsnorm_bwd_cuda(dy_2d: torch.Tensor, x_2d: torch.Tensor, norm_w: torch.Tensor, rstd: torch.Tensor,
                     d_res_in: torch.Tensor | None = None, quant_slot_id: int = -1):
    dyb = _as_bf16_contig(dy_2d)
    xb = _as_bf16_contig(x_2d)
    wb = _as_bf16_contig(norm_w)
    drb = _as_bf16_contig(d_res_in) if d_res_in is not None else None
    M, D = xb.shape
    if D == 2560:
        pool = get_pool(xb.device)
        dx = torch.empty((M, D), dtype=torch.bfloat16, device=xb.device)
        dw_f32 = torch.zeros((D,), dtype=torch.float32, device=xb.device)
        do_fused_quant = (quant_slot_id >= 0) and (pool.warmed_up == 1)
        dx_row = torch.empty((M, D), dtype=torch.float8_e5m2, device=xb.device) if do_fused_quant else None
        dx_t = torch.empty((D, M), dtype=torch.float8_e5m2, device=xb.device) if do_fused_quant else None
        get_cuda_lib().launch_rmsnorm_bwd_2560(
            _ptr(dyb), _ptr(drb), _ptr(xb), _ptr(wb), _ptr(rstd.contiguous()),
            _ptr(dx), _ptr(dx_row), _ptr(dx_t),
            _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32), quant_slot_id if do_fused_quant else 0,
            _ptr(dw_f32), M, _stream(xb.device),
        )
        dw_out = dw_f32.to(norm_w.dtype)
        if quant_slot_id >= 0:
            if not do_fused_quant:
                dx_row, dx_t, dx_s = quantize_dual_slot(dx, quant_slot_id, torch.float8_e5m2)
            else:
                dx_s = pool.curr_scale[quant_slot_id]
            return dx, dx_row, dx_t, dx_s, dw_out
        return dx, dw_out

    xf = xb.float()
    dyf = dyb.float()
    wf = wb.float()
    xh = xf * rstd[:, None]
    dw = (dyf * xh).sum(dim=0).to(norm_w.dtype)
    g = dyf * wf[None, :]
    mean_gs = (g * xh).mean(dim=-1, keepdim=True)
    dx = (rstd[:, None] * (g - xh * mean_gs)).to(torch.bfloat16)
    if drb is not None:
        dx = drb + dx
    if quant_slot_id >= 0:
        dx_row, dx_t, dx_s = quantize_dual_slot(dx, quant_slot_id, torch.float8_e5m2)
        return dx, dx_row, dx_t, dx_s, dw
    return dx, dw


def layernorm_1024_fwd_cuda(x_2d: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float, save_stats: bool):
    xb = _as_bf16_contig(x_2d)
    gb = _as_bf16_contig(gamma)
    bb = _as_bf16_contig(beta)
    M, D = xb.shape
    assert D == 1024, D
    y = torch.empty((M, D), dtype=torch.bfloat16, device=xb.device)
    mean = torch.empty((M,), dtype=torch.float32, device=xb.device) if save_stats else None
    rstd = torch.empty((M,), dtype=torch.float32, device=xb.device) if save_stats else None
    get_cuda_lib().launch_ln_fwd_1024(
        _ptr(xb), _ptr(gb), _ptr(bb), _ptr(y), _ptr(mean), _ptr(rstd), M, eps, _stream(xb.device)
    )
    return y, mean, rstd


def layernorm_1024_bwd_cuda(
    dy_2d: torch.Tensor, x_2d: torch.Tensor, gamma: torch.Tensor,
    mean: torch.Tensor, rstd: torch.Tensor, d_res_in: torch.Tensor | None = None,
    need_dres_sum: bool = False,
):
    dyb = _as_bf16_contig(dy_2d)
    xb = _as_bf16_contig(x_2d)
    gb = _as_bf16_contig(gamma)
    drb = _as_bf16_contig(d_res_in) if d_res_in is not None else None
    M, D = xb.shape
    assert D == 1024, D
    pool = get_pool(xb.device)
    dx = torch.empty((M, D), dtype=torch.bfloat16, device=xb.device)
    dg_f32 = torch.zeros((D,), dtype=torch.float32, device=xb.device)
    db_f32 = torch.zeros((D,), dtype=torch.float32, device=xb.device)
    dres_sum_bf16 = torch.empty((D,), dtype=torch.bfloat16, device=xb.device) if (need_dres_sum and drb is not None) else None
    get_cuda_lib().launch_ln_bwd_1024(
        _ptr(dyb), _ptr(drb), _ptr(xb), _ptr(gb),
        _ptr(mean.contiguous()), _ptr(rstd.contiguous()),
        _ptr(dx), _ptr(dg_f32), _ptr(db_f32),
        _ptr(dres_sum_bf16), _ptr(pool.dbias_f32_ws),
        M, _stream(xb.device),
    )
    if need_dres_sum:
        return dx, dg_f32.to(gamma.dtype), db_f32.to(gamma.dtype), dres_sum_bf16
    return dx, dg_f32.to(gamma.dtype), db_f32.to(gamma.dtype)


def qwen_qkv_norm_rope_raw_fwd(
    qkv: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, h_q: int, h_kv: int, save_rstd: bool,
    bhsd: bool = False,
):
    B, S, _ = qkv.shape
    T = B * S
    qkv_c = _as_bf16_contig(qkv)
    if bhsd:
        q_out = torch.empty((B, h_q, S, 128), dtype=torch.bfloat16, device=qkv.device)
        k_out = torch.empty((B, h_kv, S, 128), dtype=torch.bfloat16, device=qkv.device)
        v_out = torch.empty((B, h_kv, S, 128), dtype=torch.bfloat16, device=qkv.device)
    else:
        q_out = torch.empty((B, S, h_q, 128), dtype=torch.bfloat16, device=qkv.device)
        k_out = torch.empty((B, S, h_kv, 128), dtype=torch.bfloat16, device=qkv.device)
        v_out = torch.empty((B, S, h_kv, 128), dtype=torch.bfloat16, device=qkv.device)
    rstd = torch.empty((T, h_q + h_kv), dtype=torch.float32, device=qkv.device) if save_rstd else None
    get_cuda_lib().launch_qwen_qkv_norm_rope_fwd(
        _ptr(qkv_c), _ptr(_as_bf16_contig(qw)), _ptr(_as_bf16_contig(kw)),
        _ptr(_as_bf16_contig(cos)), _ptr(_as_bf16_contig(sin)),
        _ptr(q_out), _ptr(k_out), _ptr(v_out), _ptr(rstd),
        B, S, h_q, h_kv, 1 if bhsd else 0, 1e-6, _stream(qkv.device),
    )
    return q_out, k_out, v_out, rstd


def qwen_qkv_norm_rope_raw_bwd(
    dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
    qkv: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, rstd: torch.Tensor,
    h_q: int, h_kv: int, *, bhsd: bool = False, quant_slot_id: int = -1,
):
    B, S, D_qkv = qkv.shape
    M = B * S
    qkv_c = _as_bf16_contig(qkv)
    pool = get_pool(qkv.device)
    dqw_f32 = torch.zeros(128, dtype=torch.float32, device=qkv.device)
    dkw_f32 = torch.zeros(128, dtype=torch.float32, device=qkv.device)
    do_fused_quant = (quant_slot_id >= 0) and (pool.warmed_up == 1)
    dqkv_bf16 = None if do_fused_quant else torch.empty_like(qkv_c)
    dqkv_row = torch.empty((M, D_qkv), dtype=torch.float8_e5m2, device=qkv.device) if do_fused_quant else None
    dqkv_t = torch.empty((D_qkv, M), dtype=torch.float8_e5m2, device=qkv.device) if do_fused_quant else None
    dq_c, dk_c, dv_c = _as_bf16_contig(dq), _as_bf16_contig(dk), _as_bf16_contig(dv)
    qw_c, kw_c = _as_bf16_contig(qw), _as_bf16_contig(kw)
    cos_c, sin_c = _as_bf16_contig(cos), _as_bf16_contig(sin)
    get_cuda_lib().launch_qwen_qkv_norm_rope_bwd(
        _ptr(dq_c), _ptr(dk_c), _ptr(dv_c),
        _ptr(qkv_c), _ptr(qw_c), _ptr(kw_c),
        _ptr(cos_c), _ptr(sin_c), _ptr(rstd),
        _ptr(dqkv_bf16), _ptr(dqkv_row), _ptr(dqkv_t),
        _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32), quant_slot_id if do_fused_quant else 0,
        _ptr(dqw_f32), _ptr(dkw_f32), B, S, h_q, h_kv, 1 if bhsd else 0, _stream(qkv.device),
    )
    dqw_o, dkw_o = dqw_f32.to(qw.dtype), dkw_f32.to(kw.dtype)
    if quant_slot_id >= 0:
        if not do_fused_quant:
            dqkv_row, dqkv_t, dqkv_s = quantize_dual_slot(dqkv_bf16.view(M, D_qkv), quant_slot_id, torch.float8_e5m2)
        else:
            dqkv_s = pool.curr_scale[quant_slot_id]
        return dqkv_row, dqkv_t, dqkv_s, dqw_o, dkw_o
    return dqkv_bf16, dqw_o, dkw_o


def vis_qkv_rope_raw_fwd(qkv_2d: torch.Tensor, cos_f32: torch.Tensor, sin_f32: torch.Tensor, seq_per_cam: int, num_heads: int):
    total_tokens = qkv_2d.shape[0]
    B_vis = total_tokens // seq_per_cam
    qkv_c = _as_bf16_contig(qkv_2d)
    q_out = torch.empty((B_vis, seq_per_cam, num_heads, 64), dtype=torch.bfloat16, device=qkv_2d.device)
    k_out = torch.empty((B_vis, seq_per_cam, num_heads, 64), dtype=torch.bfloat16, device=qkv_2d.device)
    v_out = torch.empty((B_vis, seq_per_cam, num_heads, 64), dtype=torch.bfloat16, device=qkv_2d.device)
    get_cuda_lib().launch_vis_qkv_rope_fwd(
        _ptr(qkv_c), _ptr(cos_f32), _ptr(sin_f32),
        _ptr(q_out), _ptr(k_out), _ptr(v_out),
        total_tokens, seq_per_cam, num_heads, _stream(qkv_2d.device),
    )
    return q_out, k_out, v_out


def vis_qkv_rope_raw_bwd(
    dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
    cos_f32: torch.Tensor, sin_f32: torch.Tensor,
    seq_per_cam: int, num_heads: int, *, need_dbias: bool = False,
):
    B_vis, S_vis, _, _ = dq.shape
    total_tokens = B_vis * S_vis
    cols = 3 * num_heads * 64
    pool = get_pool(dq.device)
    dqkv = torch.empty((total_tokens, cols), dtype=torch.bfloat16, device=dq.device)
    dbias = torch.empty((cols,), dtype=torch.bfloat16, device=dq.device) if need_dbias else None
    dq_c, dk_c, dv_c = _as_bf16_contig(dq), _as_bf16_contig(dk), _as_bf16_contig(dv)
    get_cuda_lib().launch_vis_qkv_rope_bwd(
        _ptr(dq_c), _ptr(dk_c), _ptr(dv_c),
        _ptr(cos_f32), _ptr(sin_f32), _ptr(dqkv),
        _ptr(dbias), _ptr(pool.dbias_f32_ws),
        total_tokens, seq_per_cam, num_heads, _stream(dq.device),
    )
    if need_dbias:
        return dqkv, dbias
    return dqkv


# -----------------------------------------------------------------------------
# 3. Pure C++ Flat Checkpointed Vision Block & Qwen Decoder Layer
# -----------------------------------------------------------------------------
def _fa4_raw_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, scale: float, need_lse: bool, seqused: torch.Tensor | None = None):
    res = fa4_mod.flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=None, cu_seqlens_k=None,
        seqused_q=seqused, seqused_k=seqused,
        max_seqlen_q=q.shape[1], max_seqlen_k=k.shape[1],
        causal=causal, softmax_scale=scale,
        deterministic=DETERMINISTIC, return_lse=need_lse,
    )
    if need_lse:
        return res[0], res[1]
    return (res[0] if isinstance(res, tuple) else res), None


def _fa4_raw_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor, dout: torch.Tensor, lse: torch.Tensor, causal: bool, scale: float, seqused: torch.Tensor | None = None):
    if seqused is not None:
        dq, dk, dv = [torch.zeros_like(t) for t in (q, k, v)]
    else:
        dq, dk, dv = [torch.empty_like(t) for t in (q, k, v)]
    return fa4_mod._flash_attn_bwd(
        q, k, v, out, _as_bf16_contig(dout), lse,
        softmax_scale=scale, causal=causal,
        cu_seqlens_q=None, cu_seqlens_k=None,
        seqused_q=seqused, seqused_k=seqused,
        max_seqlen_q=q.shape[1], max_seqlen_k=k.shape[1],
        deterministic=DETERMINISTIC, dq=dq, dk=dk, dv=dv,
    )

def _cuda_device(model):
    return next((t.device for t in chain(model.parameters(), model.buffers()) if t.is_cuda), None)


def wait_fp8_weight_quant(dev: torch.device):
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    pool = _POOLS_V9.get(idx)
    if pool is not None and pool.quant_pending:
        torch.cuda.current_stream(dev).wait_stream(pool.quant_stream)
        pool.quant_pending = False

def adaln_modulate_quant_1536_slot(
    x_3d: torch.Tensor, mod_3d: torch.Tensor, mod_offset: int, eps: float, slot_id: int,
    *, dual: bool, save_stats: bool,
):
    B, L, _ = x_3d.shape
    L_mod = mod_3d.shape[1]
    M = B * L
    dev = x_3d.device
    pool = get_pool(dev)
    y_row = torch.empty((M, 1536), dtype=torch.float8_e4m3fn, device=dev)
    y_col = torch.empty((1536, M), dtype=torch.float8_e4m3fn, device=dev) if dual else None
    mean_o = torch.empty((M,), dtype=torch.float32, device=dev) if save_stats else None
    rstd_o = torch.empty((M,), dtype=torch.float32, device=dev) if save_stats else None
    get_cuda_lib().launch_adaln_modulate_quant_1536(
        _ptr(x_3d), _ptr(mod_3d), _ptr(y_row), _ptr(y_col),
        _ptr(mean_o), _ptr(rstd_o),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_id, B, L, L_mod, mod_offset, eps,
        1 if dual else 0, pool.warmed_up, _stream(dev),
    )
    return y_row, y_col, mean_o, rstd_o, pool.curr_scale[slot_id]


def gated_add_fwd_1536_cuda(x_3d: torch.Tensor, sub_2d: torch.Tensor, mod_3d: torch.Tensor, mod_offset: int) -> torch.Tensor:
    B, L, _ = x_3d.shape
    out = torch.empty_like(x_3d)
    get_cuda_lib().launch_gated_add_fwd_1536(
        _ptr(x_3d), _ptr(sub_2d), _ptr(mod_3d), _ptr(out),
        B, L, mod_3d.shape[1], mod_offset, _stream(x_3d.device),
    )
    return out


def gated_bwd_quant_e5_1536_slot(
    dout_3d: torch.Tensor, sub_2d: torch.Tensor, mod_3d: torch.Tensor, mod_offset: int,
    d_mod_f32: torch.Tensor, slot_id: int, *, need_dbias: bool = True,
):
    B, L, _ = dout_3d.shape
    M = B * L
    dev = dout_3d.device
    pool = get_pool(dev)
    do_fused = (pool.warmed_up == 1)
    d_sub_bf16 = None if do_fused else torch.empty((M, 1536), dtype=torch.bfloat16, device=dev)
    d_sub_row = torch.empty((M, 1536), dtype=torch.float8_e5m2, device=dev) if do_fused else None
    d_sub_t = torch.empty((1536, M), dtype=torch.float8_e5m2, device=dev) if do_fused else None
    dbias = torch.empty((1536,), dtype=torch.bfloat16, device=dev) if (do_fused and need_dbias) else None
    get_cuda_lib().launch_gated_bwd_1536(
        _ptr(dout_3d), _ptr(sub_2d), _ptr(mod_3d),
        _ptr(d_sub_bf16), _ptr(d_sub_row), _ptr(d_sub_t),
        _ptr(dbias), _ptr(pool.dbias_f32_ws), _ptr(d_mod_f32),
        _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32), slot_id if do_fused else 0,
        B, L, mod_3d.shape[1], mod_offset, _stream(dev),
    )
    if not do_fused:
        if need_dbias:
            d_sub_row, d_sub_t, d_sub_s, dbias = quantize_dual_slot(d_sub_bf16, slot_id, torch.float8_e5m2, need_dbias=True)
        else:
            d_sub_row, d_sub_t, d_sub_s = quantize_dual_slot(d_sub_bf16, slot_id, torch.float8_e5m2, need_dbias=False)
        return d_sub_row, d_sub_t, d_sub_s, dbias
    return d_sub_row, d_sub_t, pool.curr_scale[slot_id], dbias


def adaln_modulate_bwd_1536_cuda(
    d_mod_x_2d: torch.Tensor, d_res_3d: torch.Tensor | None, x_3d: torch.Tensor,
    mod_3d: torch.Tensor, mod_offset: int, mean_t: torch.Tensor, rstd_t: torch.Tensor,
    d_mod_f32: torch.Tensor,
) -> torch.Tensor:
    B, L, _ = x_3d.shape
    dx = torch.empty_like(x_3d)
    get_cuda_lib().launch_adaln_modulate_bwd_1536(
        _ptr(d_mod_x_2d), _ptr(d_res_3d), _ptr(x_3d), _ptr(mod_3d),
        _ptr(mean_t), _ptr(rstd_t), _ptr(dx), _ptr(d_mod_f32),
        B, L, mod_3d.shape[1], mod_offset, _stream(x_3d.device),
    )
    return dx


def _mmdit_joint_qkv_raw_fwd(
    act_qkv_2d: torch.Tensor, txt_qkv_2d: torch.Tensor,
    nq_w: torch.Tensor, nk_w: torch.Tensor, naq_w: torch.Tensor, nak_w: torch.Tensor,
    act_cos: torch.Tensor, act_sin: torch.Tensor, txt_cos: torch.Tensor, txt_sin: torch.Tensor,
    B: int, S_act: int, S_txt: int, H: int, eps: float, save_rstd: bool, hd48: bool = False,
):
    S_tot = S_act + S_txt
    dev = act_qkv_2d.device
    hd = 48 if hd48 else 64
    jq = torch.empty((B, S_tot, H, hd), dtype=torch.bfloat16, device=dev)
    jk = torch.empty((B, S_tot, H, hd), dtype=torch.bfloat16, device=dev)
    jv = torch.empty((B, S_tot, H, hd), dtype=torch.bfloat16, device=dev)
    rstd = torch.empty((B * S_tot, 2 * H), dtype=torch.float32, device=dev) if save_rstd else None
    fn = get_cuda_lib().launch_mmdit_joint_qkv_48_fwd if hd48 else get_cuda_lib().launch_mmdit_joint_qkv_fwd
    fn(
        _ptr(act_qkv_2d), _ptr(txt_qkv_2d),
        _ptr(nq_w), _ptr(nk_w), _ptr(naq_w), _ptr(nak_w),
        _ptr(act_cos), _ptr(act_sin), _ptr(txt_cos), _ptr(txt_sin),
        _ptr(jq), _ptr(jk), _ptr(jv), _ptr(rstd),
        B, S_act, S_txt, H, eps, _stream(dev),
    )
    return jq, jk, jv, rstd


def _mmdit_joint_qkv_raw_bwd(
    dj_q: torch.Tensor, dj_k: torch.Tensor, dj_v: torch.Tensor,
    act_qkv_2d: torch.Tensor, txt_qkv_2d: torch.Tensor,
    nq_w: torch.Tensor, nk_w: torch.Tensor, naq_w: torch.Tensor, nak_w: torch.Tensor,
    act_cos: torch.Tensor, act_sin: torch.Tensor, txt_cos: torch.Tensor, txt_sin: torch.Tensor,
    rstd: torch.Tensor, B: int, S_act: int, S_txt: int, H: int, hd48: bool = False,
):
    dev = act_qkv_2d.device
    jq_c, jk_c, jv_c = _as_bf16_contig(dj_q), _as_bf16_contig(dj_k), _as_bf16_contig(dj_v)
    d_act = torch.empty_like(act_qkv_2d)
    d_txt = torch.empty_like(txt_qkv_2d)
    dnq = torch.zeros(48, dtype=torch.float32, device=dev)
    dnk = torch.zeros(48, dtype=torch.float32, device=dev)
    dnaq = torch.zeros(48, dtype=torch.float32, device=dev)
    dnak = torch.zeros(48, dtype=torch.float32, device=dev)
    fn = get_cuda_lib().launch_mmdit_joint_qkv_48_bwd if hd48 else get_cuda_lib().launch_mmdit_joint_qkv_bwd
    fn(
        _ptr(jq_c), _ptr(jk_c), _ptr(jv_c),
        _ptr(act_qkv_2d), _ptr(txt_qkv_2d),
        _ptr(nq_w), _ptr(nk_w), _ptr(naq_w), _ptr(nak_w),
        _ptr(act_cos), _ptr(act_sin), _ptr(txt_cos), _ptr(txt_sin), _ptr(rstd),
        _ptr(d_act), _ptr(d_txt), _ptr(dnq), _ptr(dnk), _ptr(dnaq), _ptr(dnak),
        B, S_act, S_txt, H, _stream(dev),
    )
    return d_act, d_txt, dnq, dnk, dnaq, dnak


def _mmdit_joint_qkv_48_bwd_e5_dual(
    dj_q: torch.Tensor, dj_k: torch.Tensor, dj_v: torch.Tensor,
    act_qkv_2d: torch.Tensor, txt_qkv_2d: torch.Tensor,
    nq_w: torch.Tensor, nk_w: torch.Tensor, naq_w: torch.Tensor, nak_w: torch.Tensor,
    act_cos: torch.Tensor, act_sin: torch.Tensor, txt_cos: torch.Tensor, txt_sin: torch.Tensor,
    rstd: torch.Tensor, B: int, S_act: int, S_txt: int, H: int,
    slot_act: int, slot_txt: int, bhsd: bool,
):
    dev = act_qkv_2d.device
    pool = get_pool(dev)
    jq_c, jk_c, jv_c = _as_bf16_contig(dj_q), _as_bf16_contig(dj_k), _as_bf16_contig(dj_v)
    M_act = B * S_act
    M_txt = B * S_txt
    da_row = torch.empty((M_act, 4608), dtype=torch.float8_e5m2, device=dev)
    da_t = torch.empty((4608, M_act), dtype=torch.float8_e5m2, device=dev)
    d_bqkv = torch.empty((4608,), dtype=torch.bfloat16, device=dev)
    dt_row = torch.empty((M_txt, 4608), dtype=torch.float8_e5m2, device=dev)
    dt_t = torch.empty((4608, M_txt), dtype=torch.float8_e5m2, device=dev)
    d_badd = torch.empty((4608,), dtype=torch.bfloat16, device=dev)
    dnq = torch.zeros(48, dtype=torch.float32, device=dev)
    dnk = torch.zeros(48, dtype=torch.float32, device=dev)
    dnaq = torch.zeros(48, dtype=torch.float32, device=dev)
    dnak = torch.zeros(48, dtype=torch.float32, device=dev)
    ws_a = pool.dbias_f32_ws[:4608]
    ws_t = pool.dbias_f32_ws[4608:9216]
    get_cuda_lib().launch_mmdit_joint_qkv_48_bwd_e5_dual(
        _ptr(jq_c), _ptr(jk_c), _ptr(jv_c),
        _ptr(act_qkv_2d), _ptr(txt_qkv_2d),
        _ptr(nq_w), _ptr(nk_w), _ptr(naq_w), _ptr(nak_w),
        _ptr(act_cos), _ptr(act_sin), _ptr(txt_cos), _ptr(txt_sin), _ptr(rstd),
        _ptr(da_row), _ptr(da_t), _ptr(d_bqkv), _ptr(ws_a),
        _ptr(dt_row), _ptr(dt_t), _ptr(d_badd), _ptr(ws_t),
        _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32), slot_act, slot_txt,
        _ptr(dnq), _ptr(dnk), _ptr(dnaq), _ptr(dnak),
        B, S_act, S_txt, H, 1 if bhsd else 0, _stream(dev),
    )
    return (
        da_row, da_t, pool.curr_scale[slot_act], d_bqkv,
        dt_row, dt_t, pool.curr_scale[slot_txt], d_badd,
        dnq, dnk, dnaq, dnak,
    )


def _mmdit_split_quant_fwd(
    j_tensor: torch.Tensor, B: int, S_act: int, S_txt: int, is_last: bool,
    slot_act: int, slot_txt: int, dual: bool, hd48: bool,
):
    dev = j_tensor.device
    pool = get_pool(dev)
    lib = get_cuda_lib()
    st = _stream(dev)
    S_tot = S_act + S_txt
    # Zero-copy layout detection:
    # j_tensor is either [B, 32, S_tot, 48] (bhsd_48) or [B, S_tot, 32, 48/64] (bshd)
    if not hd48:
        jc = _as_bf16_contig(j_tensor)
        layout_mode = 0
    elif j_tensor.shape[1] == 32 and j_tensor.is_contiguous():
        jc = j_tensor
        layout_mode = 2
    elif j_tensor.shape[1] == 32 and j_tensor.transpose(1, 2).is_contiguous():
        jc = j_tensor.transpose(1, 2)
        layout_mode = 1
    else:
        jc = _as_bf16_contig(j_tensor if j_tensor.shape[-1] == 48 and j_tensor.shape[1] == S_tot else j_tensor.transpose(1, 2))
        layout_mode = 1

    M_act = B * S_act
    aa_row = torch.empty((M_act, 1536), dtype=torch.float8_e4m3fn, device=dev)
    aa_t = torch.empty((1536, M_act), dtype=torch.float8_e4m3fn, device=dev) if dual else None
    lib.launch_mmdit_split_quant_sub(
        _ptr(jc), _ptr(aa_row), _ptr(aa_t),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_act, B, S_act, S_tot, 0, layout_mode, 1 if dual else 0, pool.warmed_up, st,
    )
    aa_s = pool.curr_scale[slot_act]
    if is_last:
        return aa_row, aa_t, aa_s, None, None, None

    M_txt = B * S_txt
    ta_row = torch.empty((M_txt, 1536), dtype=torch.float8_e4m3fn, device=dev)
    ta_t = torch.empty((1536, M_txt), dtype=torch.float8_e4m3fn, device=dev) if dual else None
    lib.launch_mmdit_split_quant_sub(
        _ptr(jc), _ptr(ta_row), _ptr(ta_t),
        _ptr(pool.curr_scale), _ptr(pool.curr_inv_scale), _ptr(pool.next_amax_i32),
        slot_txt, B, S_txt, S_tot, S_act, layout_mode, 1 if dual else 0, pool.warmed_up, st,
    )
    ta_s = pool.curr_scale[slot_txt]
    return aa_row, aa_t, aa_s, ta_row, ta_t, ta_s


def _mmdit_split_48_bwd_bhsd(d_act_2d: torch.Tensor, d_txt_2d: torch.Tensor | None, B: int, S_act: int, S_txt: int, H: int):
    dj_bh = torch.empty((B, H, S_act + S_txt, 48), dtype=torch.bfloat16, device=d_act_2d.device)
    get_cuda_lib().launch_mmdit_split_48_bwd_bhsd(
        _ptr(d_act_2d), _ptr(d_txt_2d), _ptr(dj_bh), B, S_act, S_txt, _stream(d_act_2d.device)
    )
    return dj_bh


def _mmdit_split_unpad_raw_fwd(j_out: torch.Tensor, B: int, S_act: int, S_txt: int, H: int, is_last: bool, hd48: bool = False):
    jc = _as_bf16_contig(j_out)
    act_attn = torch.empty((B * S_act, 1536), dtype=torch.bfloat16, device=jc.device)
    txt_attn = None if is_last else torch.empty((B * S_txt, 1536), dtype=torch.bfloat16, device=jc.device)
    if hd48:
        get_cuda_lib().launch_mmdit_split_48_fwd(
            _ptr(jc), _ptr(act_attn), _ptr(txt_attn), B, S_act, S_txt, _stream(jc.device)
        )
    else:
        get_cuda_lib().launch_mmdit_split_unpad_fwd(
            _ptr(jc), _ptr(act_attn), _ptr(txt_attn), B, S_act, S_txt, H, _stream(jc.device)
        )
    return act_attn, txt_attn


def _mmdit_split_unpad_raw_bwd(d_act_2d: torch.Tensor, d_txt_2d: torch.Tensor | None, B: int, S_act: int, S_txt: int, H: int, hd48: bool = False):
    hd = 48 if hd48 else 64
    dj = torch.empty((B, S_act + S_txt, H, hd), dtype=torch.bfloat16, device=d_act_2d.device)
    if hd48:
        get_cuda_lib().launch_mmdit_split_48_bwd(
            _ptr(d_act_2d), _ptr(d_txt_2d), _ptr(dj), B, S_act, S_txt, _stream(d_act_2d.device)
        )
    else:
        get_cuda_lib().launch_mmdit_split_unpad_bwd(
            _ptr(d_act_2d), _ptr(d_txt_2d), _ptr(dj), B, S_act, S_txt, H, _stream(d_act_2d.device)
        )
    return dj



class _FastMultimodalScatterFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs_embeds: torch.Tensor, state_embeds: torch.Tensor, state_idx_i32: torch.Tensor,
                image_embeds: torch.Tensor, vis_idx_i32: torch.Tensor):
        ctx.save_for_backward(state_idx_i32, vis_idx_i32)
        ctx.se_shape = state_embeds.shape
        ctx.ie_shape = image_embeds.shape
        ctx.orig_dtype = inputs_embeds.dtype
        out = _as_bf16_contig(inputs_embeds).clone()
        out_2d = out.view(-1, 2560)
        se_c = _as_bf16_contig(state_embeds).view(-1, 2560)
        ie_c = _as_bf16_contig(image_embeds).view(-1, 2560)
        lib = get_cuda_lib()
        st = _stream(out.device)
        lib.launch_scatter_rows_2560_bf16(_ptr(out_2d), _ptr(se_c), _ptr(state_idx_i32), int(state_idx_i32.numel()), st)
        lib.launch_scatter_rows_2560_bf16(_ptr(out_2d), _ptr(ie_c), _ptr(vis_idx_i32), int(vis_idx_i32.numel()), st)
        return out.to(inputs_embeds.dtype)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        state_idx_i32, vis_idx_i32 = ctx.saved_tensors
        d_inp = _as_bf16_contig(dout).clone()
        d_inp_2d = d_inp.view(-1, 2560)
        d_se = torch.empty(ctx.se_shape, dtype=torch.bfloat16, device=dout.device)
        d_ie = torch.empty(ctx.ie_shape, dtype=torch.bfloat16, device=dout.device)
        lib = get_cuda_lib()
        st = _stream(dout.device)
        lib.launch_gather_and_zero_rows_2560_bf16(_ptr(d_inp_2d), _ptr(d_se), _ptr(state_idx_i32), int(state_idx_i32.numel()), 1, st)
        lib.launch_gather_and_zero_rows_2560_bf16(_ptr(d_inp_2d), _ptr(d_ie), _ptr(vis_idx_i32), int(vis_idx_i32.numel()), 1, st)
        return d_inp.to(ctx.orig_dtype), d_se.to(ctx.orig_dtype), None, d_ie.to(ctx.orig_dtype), None


class _FastDeepStackAddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, visual_embeds: torch.Tensor, vis_idx_i32: torch.Tensor):
        ctx.save_for_backward(vis_idx_i32)
        ctx.vis_shape = visual_embeds.shape
        ctx.h_dtype = hidden_states.dtype
        ctx.v_dtype = visual_embeds.dtype
        out = _as_bf16_contig(hidden_states).clone()
        ve_c = _as_bf16_contig(visual_embeds).view(-1, 2560)
        get_cuda_lib().launch_deepstack_add_2560_bf16(
            _ptr(out.view(-1, 2560)), _ptr(ve_c), _ptr(vis_idx_i32), int(vis_idx_i32.numel()), _stream(out.device)
        )
        return out.to(hidden_states.dtype)

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (vis_idx_i32,) = ctx.saved_tensors
        d_h = _as_bf16_contig(dout)
        d_vis = torch.empty(ctx.vis_shape, dtype=torch.bfloat16, device=dout.device)
        get_cuda_lib().launch_gather_and_zero_rows_2560_bf16(
            _ptr(d_h.view(-1, 2560)), _ptr(d_vis), _ptr(vis_idx_i32), int(vis_idx_i32.numel()), 0, _stream(dout.device)
        )
        return d_h.to(ctx.h_dtype), d_vis.to(ctx.v_dtype), None
