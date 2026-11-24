// High-MFU SM90 CUDA C++ Fused Kernels + CUDA 12.8 cublasLt Autotuned Engine (v6)
// for UnifoLM-WLA FP8 Training on Hopper (H100 / H800)
//
// Key v6 Architectural Upgrades over v5:
// 1. PTX `__byte_perm` 64x64 single-pass dual-layout kernels (`quant_dual_64x64_prmt_k` with
//    optional fused FP32 `dbias` reduction, `swiglu_dual_64x64_prmt_k`, `dswiglu_dual_64x64_prmt_k`,
//    `gelu_bias_dual_64x64_prmt_k`), eliminating ~735 separate `transpose_fp8` launches & ~94 `dbias` sums.
// 2. CUDA 12.8 `cublasLt` 24-algo hardware autotuned Matmul Engine for BOTH:
//    - FP8 (`E4M3/E5M2 -> BF16`) with fused `beta=1.0f` residual add & `CUBLASLT_EPILOGUE_BIAS`.
//    - Vision BF16 (`BF16 -> BF16`) with hardware `CUBLASLT_EPILOGUE_BIAS`, `CUBLASLT_EPILOGUE_GELU_BIAS`,
//      `CUBLASLT_EPILOGUE_GELU_AUX_BIAS`, `CUBLASLT_EPILOGUE_DGELU`, `CUBLASLT_EPILOGUE_BGRADB`, & `beta=1.0f`.
// 3. Single-pass `uint4` register-resident Vision `LayerNorm(1024)` Fwd & Persistent Bwd (with fused `d_res_in`).
// 4. Retains all v5 single-pass Qwen `D=2560` RMSNorm+Quant, persistent RMSNorm Bwd, QKV+Norm+RoPE, & MMDiT Joint QKV.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <unordered_map>

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16_rn(v);
}

__device__ __forceinline__ uint8_t f32_to_e4m3_raw(float v) {
    float c = fminf(fmaxf(v, -448.0f), 448.0f);
    __nv_fp8_e4m3 f = __nv_fp8_e4m3(c);
    return *reinterpret_cast<uint8_t*>(&f);
}

__device__ __forceinline__ uint8_t f32_to_e5m2_raw(float v) {
    float c = fminf(fmaxf(v, -57344.0f), 57344.0f);
    __nv_fp8_e5m2 f = __nv_fp8_e5m2(c);
    return *reinterpret_cast<uint8_t*>(&f);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ void atomic_max_pos_f32(int32_t* addr, float val) {
    if (val > 0.0f) {
        atomicMax(addr, __float_as_int(val));
    }
}

__device__ __forceinline__ void block_reduce_atomic_max(float lmax, int32_t* next_amax, int slot_id) {
    lmax = warp_reduce_max(lmax);
    __shared__ float smax[8];
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) smax[wid] = lmax;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < (blockDim.x >> 5)) ? smax[lane] : 0.0f;
        v = warp_reduce_max(v);
        if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
    }
}

__device__ __forceinline__ float silu_f32(float g) {
    return g / (1.0f + expf(-g));
}

__device__ __forceinline__ float gelu_tanh_f32(float z) {
    float u = 0.7978845608028654f * (z + 0.044715f * z * z * z);
    float s = 1.0f / (1.0f + expf(-2.0f * u));
    float t = 2.0f * s - 1.0f;
    return 0.5f * z * (1.0f + t);
}

__device__ __forceinline__ float dgelu_tanh_f32(float dh, float z) {
    float u = 0.7978845608028654f * (z + 0.044715f * z * z * z);
    float s = 1.0f / (1.0f + expf(-2.0f * u));
    float t = 2.0f * s - 1.0f;
    float dt = (1.0f - t * t) * 0.7978845608028654f * (1.0f + 0.134145f * z * z);
    return dh * (0.5f * (1.0f + t) + 0.5f * z * dt);
}

// 6-instruction PTX register 4x4 byte transpose
__device__ __forceinline__ void transpose_4x4_bytes(
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
    uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3
) {
    uint32_t t01_lo = __byte_perm(r0, r1, 0x5140);
    uint32_t t01_hi = __byte_perm(r0, r1, 0x7362);
    uint32_t t23_lo = __byte_perm(r2, r3, 0x5140);
    uint32_t t23_hi = __byte_perm(r2, r3, 0x7362);
    c0 = __byte_perm(t01_lo, t23_lo, 0x5410);
    c1 = __byte_perm(t01_lo, t23_lo, 0x7632);
    c2 = __byte_perm(t01_hi, t23_hi, 0x5410);
    c3 = __byte_perm(t01_hi, t23_hi, 0x7632);
}

// ============================================================================
// 1. Delayed Scaling Pool Finalize, Calibration, & Row-Sum Fallback Kernels
// ============================================================================
__global__ void finalize_delayed_scales_k(
    int32_t* __restrict__ next_amax,
    const float* __restrict__ fp8_max,
    float* __restrict__ curr_scale,
    float* __restrict__ curr_inv_scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t raw = next_amax[idx];
        float amax = fmaxf(__int_as_float(raw), 1e-12f) * 1.02f;
        float fmax = fp8_max[idx];
        curr_scale[idx] = amax / fmax;
        curr_inv_scale[idx] = fmax / amax;
        next_amax[idx] = 0;
    }
}

__global__ void calib_scale_slot_k(
    const int32_t* __restrict__ next_amax,
    float* __restrict__ curr_scale,
    float* __restrict__ curr_inv_scale,
    int slot_id,
    float fp8_max
) {
    float amax = fmaxf(__int_as_float(next_amax[slot_id]), 1e-12f);
    curr_scale[slot_id] = amax / fp8_max;
    curr_inv_scale[slot_id] = fp8_max / amax;
}

__global__ void fast_amax_bf16_k(
    const __nv_bfloat16* __restrict__ x,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int64_t numel
) {
    float lmax = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t vec_n = numel >> 3;
    const uint4* x_v = reinterpret_cast<const uint4*>(x);
    for (int64_t i = tid; i < vec_n; i += stride) {
        uint4 v = x_v[i];
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            lmax = fmaxf(lmax, fabsf(bf16_to_f32(p[k])));
        }
    }
    for (int64_t i = (vec_n << 3) + tid; i < numel; i += stride) {
        lmax = fmaxf(lmax, fabsf(bf16_to_f32(x[i])));
    }
    block_reduce_atomic_max(lmax, next_amax, slot_id);
}

__global__ void sum_rows_bf16_f32_k(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ dbias_f32,
    int M,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    float acc = 0.0f;
    for (int row = blockIdx.y; row < M; row += gridDim.y) {
        acc += bf16_to_f32(x[(int64_t)row * N + col]);
    }
    atomicAdd(dbias_f32 + col, acc);
}

// High-bandwidth 128B-coalesced uint4 column reduction (64 BF16 cols x 32 rows per CTA, only grid.y atomicAdds/col)
__global__ void sum_rows_bf16_vec8_k(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ dbias_f32,
    int M,
    int N
) {
    __shared__ float smem[32][65];  // zero bank conflicts
    const int tx = threadIdx.x & 7;   // 0..7 -> 8 x uint4 = 64 BF16 cols
    const int ty = threadIdx.x >> 3;  // 0..31 -> 32 rows
    const int col0 = blockIdx.x * 64 + tx * 8;

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (col0 + 7 < N) {
        for (int row = blockIdx.y * 32 + ty; row < M; row += gridDim.y * 32) {
            uint4 v = *reinterpret_cast<const uint4*>(x + (int64_t)row * N + col0);
            const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
            #pragma unroll
            for (int k = 0; k < 8; ++k) acc[k] += bf16_to_f32(p[k]);
        }
    } else if (col0 < N) {
        for (int row = blockIdx.y * 32 + ty; row < M; row += gridDim.y * 32) {
            for (int k = 0; k < 8 && col0 + k < N; ++k) {
                acc[k] += bf16_to_f32(x[(int64_t)row * N + col0 + k]);
            }
        }
    }

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        smem[ty][tx * 8 + k] = acc[k];
    }
    __syncthreads();

    if (threadIdx.x < 64) {
        int c = blockIdx.x * 64 + threadIdx.x;
        if (c < N) {
            float s = 0.0f;
            #pragma unroll
            for (int r = 0; r < 32; ++r) s += smem[r][threadIdx.x];
            atomicAdd(dbias_f32 + c, s);
        }
    }
}

__global__ void cast_f32_to_bf16_k(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16(src[i]);
}

// ============================================================================
// 2. 1D Quant, 64x64 Transpose, & Single-Pass PRMT_64x64 Dual Kernels
// ============================================================================
template <bool IS_E5M2, bool RECORD_AMAX>
__global__ void quant_1d_bf16_k(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ y,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int64_t numel
) {
    const float inv_s = inv_base[slot_id];
    float lmax = 0.0f;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t vec_n = numel >> 3;
    const uint4* x_v = reinterpret_cast<const uint4*>(x);
    uint64_t* y_v = reinterpret_cast<uint64_t*>(y);

    for (int64_t i = tid; i < vec_n; i += stride) {
        uint4 v = x_v[i];
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
        uint8_t out_b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float fv = bf16_to_f32(p[k]);
            if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fabsf(fv));
            out_b[k] = IS_E5M2 ? f32_to_e5m2_raw(fv * inv_s) : f32_to_e4m3_raw(fv * inv_s);
        }
        y_v[i] = *reinterpret_cast<uint64_t*>(out_b);
    }
    for (int64_t i = (vec_n << 3) + tid; i < numel; i += stride) {
        float fv = bf16_to_f32(x[i]);
        if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fabsf(fv));
        y[i] = IS_E5M2 ? f32_to_e5m2_raw(fv * inv_s) : f32_to_e4m3_raw(fv * inv_s);
    }
    if constexpr (RECORD_AMAX) {
        block_reduce_atomic_max(lmax, next_amax, slot_id);
    }
}

__global__ void transpose_fp8_64x64_vec4_k(
    const uint8_t* __restrict__ x,
    uint8_t* __restrict__ y_t,
    int M,
    int N
) {
    __shared__ alignas(16) uint8_t smem[64][68];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int n0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;

    const int col = n0 + tx * 4;
    #pragma unroll
    for (int r_base = 0; r_base < 64; r_base += 16) {
        int r = r_base + ty;
        int row = m0 + r;
        uint32_t v = (row < M && col < N)
            ? *reinterpret_cast<const uint32_t*>(x + (int64_t)row * N + col)
            : 0u;
        *reinterpret_cast<uint32_t*>(&smem[r][tx * 4]) = v;
    }
    __syncthreads();

    const int out_m = m0 + tx * 4;
    #pragma unroll
    for (int c_base = 0; c_base < 64; c_base += 16) {
        int c = c_base + ty;
        int out_n = n0 + c;
        if (out_n < N && out_m < M) {
            uint32_t packed = (uint32_t)smem[tx * 4 + 0][c]
                            | ((uint32_t)smem[tx * 4 + 1][c] << 8)
                            | ((uint32_t)smem[tx * 4 + 2][c] << 16)
                            | ((uint32_t)smem[tx * 4 + 3][c] << 24);
            *reinterpret_cast<uint32_t*>(y_t + (int64_t)out_n * M + out_m) = packed;
        }
    }
}

__global__ void transpose_fp8_64x64_scalar_k(
    const uint8_t* __restrict__ x,
    uint8_t* __restrict__ y_t,
    int M,
    int N
) {
    __shared__ uint8_t smem[64][68];
    const int tx = threadIdx.x & 63;
    const int ty = threadIdx.x >> 6;
    const int n0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;

    const int col = n0 + tx;
    #pragma unroll
    for (int r = 0; r < 64; r += 4) {
        int row = m0 + ty + r;
        smem[ty + r][tx] = (row < M && col < N) ? x[(int64_t)row * N + col] : 0;
    }
    __syncthreads();

    const int out_m = m0 + tx;
    #pragma unroll
    for (int c = 0; c < 64; c += 4) {
        int out_n = n0 + ty + c;
        if (out_n < N && out_m < M) {
            y_t[(int64_t)out_n * M + out_m] = smem[tx][ty + c];
        }
    }
}

// Single-pass 64x64 PRMT Dual Quant + Optional Fused FP32 dbias reduction (M%4==0 && N%4==0)
template <bool IS_E5M2, bool RECORD_AMAX, bool HAS_DBIAS>
__global__ void quant_dual_64x64_prmt_k(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ y_row,
    uint8_t* __restrict__ y_col,
    float* __restrict__ dbias_f32,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    __shared__ uint32_t smem_u32[64][17];  // 1088 x 4B slots, gcd(17,32)==1 (zero bank conflicts)
    __shared__ float smax[8];

    const float inv_s = inv_base[slot_id];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int n0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;
    const int col = n0 + tx * 4;
    const int row0 = m0 + ty * 4;

    float lmax = 0.0f;
    float bacc0 = 0.0f, bacc1 = 0.0f, bacc2 = 0.0f, bacc3 = 0.0f;
    uint32_t r_u32[4] = {0u, 0u, 0u, 0u};

    if (col < N) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int row = row0 + k;
            if (row < M) {
                uint2 xv = *reinterpret_cast<const uint2*>(x + (int64_t)row * N + col);
                const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
                float2 f0 = __bfloat1622float2(p2[0]);
                float2 f1 = __bfloat1622float2(p2[1]);
                if constexpr (HAS_DBIAS) {
                    bacc0 += f0.x; bacc1 += f0.y; bacc2 += f1.x; bacc3 += f1.y;
                }
                if constexpr (RECORD_AMAX) {
                    lmax = fmaxf(lmax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
                }
                uint8_t b0 = IS_E5M2 ? f32_to_e5m2_raw(f0.x * inv_s) : f32_to_e4m3_raw(f0.x * inv_s);
                uint8_t b1 = IS_E5M2 ? f32_to_e5m2_raw(f0.y * inv_s) : f32_to_e4m3_raw(f0.y * inv_s);
                uint8_t b2 = IS_E5M2 ? f32_to_e5m2_raw(f1.x * inv_s) : f32_to_e4m3_raw(f1.x * inv_s);
                uint8_t b3 = IS_E5M2 ? f32_to_e5m2_raw(f1.y * inv_s) : f32_to_e4m3_raw(f1.y * inv_s);
                uint32_t packed = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
                r_u32[k] = packed;
                if (y_row != nullptr) {
                    *reinterpret_cast<uint32_t*>(y_row + (int64_t)row * N + col) = packed;
                }
            }
        }
    }

    uint32_t c0, c1, c2, c3;
    transpose_4x4_bytes(r_u32[0], r_u32[1], r_u32[2], r_u32[3], c0, c1, c2, c3);
    smem_u32[tx * 4 + 0][ty] = c0;
    smem_u32[tx * 4 + 1][ty] = c1;
    smem_u32[tx * 4 + 2][ty] = c2;
    smem_u32[tx * 4 + 3][ty] = c3;
    __syncthreads();

    const int out_m = m0 + tx * 4;
    const int out_n0 = n0 + ty * 4;
    if (out_m < M) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int out_n = out_n0 + k;
            if (out_n < N) {
                *reinterpret_cast<uint32_t*>(y_col + (int64_t)out_n * M + out_m) = smem_u32[ty * 4 + k][tx];
            }
        }
    }

    if constexpr (HAS_DBIAS) {
        __syncthreads();
        float* sb = reinterpret_cast<float*>(&smem_u32[0][0]);  // 16 * 64 = 1024 <= 1088 floats
        sb[ty * 64 + tx * 4 + 0] = bacc0;
        sb[ty * 64 + tx * 4 + 1] = bacc1;
        sb[ty * 64 + tx * 4 + 2] = bacc2;
        sb[ty * 64 + tx * 4 + 3] = bacc3;
        __syncthreads();
        if (threadIdx.x < 64) {
            int c_idx = threadIdx.x;
            if (n0 + c_idx < N) {
                float s = 0.0f;
                #pragma unroll
                for (int r = 0; r < 16; ++r) s += sb[r * 64 + c_idx];
                atomicAdd(dbias_f32 + n0 + c_idx, s);
            }
        }
    }

    if constexpr (RECORD_AMAX) {
        lmax = warp_reduce_max(lmax);
        int wid = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        if (lane == 0) smax[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? smax[lane] : 0.0f;
            v = warp_reduce_max(v);
            if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
        }
    }
}

// Fused (B, H, S, 128) -> (B*S, H*128) 1D & 64x64 PRMT Dual Quantization (Zero-Copy cuDNN SDPA Out)
template <bool RECORD_AMAX>
__global__ void quant_bhsd_to_bshd_1d_k(
    const __nv_bfloat16* __restrict__ x_bhsd,
    uint8_t* __restrict__ y_row,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int B,
    int H,
    int S
) {
    const float inv_s = inv_base[slot_id];
    const int64_t total_vec8 = (int64_t)B * S * H * 16;  // 128 / 8 = 16
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float lmax = 0.0f;

    for (int64_t idx8 = tid; idx8 < total_vec8; idx8 += stride) {
        int d = (int)(idx8 & 15) << 3;
        int64_t rem = idx8 >> 4;
        int h = (int)(rem % H);
        int m = (int)(rem / H);
        int b = m / S;
        int s = m - b * S;

        int64_t in_off = (((int64_t)b * H + h) * S + s) * 128 + d;
        int64_t out_off = ((int64_t)m * H + h) * 128 + d;
        uint4 xv = *reinterpret_cast<const uint4*>(x_bhsd + in_off);
        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
        uint8_t out_b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float fv = bf16_to_f32(xp[k]);
            if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fabsf(fv));
            out_b[k] = f32_to_e4m3_raw(fv * inv_s);
        }
        *reinterpret_cast<uint64_t*>(y_row + out_off) = *reinterpret_cast<uint64_t*>(out_b);
    }
    if constexpr (RECORD_AMAX) {
        block_reduce_atomic_max(lmax, next_amax, slot_id);
    }
}

template <bool RECORD_AMAX>
__global__ void quant_bhsd_to_bshd_dual_64x64_prmt_k(
    const __nv_bfloat16* __restrict__ x_bhsd,
    uint8_t* __restrict__ y_row,
    uint8_t* __restrict__ y_col,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int B,
    int H,
    int S
) {
    __shared__ uint32_t smem_u32[64][17];
    __shared__ float smax[8];

    const int M = B * S;
    const int N = H * 128;
    const float inv_s = inv_base[slot_id];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int n0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;
    const int col = n0 + tx * 4;
    const int row0 = m0 + ty * 4;

    const int h = n0 >> 7;
    const int d_col = (n0 & 127) + tx * 4;
    float lmax = 0.0f;
    uint32_t r_u32[4] = {0u, 0u, 0u, 0u};

    if (col < N) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int row = row0 + k;
            if (row < M) {
                int b = row / S;
                int s = row - b * S;
                int64_t in_off = (((int64_t)b * H + h) * S + s) * 128 + d_col;
                uint2 xv = *reinterpret_cast<const uint2*>(x_bhsd + in_off);
                const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
                float2 f0 = __bfloat1622float2(p2[0]);
                float2 f1 = __bfloat1622float2(p2[1]);
                if constexpr (RECORD_AMAX) {
                    lmax = fmaxf(lmax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
                }
                uint32_t packed = (uint32_t)f32_to_e4m3_raw(f0.x * inv_s)
                                | ((uint32_t)f32_to_e4m3_raw(f0.y * inv_s) << 8)
                                | ((uint32_t)f32_to_e4m3_raw(f1.x * inv_s) << 16)
                                | ((uint32_t)f32_to_e4m3_raw(f1.y * inv_s) << 24);
                r_u32[k] = packed;
                *reinterpret_cast<uint32_t*>(y_row + (int64_t)row * N + col) = packed;
            }
        }
    }

    uint32_t c0, c1, c2, c3;
    transpose_4x4_bytes(r_u32[0], r_u32[1], r_u32[2], r_u32[3], c0, c1, c2, c3);
    smem_u32[tx * 4 + 0][ty] = c0;
    smem_u32[tx * 4 + 1][ty] = c1;
    smem_u32[tx * 4 + 2][ty] = c2;
    smem_u32[tx * 4 + 3][ty] = c3;
    __syncthreads();

    const int out_m = m0 + tx * 4;
    const int out_n0 = n0 + ty * 4;
    if (out_m < M) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int out_n = out_n0 + k;
            if (out_n < N) {
                *reinterpret_cast<uint32_t*>(y_col + (int64_t)out_n * M + out_m) = smem_u32[ty * 4 + k][tx];
            }
        }
    }

    if constexpr (RECORD_AMAX) {
        lmax = warp_reduce_max(lmax);
        int wid = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        if (lane == 0) smax[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? smax[lane] : 0.0f;
            v = warp_reduce_max(v);
            if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
        }
    }
}

// ============================================================================
// 3. SwiGLU, dSwiGLU, GeLU_tanh+Bias, & dGeLU (1D + Single-Pass PRMT_64x64 Dual)
// ============================================================================
__global__ void swiglu_amax_calib_vec8_k(
    const __nv_bfloat16* __restrict__ gu,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    const int n8 = N >> 3;
    const int64_t total_vec8 = (int64_t)M * n8;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float lmax = 0.0f;

    for (int64_t idx8 = tid; idx8 < total_vec8; idx8 += stride) {
        int row = (int)(idx8 / n8);
        int col = (int)(idx8 - (int64_t)row * n8) << 3;
        int64_t base = (int64_t)row * (2 * N);
        uint4 gv = *reinterpret_cast<const uint4*>(gu + base + col);
        uint4 uv = *reinterpret_cast<const uint4*>(gu + base + N + col);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            lmax = fmaxf(lmax, fabsf(silu_f32(bf16_to_f32(gp[k])) * bf16_to_f32(up[k])));
        }
    }
    block_reduce_atomic_max(lmax, next_amax, slot_id);
}

template <bool RECORD_AMAX>
__global__ void swiglu_1d_vec8_k(
    const __nv_bfloat16* __restrict__ gu,
    uint8_t* __restrict__ h_row,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    const float inv_s = inv_base[slot_id];
    const int n8 = N >> 3;
    const int64_t total_vec8 = (int64_t)M * n8;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float lmax = 0.0f;

    for (int64_t idx8 = tid; idx8 < total_vec8; idx8 += stride) {
        int row = (int)(idx8 / n8);
        int col = (int)(idx8 - (int64_t)row * n8) << 3;
        int64_t base = (int64_t)row * (2 * N);
        uint4 gv = *reinterpret_cast<const uint4*>(gu + base + col);
        uint4 uv = *reinterpret_cast<const uint4*>(gu + base + N + col);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
        uint8_t out_b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = silu_f32(bf16_to_f32(gp[k])) * bf16_to_f32(up[k]);
            if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fabsf(val));
            out_b[k] = f32_to_e4m3_raw(val * inv_s);
        }
        *reinterpret_cast<uint64_t*>(h_row + (int64_t)row * N + col) = *reinterpret_cast<uint64_t*>(out_b);
    }
    if constexpr (RECORD_AMAX) {
        block_reduce_atomic_max(lmax, next_amax, slot_id);
    }
}

template <bool RECORD_AMAX>
__global__ void swiglu_dual_64x64_prmt_k(
    const __nv_bfloat16* __restrict__ gu,
    uint8_t* __restrict__ h_row,
    uint8_t* __restrict__ h_col,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    __shared__ uint32_t smem_u32[64][17];
    __shared__ float smax[8];

    const float inv_s = inv_base[slot_id];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int n0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;
    const int col = n0 + tx * 4;
    const int row0 = m0 + ty * 4;
    float lmax = 0.0f;

    uint32_t r_u32[4] = {0u, 0u, 0u, 0u};
    if (col < N) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int row = row0 + k;
            if (row < M) {
                int64_t base = (int64_t)row * (2 * N);
                uint2 gv = *reinterpret_cast<const uint2*>(gu + base + col);
                uint2 uv = *reinterpret_cast<const uint2*>(gu + base + N + col);
                const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
                const __nv_bfloat162* up = reinterpret_cast<const __nv_bfloat162*>(&uv);
                float2 g0 = __bfloat1622float2(gp[0]), g1 = __bfloat1622float2(gp[1]);
                float2 u0 = __bfloat1622float2(up[0]), u1 = __bfloat1622float2(up[1]);
                float v0 = silu_f32(g0.x) * u0.x;
                float v1 = silu_f32(g0.y) * u0.y;
                float v2 = silu_f32(g1.x) * u1.x;
                float v3 = silu_f32(g1.y) * u1.y;
                if constexpr (RECORD_AMAX) {
                    lmax = fmaxf(lmax, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
                }
                uint32_t packed = (uint32_t)f32_to_e4m3_raw(v0 * inv_s)
                                | ((uint32_t)f32_to_e4m3_raw(v1 * inv_s) << 8)
                                | ((uint32_t)f32_to_e4m3_raw(v2 * inv_s) << 16)
                                | ((uint32_t)f32_to_e4m3_raw(v3 * inv_s) << 24);
                r_u32[k] = packed;
                if (h_row != nullptr) {
                    *reinterpret_cast<uint32_t*>(h_row + (int64_t)row * N + col) = packed;
                }
            }
        }
    }

    uint32_t c0, c1, c2, c3;
    transpose_4x4_bytes(r_u32[0], r_u32[1], r_u32[2], r_u32[3], c0, c1, c2, c3);
    smem_u32[tx * 4 + 0][ty] = c0;
    smem_u32[tx * 4 + 1][ty] = c1;
    smem_u32[tx * 4 + 2][ty] = c2;
    smem_u32[tx * 4 + 3][ty] = c3;
    __syncthreads();

    const int out_m = m0 + tx * 4;
    const int out_n0 = n0 + ty * 4;
    if (out_m < M) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int out_n = out_n0 + k;
            if (out_n < N) {
                *reinterpret_cast<uint32_t*>(h_col + (int64_t)out_n * M + out_m) = smem_u32[ty * 4 + k][tx];
            }
        }
    }

    if constexpr (RECORD_AMAX) {
        lmax = warp_reduce_max(lmax);
        int wid = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        if (lane == 0) smax[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? smax[lane] : 0.0f;
            v = warp_reduce_max(v);
            if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
        }
    }
}

__global__ void dswiglu_amax_calib_vec8_k(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ gu,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    const int n8 = N >> 3;
    const int64_t total_vec8 = (int64_t)M * n8;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float lmax = 0.0f;

    for (int64_t idx8 = tid; idx8 < total_vec8; idx8 += stride) {
        int row = (int)(idx8 / n8);
        int col = (int)(idx8 - (int64_t)row * n8) << 3;
        uint4 dhv = *reinterpret_cast<const uint4*>(dh + (int64_t)row * N + col);
        int64_t base = (int64_t)row * (2 * N);
        uint4 gv = *reinterpret_cast<const uint4*>(gu + base + col);
        uint4 uv = *reinterpret_cast<const uint4*>(gu + base + N + col);
        const __nv_bfloat16* dp = reinterpret_cast<const __nv_bfloat16*>(&dhv);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float d = bf16_to_f32(dp[k]);
            float g = bf16_to_f32(gp[k]);
            float u = bf16_to_f32(up[k]);
            float sig = 1.0f / (1.0f + expf(-g));
            float du = d * (g * sig);
            float dg = d * u * sig * (1.0f + g * (1.0f - sig));
            lmax = fmaxf(lmax, fmaxf(fabsf(dg), fabsf(du)));
        }
    }
    block_reduce_atomic_max(lmax, next_amax, slot_id);
}

template <bool RECORD_AMAX>
__global__ void dswiglu_1d_vec8_k(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ gu,
    uint8_t* __restrict__ dgu_row,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    const float inv_s = inv_base[slot_id];
    const int n8 = N >> 3;
    const int64_t total_vec8 = (int64_t)M * n8;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float lmax = 0.0f;

    for (int64_t idx8 = tid; idx8 < total_vec8; idx8 += stride) {
        int row = (int)(idx8 / n8);
        int col = (int)(idx8 - (int64_t)row * n8) << 3;
        uint4 dhv = *reinterpret_cast<const uint4*>(dh + (int64_t)row * N + col);
        int64_t base = (int64_t)row * (2 * N);
        uint4 gv = *reinterpret_cast<const uint4*>(gu + base + col);
        uint4 uv = *reinterpret_cast<const uint4*>(gu + base + N + col);
        const __nv_bfloat16* dp = reinterpret_cast<const __nv_bfloat16*>(&dhv);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
        const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
        uint8_t qg_b[8], qu_b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float d = bf16_to_f32(dp[k]);
            float g = bf16_to_f32(gp[k]);
            float u = bf16_to_f32(up[k]);
            float sig = 1.0f / (1.0f + expf(-g));
            float du = d * (g * sig);
            float dg = d * u * sig * (1.0f + g * (1.0f - sig));
            if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fmaxf(fabsf(dg), fabsf(du)));
            qg_b[k] = f32_to_e5m2_raw(dg * inv_s);
            qu_b[k] = f32_to_e5m2_raw(du * inv_s);
        }
        *reinterpret_cast<uint64_t*>(dgu_row + base + col) = *reinterpret_cast<uint64_t*>(qg_b);
        *reinterpret_cast<uint64_t*>(dgu_row + base + N + col) = *reinterpret_cast<uint64_t*>(qu_b);
    }
    if constexpr (RECORD_AMAX) {
        block_reduce_atomic_max(lmax, next_amax, slot_id);
    }
}

template <bool RECORD_AMAX>
__global__ void dswiglu_dual_64x64_prmt_k(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ gu,
    uint8_t* __restrict__ dgu_row,
    uint8_t* __restrict__ dgu_t,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    __shared__ uint32_t smem_g[64][17];
    __shared__ uint32_t smem_u[64][17];
    __shared__ float smax[8];

    const float inv_s = inv_base[slot_id];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int n0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;
    const int col = n0 + tx * 4;
    const int row0 = m0 + ty * 4;
    float lmax = 0.0f;

    uint32_t rg[4] = {0u, 0u, 0u, 0u};
    uint32_t ru[4] = {0u, 0u, 0u, 0u};
    if (col < N) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int row = row0 + k;
            if (row < M) {
                uint2 dhv = *reinterpret_cast<const uint2*>(dh + (int64_t)row * N + col);
                int64_t base = (int64_t)row * (2 * N);
                uint2 gv = *reinterpret_cast<const uint2*>(gu + base + col);
                uint2 uv = *reinterpret_cast<const uint2*>(gu + base + N + col);
                const __nv_bfloat16* dp = reinterpret_cast<const __nv_bfloat16*>(&dhv);
                const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
                const __nv_bfloat16* up = reinterpret_cast<const __nv_bfloat16*>(&uv);
                uint8_t bg[4], bu[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float d = bf16_to_f32(dp[j]);
                    float g = bf16_to_f32(gp[j]);
                    float u = bf16_to_f32(up[j]);
                    float sig = 1.0f / (1.0f + expf(-g));
                    float du = d * (g * sig);
                    float dg = d * u * sig * (1.0f + g * (1.0f - sig));
                    if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fmaxf(fabsf(dg), fabsf(du)));
                    bg[j] = f32_to_e5m2_raw(dg * inv_s);
                    bu[j] = f32_to_e5m2_raw(du * inv_s);
                }
                uint32_t pg = *reinterpret_cast<uint32_t*>(bg);
                uint32_t pu = *reinterpret_cast<uint32_t*>(bu);
                rg[k] = pg;
                ru[k] = pu;
                *reinterpret_cast<uint32_t*>(dgu_row + base + col) = pg;
                *reinterpret_cast<uint32_t*>(dgu_row + base + N + col) = pu;
            }
        }
    }

    uint32_t cg0, cg1, cg2, cg3, cu0, cu1, cu2, cu3;
    transpose_4x4_bytes(rg[0], rg[1], rg[2], rg[3], cg0, cg1, cg2, cg3);
    transpose_4x4_bytes(ru[0], ru[1], ru[2], ru[3], cu0, cu1, cu2, cu3);
    smem_g[tx * 4 + 0][ty] = cg0; smem_g[tx * 4 + 1][ty] = cg1;
    smem_g[tx * 4 + 2][ty] = cg2; smem_g[tx * 4 + 3][ty] = cg3;
    smem_u[tx * 4 + 0][ty] = cu0; smem_u[tx * 4 + 1][ty] = cu1;
    smem_u[tx * 4 + 2][ty] = cu2; smem_u[tx * 4 + 3][ty] = cu3;
    __syncthreads();

    const int out_m = m0 + tx * 4;
    const int out_n0 = n0 + ty * 4;
    if (out_m < M) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int out_n = out_n0 + k;
            if (out_n < N) {
                *reinterpret_cast<uint32_t*>(dgu_t + (int64_t)out_n * M + out_m) = smem_g[ty * 4 + k][tx];
                *reinterpret_cast<uint32_t*>(dgu_t + (int64_t)(N + out_n) * M + out_m) = smem_u[ty * 4 + k][tx];
            }
        }
    }

    if constexpr (RECORD_AMAX) {
        lmax = warp_reduce_max(lmax);
        int wid = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        if (lane == 0) smax[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? smax[lane] : 0.0f;
            v = warp_reduce_max(v);
            if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
        }
    }
}

// --- GeLU_tanh + Bias (MMDiT 31 MLPs) ---
__global__ void gelu_bias_amax_calib_vec8_k(
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ b,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    const int n8 = N >> 3;
    const int64_t total_vec8 = (int64_t)M * n8;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float lmax = 0.0f;

    for (int64_t idx8 = tid; idx8 < total_vec8; idx8 += stride) {
        int row = (int)(idx8 / n8);
        int col = (int)(idx8 - (int64_t)row * n8) << 3;
        uint4 zv = *reinterpret_cast<const uint4*>(z + (int64_t)row * N + col);
        uint4 bv = *reinterpret_cast<const uint4*>(b + col);
        const __nv_bfloat16* zp = reinterpret_cast<const __nv_bfloat16*>(&zv);
        const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bv);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float val = gelu_tanh_f32(bf16_to_f32(zp[k]) + bf16_to_f32(bp[k]));
            lmax = fmaxf(lmax, fabsf(val));
        }
    }
    block_reduce_atomic_max(lmax, next_amax, slot_id);
}

template <bool SAVE_ZB, bool RECORD_AMAX>
__global__ void gelu_bias_1d_vec8_k(
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ zb_out,
    uint8_t* __restrict__ h_row,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    const float inv_s = inv_base[slot_id];
    const int n8 = N >> 3;
    const int64_t total_vec8 = (int64_t)M * n8;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float lmax = 0.0f;

    for (int64_t idx8 = tid; idx8 < total_vec8; idx8 += stride) {
        int row = (int)(idx8 / n8);
        int col = (int)(idx8 - (int64_t)row * n8) << 3;
        int64_t off = (int64_t)row * N + col;
        uint4 zv = *reinterpret_cast<const uint4*>(z + off);
        uint4 bv = *reinterpret_cast<const uint4*>(b + col);
        const __nv_bfloat16* zp = reinterpret_cast<const __nv_bfloat16*>(&zv);
        const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bv);
        __nv_bfloat16 zb_b[8];
        uint8_t out_b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float zf = bf16_to_f32(zp[k]) + bf16_to_f32(bp[k]);
            if constexpr (SAVE_ZB) zb_b[k] = f32_to_bf16(zf);
            float val = gelu_tanh_f32(zf);
            if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fabsf(val));
            out_b[k] = f32_to_e4m3_raw(val * inv_s);
        }
        if constexpr (SAVE_ZB) {
            *reinterpret_cast<uint4*>(zb_out + off) = *reinterpret_cast<uint4*>(zb_b);
        }
        *reinterpret_cast<uint64_t*>(h_row + off) = *reinterpret_cast<uint64_t*>(out_b);
    }
    if constexpr (RECORD_AMAX) {
        block_reduce_atomic_max(lmax, next_amax, slot_id);
    }
}

template <bool RECORD_AMAX>
__global__ void gelu_bias_dual_64x64_prmt_k(
    const __nv_bfloat16* __restrict__ z,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ zb_out,
    uint8_t* __restrict__ h_row,
    uint8_t* __restrict__ h_col,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    __shared__ uint32_t smem_u32[64][17];
    __shared__ float smax[8];

    const float inv_s = inv_base[slot_id];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int n0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;
    const int col = n0 + tx * 4;
    const int row0 = m0 + ty * 4;
    float lmax = 0.0f;

    uint32_t r_u32[4] = {0u, 0u, 0u, 0u};
    if (col < N) {
        uint2 bv = *reinterpret_cast<const uint2*>(b + col);
        const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bv);
        float b0 = bf16_to_f32(bp[0]), b1 = bf16_to_f32(bp[1]), b2 = bf16_to_f32(bp[2]), b3 = bf16_to_f32(bp[3]);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int row = row0 + k;
            if (row < M) {
                int64_t off = (int64_t)row * N + col;
                uint2 zv = *reinterpret_cast<const uint2*>(z + off);
                const __nv_bfloat16* zp = reinterpret_cast<const __nv_bfloat16*>(&zv);
                float z0 = bf16_to_f32(zp[0]) + b0;
                float z1 = bf16_to_f32(zp[1]) + b1;
                float z2 = bf16_to_f32(zp[2]) + b2;
                float z3 = bf16_to_f32(zp[3]) + b3;
                __nv_bfloat16 zb_b[4] = {f32_to_bf16(z0), f32_to_bf16(z1), f32_to_bf16(z2), f32_to_bf16(z3)};
                *reinterpret_cast<uint2*>(zb_out + off) = *reinterpret_cast<uint2*>(zb_b);
                float v0 = gelu_tanh_f32(z0), v1 = gelu_tanh_f32(z1), v2 = gelu_tanh_f32(z2), v3 = gelu_tanh_f32(z3);
                if constexpr (RECORD_AMAX) {
                    lmax = fmaxf(lmax, fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))));
                }
                uint32_t packed = (uint32_t)f32_to_e4m3_raw(v0 * inv_s)
                                | ((uint32_t)f32_to_e4m3_raw(v1 * inv_s) << 8)
                                | ((uint32_t)f32_to_e4m3_raw(v2 * inv_s) << 16)
                                | ((uint32_t)f32_to_e4m3_raw(v3 * inv_s) << 24);
                r_u32[k] = packed;
                *reinterpret_cast<uint32_t*>(h_row + off) = packed;
            }
        }
    }

    uint32_t c0, c1, c2, c3;
    transpose_4x4_bytes(r_u32[0], r_u32[1], r_u32[2], r_u32[3], c0, c1, c2, c3);
    smem_u32[tx * 4 + 0][ty] = c0;
    smem_u32[tx * 4 + 1][ty] = c1;
    smem_u32[tx * 4 + 2][ty] = c2;
    smem_u32[tx * 4 + 3][ty] = c3;
    __syncthreads();

    const int out_m = m0 + tx * 4;
    const int out_n0 = n0 + ty * 4;
    if (out_m < M) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int out_n = out_n0 + k;
            if (out_n < N) {
                *reinterpret_cast<uint32_t*>(h_col + (int64_t)out_n * M + out_m) = smem_u32[ty * 4 + k][tx];
            }
        }
    }

    if constexpr (RECORD_AMAX) {
        lmax = warp_reduce_max(lmax);
        int wid = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        if (lane == 0) smax[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? smax[lane] : 0.0f;
            v = warp_reduce_max(v);
            if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
        }
    }
}

__global__ void dgelu_amax_calib_vec8_k(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ zb,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    const int n8 = N >> 3;
    const int64_t total_vec8 = (int64_t)M * n8;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float lmax = 0.0f;

    for (int64_t idx8 = tid; idx8 < total_vec8; idx8 += stride) {
        int row = (int)(idx8 / n8);
        int col = (int)(idx8 - (int64_t)row * n8) << 3;
        int64_t off = (int64_t)row * N + col;
        uint4 dhv = *reinterpret_cast<const uint4*>(dh + off);
        uint4 zbv = *reinterpret_cast<const uint4*>(zb + off);
        const __nv_bfloat16* dp = reinterpret_cast<const __nv_bfloat16*>(&dhv);
        const __nv_bfloat16* zp = reinterpret_cast<const __nv_bfloat16*>(&zbv);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float dz = dgelu_tanh_f32(bf16_to_f32(dp[k]), bf16_to_f32(zp[k]));
            lmax = fmaxf(lmax, fabsf(dz));
        }
    }
    block_reduce_atomic_max(lmax, next_amax, slot_id);
}

template <bool RECORD_AMAX>
__global__ void dgelu_1d_vec8_k(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ zb,
    uint8_t* __restrict__ dz_row,
    float* __restrict__ db1_f32,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    const float inv_s = inv_base[slot_id];
    const int n8 = N >> 3;
    const int c8 = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = c8 << 3;
    float bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float lmax = 0.0f;

    if (c8 < n8) {
        for (int row = blockIdx.y; row < M; row += gridDim.y) {
            int64_t off = (int64_t)row * N + col;
            uint4 dhv = *reinterpret_cast<const uint4*>(dh + off);
            uint4 zbv = *reinterpret_cast<const uint4*>(zb + off);
            const __nv_bfloat16* dp = reinterpret_cast<const __nv_bfloat16*>(&dhv);
            const __nv_bfloat16* zp = reinterpret_cast<const __nv_bfloat16*>(&zbv);
            uint8_t out_b[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dz = dgelu_tanh_f32(bf16_to_f32(dp[k]), bf16_to_f32(zp[k]));
                bsum[k] += dz;
                if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fabsf(dz));
                out_b[k] = f32_to_e5m2_raw(dz * inv_s);
            }
            *reinterpret_cast<uint64_t*>(dz_row + off) = *reinterpret_cast<uint64_t*>(out_b);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            atomicAdd(db1_f32 + col + k, bsum[k]);
        }
    }
    if constexpr (RECORD_AMAX) {
        block_reduce_atomic_max(lmax, next_amax, slot_id);
    }
}

template <bool RECORD_AMAX>
__global__ void dgelu_dual_64x64_prmt_k(
    const __nv_bfloat16* __restrict__ dh,
    const __nv_bfloat16* __restrict__ zb,
    uint8_t* __restrict__ dz_row,
    uint8_t* __restrict__ dz_col,
    float* __restrict__ db1_f32,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int N
) {
    __shared__ uint32_t smem_u32[64][17];
    __shared__ float smax[8];

    const float inv_s = inv_base[slot_id];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int n0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;
    const int col = n0 + tx * 4;
    const int row0 = m0 + ty * 4;

    float lmax = 0.0f;
    float bacc0 = 0.0f, bacc1 = 0.0f, bacc2 = 0.0f, bacc3 = 0.0f;
    uint32_t r_u32[4] = {0u, 0u, 0u, 0u};

    if (col < N) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int row = row0 + k;
            if (row < M) {
                int64_t off = (int64_t)row * N + col;
                uint2 dhv = *reinterpret_cast<const uint2*>(dh + off);
                uint2 zbv = *reinterpret_cast<const uint2*>(zb + off);
                const __nv_bfloat16* dp = reinterpret_cast<const __nv_bfloat16*>(&dhv);
                const __nv_bfloat16* zp = reinterpret_cast<const __nv_bfloat16*>(&zbv);
                float dz0 = dgelu_tanh_f32(bf16_to_f32(dp[0]), bf16_to_f32(zp[0]));
                float dz1 = dgelu_tanh_f32(bf16_to_f32(dp[1]), bf16_to_f32(zp[1]));
                float dz2 = dgelu_tanh_f32(bf16_to_f32(dp[2]), bf16_to_f32(zp[2]));
                float dz3 = dgelu_tanh_f32(bf16_to_f32(dp[3]), bf16_to_f32(zp[3]));
                bacc0 += dz0; bacc1 += dz1; bacc2 += dz2; bacc3 += dz3;
                if constexpr (RECORD_AMAX) {
                    lmax = fmaxf(lmax, fmaxf(fmaxf(fabsf(dz0), fabsf(dz1)), fmaxf(fabsf(dz2), fabsf(dz3))));
                }
                uint32_t packed = (uint32_t)f32_to_e5m2_raw(dz0 * inv_s)
                                | ((uint32_t)f32_to_e5m2_raw(dz1 * inv_s) << 8)
                                | ((uint32_t)f32_to_e5m2_raw(dz2 * inv_s) << 16)
                                | ((uint32_t)f32_to_e5m2_raw(dz3 * inv_s) << 24);
                r_u32[k] = packed;
                *reinterpret_cast<uint32_t*>(dz_row + off) = packed;
            }
        }
    }

    uint32_t c0, c1, c2, c3;
    transpose_4x4_bytes(r_u32[0], r_u32[1], r_u32[2], r_u32[3], c0, c1, c2, c3);
    smem_u32[tx * 4 + 0][ty] = c0;
    smem_u32[tx * 4 + 1][ty] = c1;
    smem_u32[tx * 4 + 2][ty] = c2;
    smem_u32[tx * 4 + 3][ty] = c3;
    __syncthreads();

    const int out_m = m0 + tx * 4;
    const int out_n0 = n0 + ty * 4;
    if (out_m < M) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int out_n = out_n0 + k;
            if (out_n < N) {
                *reinterpret_cast<uint32_t*>(dz_col + (int64_t)out_n * M + out_m) = smem_u32[ty * 4 + k][tx];
            }
        }
    }

    __syncthreads();
    float* sb = reinterpret_cast<float*>(&smem_u32[0][0]);
    sb[ty * 64 + tx * 4 + 0] = bacc0;
    sb[ty * 64 + tx * 4 + 1] = bacc1;
    sb[ty * 64 + tx * 4 + 2] = bacc2;
    sb[ty * 64 + tx * 4 + 3] = bacc3;
    __syncthreads();
    if (threadIdx.x < 64) {
        int c_idx = threadIdx.x;
        if (n0 + c_idx < N) {
            float s = 0.0f;
            #pragma unroll
            for (int r = 0; r < 16; ++r) s += sb[r * 64 + c_idx];
            atomicAdd(db1_f32 + n0 + c_idx, s);
        }
    }

    if constexpr (RECORD_AMAX) {
        lmax = warp_reduce_max(lmax);
        int wid = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        if (lane == 0) smax[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? smax[lane] : 0.0f;
            v = warp_reduce_max(v);
            if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
        }
    }
}

// ============================================================================
// 4. Single-Pass Register-Resident Vision LayerNorm(1024) Fwd & Persistent Bwd
// ============================================================================
template <bool SAVE_STATS>
__global__ void ln_fwd_1024_k(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int M,
    float eps
) {
    constexpr int D = 1024;
    int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;  // 0..127 (4 warps, 1 uint4 = 8 BF16 per thread!)
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int col = tid << 3;
    const int64_t off = (int64_t)row * D + col;

    uint4 xv = *reinterpret_cast<const uint4*>(x + off);
    uint4 gv = *reinterpret_cast<const uint4*>(gamma + col);
    uint4 bv = *reinterpret_cast<const uint4*>(beta + col);
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
    const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bv);

    float xf[8];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        xf[k] = bf16_to_f32(xp[k]);
        s += xf[k];
    }

    s = warp_reduce_sum(s);
    __shared__ float smem[4];
    if (lane == 0) smem[wid] = s;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 4) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) smem[0] = v * (1.0f / (float)D);
    }
    __syncthreads();
    const float mean = smem[0];

    float var_s = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        xf[k] -= mean;
        var_s += xf[k] * xf[k];
    }
    var_s = warp_reduce_sum(var_s);
    if (lane == 0) smem[wid] = var_s;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 4) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            float r = rsqrtf(v * (1.0f / (float)D) + eps);
            smem[0] = r;
            if constexpr (SAVE_STATS) {
                mean_out[row] = mean;
                rstd_out[row] = r;
            }
        }
    }
    __syncthreads();
    const float rstd = smem[0];

    __nv_bfloat16 out_b[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        out_b[k] = f32_to_bf16(xf[k] * rstd * bf16_to_f32(gp[k]) + bf16_to_f32(bp[k]));
    }
    *reinterpret_cast<uint4*>(y_out + off) = *reinterpret_cast<uint4*>(out_b);
}

template <bool HAS_DRES, bool HAS_DRES_SUM>
__global__ void ln_bwd_1024_k(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ d_res_in,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__ mean_p,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ dx,
    float* __restrict__ dgamma_f32,
    float* __restrict__ dbeta_f32,
    float* __restrict__ dres_sum_f32,
    int M
) {
    constexpr int D = 1024;
    const int tid = threadIdx.x;  // 0..127
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int col = tid << 3;

    uint4 gv = *reinterpret_cast<const uint4*>(gamma + col);
    const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
    float gf[8];
    float dg_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float db_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float dr_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (int k = 0; k < 8; ++k) gf[k] = bf16_to_f32(gp[k]);

    __shared__ float2 s2[4];

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const int64_t off = (int64_t)row * D + col;
        const float mean = mean_p[row];
        const float rstd = rstd_p[row];

        uint4 xv = *reinterpret_cast<const uint4*>(x + off);
        uint4 dyv = *reinterpret_cast<const uint4*>(dy + off);
        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
        const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dyv);

        float xh[8], g[8];
        float s_g = 0.0f, s_gxh = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float dyk = bf16_to_f32(dyp[k]);
            float xhk = (bf16_to_f32(xp[k]) - mean) * rstd;
            xh[k] = xhk;
            dg_acc[k] += dyk * xhk;
            db_acc[k] += dyk;
            float gk = dyk * gf[k];
            g[k] = gk;
            s_g += gk;
            s_gxh += gk * xhk;
        }

        s_g = warp_reduce_sum(s_g);
        s_gxh = warp_reduce_sum(s_gxh);
        if (lane == 0) s2[wid] = make_float2(s_g, s_gxh);
        __syncthreads();
        if (wid == 0) {
            float vg = (lane < 4) ? s2[lane].x : 0.0f;
            float vgxh = (lane < 4) ? s2[lane].y : 0.0f;
            vg = warp_reduce_sum(vg);
            vgxh = warp_reduce_sum(vgxh);
            if (lane == 0) {
                s2[0] = make_float2(vg * (1.0f / (float)D), vgxh * (1.0f / (float)D));
            }
        }
        __syncthreads();
        const float c1 = s2[0].x;
        const float c2 = s2[0].y;

        uint4 drv;
        if constexpr (HAS_DRES) {
            drv = *reinterpret_cast<const uint4*>(d_res_in + off);
        }
        const __nv_bfloat16* drp = reinterpret_cast<const __nv_bfloat16*>(&drv);

        __nv_bfloat16 out_b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float dxf = rstd * (g[k] - c1 - xh[k] * c2);
            if constexpr (HAS_DRES) {
                float drf = bf16_to_f32(drp[k]);
                if constexpr (HAS_DRES_SUM) dr_acc[k] += drf;
                float drn = bf16_to_f32(f32_to_bf16(dxf));
                dxf = drf + drn;
            }
            out_b[k] = f32_to_bf16(dxf);
        }
        *reinterpret_cast<uint4*>(dx + off) = *reinterpret_cast<uint4*>(out_b);
        __syncthreads();
    }

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        atomicAdd(dgamma_f32 + col + k, dg_acc[k]);
        atomicAdd(dbeta_f32 + col + k, db_acc[k]);
        if constexpr (HAS_DRES && HAS_DRES_SUM) {
            atomicAdd(dres_sum_f32 + col + k, dr_acc[k]);
        }
    }
}

// ============================================================================
// 5. v8 Zero-Copy bhsd/bshd QKV+Norm+RoPE, Fused E5M2 Bwd Quant, 100%-Occ RMSNorm2560, & Vision Persistent RoPE
// ============================================================================
__device__ __forceinline__ float subwarp8_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

template <bool BHSD>
__global__ void qwen_qkv_norm_rope_fwd_vec8_k(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const __nv_bfloat16* __restrict__ cos_p,
    const __nv_bfloat16* __restrict__ sin_p,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    float* __restrict__ rstd_out,
    int B,
    int S,
    int h_q,
    int h_kv,
    float eps
) {
    constexpr int TOK_PER_SW = 4;
    const int T = B * S;
    const int h_qk = h_q + h_kv;
    const int h_tot = h_q + 2 * h_kv;
    const int t_groups = (T + TOK_PER_SW - 1) / TOK_PER_SW;
    const int sw_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 3;
    if (sw_global >= t_groups * h_tot) return;

    const int sub = threadIdx.x & 7;
    const int tg = sw_global / h_tot;
    const int h = sw_global - tg * h_tot;
    const int t0 = tg * TOK_PER_SW;

    const int d0 = sub * 8;
    const int d1 = 64 + d0;

    if (h >= h_qk) {
        int vh = h - h_qk;
        #pragma unroll
        for (int k = 0; k < TOK_PER_SW; ++k) {
            int t = t0 + k;
            if (t < T) {
                int64_t base_in = ((int64_t)t * h_tot + h) * 128;
                int64_t base_v;
                if constexpr (BHSD) {
                    int b = t / S;
                    int s = t - b * S;
                    base_v = (((int64_t)b * h_kv + vh) * S + s) * 128;
                } else {
                    base_v = ((int64_t)t * h_kv + vh) * 128;
                }
                *reinterpret_cast<uint4*>(v_out + base_v + d0) = *reinterpret_cast<const uint4*>(qkv + base_in + d0);
                *reinterpret_cast<uint4*>(v_out + base_v + d1) = *reinterpret_cast<const uint4*>(qkv + base_in + d1);
            }
        }
        return;
    }

    const __nv_bfloat16* w_ptr = (h < h_q) ? qw : kw;
    uint4 w0_v = *reinterpret_cast<const uint4*>(w_ptr + d0);
    uint4 w1_v = *reinterpret_cast<const uint4*>(w_ptr + d1);
    const __nv_bfloat16* w0_p = reinterpret_cast<const __nv_bfloat16*>(&w0_v);
    const __nv_bfloat16* w1_p = reinterpret_cast<const __nv_bfloat16*>(&w1_v);
    float wf0[8], wf1[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        wf0[j] = bf16_to_f32(w0_p[j]);
        wf1[j] = bf16_to_f32(w1_p[j]);
    }

    #pragma unroll
    for (int k = 0; k < TOK_PER_SW; ++k) {
        int t = t0 + k;
        if (t < T) {
            int64_t base_in = ((int64_t)t * h_tot + h) * 128;
            uint4 x0_v = *reinterpret_cast<const uint4*>(qkv + base_in + d0);
            uint4 x1_v = *reinterpret_cast<const uint4*>(qkv + base_in + d1);
            const __nv_bfloat16* x0_p = reinterpret_cast<const __nv_bfloat16*>(&x0_v);
            const __nv_bfloat16* x1_p = reinterpret_cast<const __nv_bfloat16*>(&x1_v);

            float xf0[8], xf1[8];
            float sq = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                xf0[j] = bf16_to_f32(x0_p[j]);
                xf1[j] = bf16_to_f32(x1_p[j]);
                sq += xf0[j] * xf0[j] + xf1[j] * xf1[j];
            }
            sq = subwarp8_reduce_sum(sq);
            float rstd = rsqrtf(sq * (1.0f / 128.0f) + eps);
            if (sub == 0 && rstd_out != nullptr) {
                rstd_out[(int64_t)t * h_qk + h] = rstd;
            }

            int64_t base_cs = (int64_t)t * 128;
            uint4 c0_v = *reinterpret_cast<const uint4*>(cos_p + base_cs + d0);
            uint4 c1_v = *reinterpret_cast<const uint4*>(cos_p + base_cs + d1);
            uint4 s0_v = *reinterpret_cast<const uint4*>(sin_p + base_cs + d0);
            uint4 s1_v = *reinterpret_cast<const uint4*>(sin_p + base_cs + d1);
            const __nv_bfloat16* c0_p = reinterpret_cast<const __nv_bfloat16*>(&c0_v);
            const __nv_bfloat16* c1_p = reinterpret_cast<const __nv_bfloat16*>(&c1_v);
            const __nv_bfloat16* s0_p = reinterpret_cast<const __nv_bfloat16*>(&s0_v);
            const __nv_bfloat16* s1_p = reinterpret_cast<const __nv_bfloat16*>(&s1_v);

            __nv_bfloat16 y0_b[8], y1_b[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float nx0 = xf0[j] * rstd * wf0[j];
                float nx1 = xf1[j] * rstd * wf1[j];
                y0_b[j] = f32_to_bf16(nx0 * bf16_to_f32(c0_p[j]) - nx1 * bf16_to_f32(s0_p[j]));
                y1_b[j] = f32_to_bf16(nx1 * bf16_to_f32(c1_p[j]) + nx0 * bf16_to_f32(s1_p[j]));
            }

            if (h < h_q) {
                int64_t base_q;
                if constexpr (BHSD) {
                    int b = t / S;
                    int s = t - b * S;
                    base_q = (((int64_t)b * h_q + h) * S + s) * 128;
                } else {
                    base_q = ((int64_t)t * h_q + h) * 128;
                }
                *reinterpret_cast<uint4*>(q_out + base_q + d0) = *reinterpret_cast<uint4*>(y0_b);
                *reinterpret_cast<uint4*>(q_out + base_q + d1) = *reinterpret_cast<uint4*>(y1_b);
            } else {
                int kh = h - h_q;
                int64_t base_k;
                if constexpr (BHSD) {
                    int b = t / S;
                    int s = t - b * S;
                    base_k = (((int64_t)b * h_kv + kh) * S + s) * 128;
                } else {
                    base_k = ((int64_t)t * h_kv + kh) * 128;
                }
                *reinterpret_cast<uint4*>(k_out + base_k + d0) = *reinterpret_cast<uint4*>(y0_b);
                *reinterpret_cast<uint4*>(k_out + base_k + d1) = *reinterpret_cast<uint4*>(y1_b);
            }
        }
    }
}

// Non-persistent, zero-idle-warp Qwen QKV+Norm+RoPE Bwd with CTA smem dqw/dkw reduction
// AND optional direct E5M2 quantization (eliminating 16.6GB BF16 dqkv write + 16.6GB BF16 read!)
template <bool BHSD, bool QUANT_E5>
__global__ void qwen_qkv_norm_rope_bwd_opt_k(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ dk,
    const __nv_bfloat16* __restrict__ dv,
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const __nv_bfloat16* __restrict__ cos_p,
    const __nv_bfloat16* __restrict__ sin_p,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ dqkv_bf16,
    uint8_t* __restrict__ dqkv_e5_row,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_dqkv,
    float* __restrict__ dqw_f32,
    float* __restrict__ dkw_f32,
    int B,
    int S,
    int h_q,
    int h_kv
) {
    constexpr int TOK_PER_SW = 8;
    __shared__ float s_dw[2][128];
    (&s_dw[0][0])[threadIdx.x] = 0.0f;
    __syncthreads();

    const int T = B * S;
    const int h_qk = h_q + h_kv;
    const int h_tot = h_q + 2 * h_kv;
    const int t_groups = (T + TOK_PER_SW - 1) / TOK_PER_SW;
    const int sw_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 3;
    const float inv_s = QUANT_E5 ? inv_base[slot_dqkv] : 1.0f;
    float lmax = 0.0f;

    if (sw_global < t_groups * h_tot) {
        const int sub = threadIdx.x & 7;
        const int tg = sw_global / h_tot;
        const int h = sw_global - tg * h_tot;
        const int t0 = tg * TOK_PER_SW;
        const int d0 = sub * 8;
        const int d1 = 64 + d0;

        if (h >= h_qk) {
            int vh = h - h_qk;
            for (int k = 0; k < TOK_PER_SW; ++k) {
                int t = t0 + k;
                if (t < T) {
                    int64_t base_qkv = ((int64_t)t * h_tot + h) * 128;
                    int64_t base_v;
                    if constexpr (BHSD) {
                        int b = t / S;
                        int s = t - b * S;
                        base_v = (((int64_t)b * h_kv + vh) * S + s) * 128;
                    } else {
                        base_v = ((int64_t)t * h_kv + vh) * 128;
                    }
                    uint4 v0_u4 = *reinterpret_cast<const uint4*>(dv + base_v + d0);
                    uint4 v1_u4 = *reinterpret_cast<const uint4*>(dv + base_v + d1);
                    if constexpr (QUANT_E5) {
                        const __nv_bfloat16* p0 = reinterpret_cast<const __nv_bfloat16*>(&v0_u4);
                        const __nv_bfloat16* p1 = reinterpret_cast<const __nv_bfloat16*>(&v1_u4);
                        uint8_t e0[8], e1[8];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            float f0 = bf16_to_f32(p0[j]);
                            float f1 = bf16_to_f32(p1[j]);
                            lmax = fmaxf(lmax, fmaxf(fabsf(f0), fabsf(f1)));
                            e0[j] = f32_to_e5m2_raw(f0 * inv_s);
                            e1[j] = f32_to_e5m2_raw(f1 * inv_s);
                        }
                        *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d0) = *reinterpret_cast<uint64_t*>(e0);
                        *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d1) = *reinterpret_cast<uint64_t*>(e1);
                    } else {
                        *reinterpret_cast<uint4*>(dqkv_bf16 + base_qkv + d0) = v0_u4;
                        *reinterpret_cast<uint4*>(dqkv_bf16 + base_qkv + d1) = v1_u4;
                    }
                }
            }
        } else {
            const bool is_q = (h < h_q);
            const __nv_bfloat16* w_ptr = is_q ? qw : kw;
            uint4 w0_v = *reinterpret_cast<const uint4*>(w_ptr + d0);
            uint4 w1_v = *reinterpret_cast<const uint4*>(w_ptr + d1);
            const __nv_bfloat16* w0_p = reinterpret_cast<const __nv_bfloat16*>(&w0_v);
            const __nv_bfloat16* w1_p = reinterpret_cast<const __nv_bfloat16*>(&w1_v);
            float wf0[8], wf1[8];
            float dw0_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float dw1_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                wf0[j] = bf16_to_f32(w0_p[j]);
                wf1[j] = bf16_to_f32(w1_p[j]);
            }

            for (int k = 0; k < TOK_PER_SW; ++k) {
                int t = t0 + k;
                if (t < T) {
                    int64_t base_qkv = ((int64_t)t * h_tot + h) * 128;
                    uint4 dy0_v, dy1_v;
                    if (is_q) {
                        int64_t base_q;
                        if constexpr (BHSD) {
                            int b = t / S;
                            int s = t - b * S;
                            base_q = (((int64_t)b * h_q + h) * S + s) * 128;
                        } else {
                            base_q = ((int64_t)t * h_q + h) * 128;
                        }
                        dy0_v = *reinterpret_cast<const uint4*>(dq + base_q + d0);
                        dy1_v = *reinterpret_cast<const uint4*>(dq + base_q + d1);
                    } else {
                        int kh = h - h_q;
                        int64_t base_k;
                        if constexpr (BHSD) {
                            int b = t / S;
                            int s = t - b * S;
                            base_k = (((int64_t)b * h_kv + kh) * S + s) * 128;
                        } else {
                            base_k = ((int64_t)t * h_kv + kh) * 128;
                        }
                        dy0_v = *reinterpret_cast<const uint4*>(dk + base_k + d0);
                        dy1_v = *reinterpret_cast<const uint4*>(dk + base_k + d1);
                    }
                    const __nv_bfloat16* dy0_p = reinterpret_cast<const __nv_bfloat16*>(&dy0_v);
                    const __nv_bfloat16* dy1_p = reinterpret_cast<const __nv_bfloat16*>(&dy1_v);

                    int64_t base_cs = (int64_t)t * 128;
                    uint4 c0_v = *reinterpret_cast<const uint4*>(cos_p + base_cs + d0);
                    uint4 c1_v = *reinterpret_cast<const uint4*>(cos_p + base_cs + d1);
                    uint4 s0_v = *reinterpret_cast<const uint4*>(sin_p + base_cs + d0);
                    uint4 s1_v = *reinterpret_cast<const uint4*>(sin_p + base_cs + d1);
                    const __nv_bfloat16* c0_p = reinterpret_cast<const __nv_bfloat16*>(&c0_v);
                    const __nv_bfloat16* c1_p = reinterpret_cast<const __nv_bfloat16*>(&c1_v);
                    const __nv_bfloat16* s0_p = reinterpret_cast<const __nv_bfloat16*>(&s0_v);
                    const __nv_bfloat16* s1_p = reinterpret_cast<const __nv_bfloat16*>(&s1_v);

                    uint4 x0_v = *reinterpret_cast<const uint4*>(qkv + base_qkv + d0);
                    uint4 x1_v = *reinterpret_cast<const uint4*>(qkv + base_qkv + d1);
                    const __nv_bfloat16* x0_p = reinterpret_cast<const __nv_bfloat16*>(&x0_v);
                    const __nv_bfloat16* x1_p = reinterpret_cast<const __nv_bfloat16*>(&x1_v);
                    float rstd = rstd_p[(int64_t)t * h_qk + h];

                    float xh0[8], xh1[8], g0[8], g1[8];
                    float cdot = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float dy0 = bf16_to_f32(dy0_p[j]);
                        float dy1 = bf16_to_f32(dy1_p[j]);
                        float dnx0 = dy0 * bf16_to_f32(c0_p[j]) + dy1 * bf16_to_f32(s1_p[j]);
                        float dnx1 = dy1 * bf16_to_f32(c1_p[j]) - dy0 * bf16_to_f32(s0_p[j]);
                        xh0[j] = bf16_to_f32(x0_p[j]) * rstd;
                        xh1[j] = bf16_to_f32(x1_p[j]) * rstd;
                        dw0_acc[j] += dnx0 * xh0[j];
                        dw1_acc[j] += dnx1 * xh1[j];
                        g0[j] = dnx0 * wf0[j];
                        g1[j] = dnx1 * wf1[j];
                        cdot += g0[j] * xh0[j] + g1[j] * xh1[j];
                    }
                    cdot = subwarp8_reduce_sum(cdot) * (1.0f / 128.0f);

                    if constexpr (QUANT_E5) {
                        uint8_t e0[8], e1[8];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            float dx0 = rstd * (g0[j] - xh0[j] * cdot);
                            float dx1 = rstd * (g1[j] - xh1[j] * cdot);
                            lmax = fmaxf(lmax, fmaxf(fabsf(dx0), fabsf(dx1)));
                            e0[j] = f32_to_e5m2_raw(dx0 * inv_s);
                            e1[j] = f32_to_e5m2_raw(dx1 * inv_s);
                        }
                        *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d0) = *reinterpret_cast<uint64_t*>(e0);
                        *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d1) = *reinterpret_cast<uint64_t*>(e1);
                    } else {
                        __nv_bfloat16 dx0_b[8], dx1_b[8];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            dx0_b[j] = f32_to_bf16(rstd * (g0[j] - xh0[j] * cdot));
                            dx1_b[j] = f32_to_bf16(rstd * (g1[j] - xh1[j] * cdot));
                        }
                        *reinterpret_cast<uint4*>(dqkv_bf16 + base_qkv + d0) = *reinterpret_cast<uint4*>(dx0_b);
                        *reinterpret_cast<uint4*>(dqkv_bf16 + base_qkv + d1) = *reinterpret_cast<uint4*>(dx1_b);
                    }
                }
            }

            float* s_dst = s_dw[is_q ? 0 : 1];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                atomicAdd(s_dst + d0 + j, dw0_acc[j]);
                atomicAdd(s_dst + d1 + j, dw1_acc[j]);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x < 128) {
        float vq = s_dw[0][threadIdx.x];
        if (vq != 0.0f) atomicAdd(dqw_f32 + threadIdx.x, vq);
    } else {
        int idx = threadIdx.x - 128;
        float vk = s_dw[1][idx];
        if (vk != 0.0f) atomicAdd(dkw_f32 + idx, vk);
    }
    if constexpr (QUANT_E5) {
        block_reduce_atomic_max(lmax, next_amax, slot_dqkv);
    }
}

__global__ void vis_qkv_rope_fwd_vec8_k(
    const __nv_bfloat16* __restrict__ qkv,
    const float* __restrict__ cos_f32,
    const float* __restrict__ sin_f32,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int total_tokens,
    int seq_per_cam,
    int num_heads
) {
    const int items_per_tok = 3 * num_heads;
    const int head_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 2;
    if (head_global >= total_tokens * items_per_tok) return;

    const int sub = threadIdx.x & 3;
    const int tok = head_global / items_per_tok;
    const int rem = head_global - tok * items_per_tok;
    const int qkv_idx = rem / num_heads;
    const int h = rem - qkv_idx * num_heads;

    const int64_t in_off = ((int64_t)tok * items_per_tok + rem) * 64;
    const int64_t out_off = ((int64_t)tok * num_heads + h) * 64;
    const int d0 = sub * 8;
    const int d1 = 32 + d0;

    uint4 x0_v = *reinterpret_cast<const uint4*>(qkv + in_off + d0);
    uint4 x1_v = *reinterpret_cast<const uint4*>(qkv + in_off + d1);

    if (qkv_idx == 2) {
        *reinterpret_cast<uint4*>(v_out + out_off + d0) = x0_v;
        *reinterpret_cast<uint4*>(v_out + out_off + d1) = x1_v;
        return;
    }

    const __nv_bfloat16* x0_p = reinterpret_cast<const __nv_bfloat16*>(&x0_v);
    const __nv_bfloat16* x1_p = reinterpret_cast<const __nv_bfloat16*>(&x1_v);
    const int cs_base = (tok % seq_per_cam) * 64;
    const float4 c0_a = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d0);
    const float4 c0_b = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d0 + 4);
    const float4 c1_a = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d1);
    const float4 c1_b = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d1 + 4);
    const float4 s0_a = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d0);
    const float4 s0_b = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d0 + 4);
    const float4 s1_a = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d1);
    const float4 s1_b = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d1 + 4);

    const float c0[8] = {c0_a.x, c0_a.y, c0_a.z, c0_a.w, c0_b.x, c0_b.y, c0_b.z, c0_b.w};
    const float c1[8] = {c1_a.x, c1_a.y, c1_a.z, c1_a.w, c1_b.x, c1_b.y, c1_b.z, c1_b.w};
    const float s0[8] = {s0_a.x, s0_a.y, s0_a.z, s0_a.w, s0_b.x, s0_b.y, s0_b.z, s0_b.w};
    const float s1[8] = {s1_a.x, s1_a.y, s1_a.z, s1_a.w, s1_b.x, s1_b.y, s1_b.z, s1_b.w};

    __nv_bfloat16 y0_b[8], y1_b[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float x0 = bf16_to_f32(x0_p[k]);
        float x1 = bf16_to_f32(x1_p[k]);
        y0_b[k] = f32_to_bf16(x0 * c0[k] - x1 * s0[k]);
        y1_b[k] = f32_to_bf16(x1 * c1[k] + x0 * s1[k]);
    }
    __nv_bfloat16* dst = (qkv_idx == 0) ? q_out : k_out;
    *reinterpret_cast<uint4*>(dst + out_off + d0) = *reinterpret_cast<uint4*>(y0_b);
    *reinterpret_cast<uint4*>(dst + out_off + d1) = *reinterpret_cast<uint4*>(y1_b);
}

// Vision QKV+RoPE Bwd with TOK_PER_SW=16 register accumulation for fused d_qkv_b (zero globalKernelBgradAB_v2!)
__global__ void vis_qkv_rope_bwd_vec8_k(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ dk,
    const __nv_bfloat16* __restrict__ dv,
    const float* __restrict__ cos_f32,
    const float* __restrict__ sin_f32,
    __nv_bfloat16* __restrict__ dqkv,
    int total_tokens,
    int seq_per_cam,
    int num_heads
) {
    const int items_per_tok = 3 * num_heads;
    const int head_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 2;
    if (head_global >= total_tokens * items_per_tok) return;

    const int sub = threadIdx.x & 3;
    const int tok = head_global / items_per_tok;
    const int rem = head_global - tok * items_per_tok;
    const int qkv_idx = rem / num_heads;
    const int h = rem - qkv_idx * num_heads;

    const int64_t out_off = ((int64_t)tok * items_per_tok + rem) * 64;
    const int64_t in_off = ((int64_t)tok * num_heads + h) * 64;
    const int d0 = sub * 8;
    const int d1 = 32 + d0;

    if (qkv_idx == 2) {
        *reinterpret_cast<uint4*>(dqkv + out_off + d0) = *reinterpret_cast<const uint4*>(dv + in_off + d0);
        *reinterpret_cast<uint4*>(dqkv + out_off + d1) = *reinterpret_cast<const uint4*>(dv + in_off + d1);
        return;
    }

    const __nv_bfloat16* src = (qkv_idx == 0) ? dq : dk;
    uint4 dy0_v = *reinterpret_cast<const uint4*>(src + in_off + d0);
    uint4 dy1_v = *reinterpret_cast<const uint4*>(src + in_off + d1);
    const __nv_bfloat16* dy0_p = reinterpret_cast<const __nv_bfloat16*>(&dy0_v);
    const __nv_bfloat16* dy1_p = reinterpret_cast<const __nv_bfloat16*>(&dy1_v);

    const int cs_base = (tok % seq_per_cam) * 64;
    const float4 c0_a = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d0);
    const float4 c0_b = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d0 + 4);
    const float4 c1_a = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d1);
    const float4 c1_b = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d1 + 4);
    const float4 s0_a = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d0);
    const float4 s0_b = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d0 + 4);
    const float4 s1_a = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d1);
    const float4 s1_b = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d1 + 4);

    const float c0[8] = {c0_a.x, c0_a.y, c0_a.z, c0_a.w, c0_b.x, c0_b.y, c0_b.z, c0_b.w};
    const float c1[8] = {c1_a.x, c1_a.y, c1_a.z, c1_a.w, c1_b.x, c1_b.y, c1_b.z, c1_b.w};
    const float s0[8] = {s0_a.x, s0_a.y, s0_a.z, s0_a.w, s0_b.x, s0_b.y, s0_b.z, s0_b.w};
    const float s1[8] = {s1_a.x, s1_a.y, s1_a.z, s1_a.w, s1_b.x, s1_b.y, s1_b.z, s1_b.w};

    __nv_bfloat16 dx0_b[8], dx1_b[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float dy0 = bf16_to_f32(dy0_p[k]);
        float dy1 = bf16_to_f32(dy1_p[k]);
        dx0_b[k] = f32_to_bf16(dy0 * c0[k] + dy1 * s1[k]);
        dx1_b[k] = f32_to_bf16(dy1 * c1[k] - dy0 * s0[k]);
    }
    *reinterpret_cast<uint4*>(dqkv + out_off + d0) = *reinterpret_cast<uint4*>(dx0_b);
    *reinterpret_cast<uint4*>(dqkv + out_off + d1) = *reinterpret_cast<uint4*>(dx1_b);
}

template <bool HAS_DBIAS>
__global__ void vis_qkv_rope_bwd_opt_k(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ dk,
    const __nv_bfloat16* __restrict__ dv,
    const float* __restrict__ cos_f32,
    const float* __restrict__ sin_f32,
    __nv_bfloat16* __restrict__ dqkv,
    float* __restrict__ dqkv_bias_f32,
    int total_tokens,
    int seq_per_cam,
    int num_heads
) {
    constexpr int TOK_PER_SW = 16;
    const int items_per_tok = 3 * num_heads;
    const int t_groups = (total_tokens + TOK_PER_SW - 1) / TOK_PER_SW;
    const int sw_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 2;
    if (sw_global >= t_groups * items_per_tok) return;

    const int sub = threadIdx.x & 3;
    const int tg = sw_global / items_per_tok;
    const int rem = sw_global - tg * items_per_tok;
    const int qkv_idx = rem / num_heads;
    const int h = rem - qkv_idx * num_heads;
    const int tok0 = tg * TOK_PER_SW;
    const int d0 = sub * 8;
    const int d1 = 32 + d0;

    float db0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float db1[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    if (qkv_idx == 2) {
        for (int s = 0; s < TOK_PER_SW; ++s) {
            int tok = tok0 + s;
            if (tok < total_tokens) {
                int64_t out_off = ((int64_t)tok * items_per_tok + rem) * 64;
                int64_t in_off = ((int64_t)tok * num_heads + h) * 64;
                uint4 v0 = *reinterpret_cast<const uint4*>(dv + in_off + d0);
                uint4 v1 = *reinterpret_cast<const uint4*>(dv + in_off + d1);
                if constexpr (HAS_DBIAS) {
                    const __nv_bfloat16* p0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
                    const __nv_bfloat16* p1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        db0[k] += bf16_to_f32(p0[k]);
                        db1[k] += bf16_to_f32(p1[k]);
                    }
                }
                *reinterpret_cast<uint4*>(dqkv + out_off + d0) = v0;
                *reinterpret_cast<uint4*>(dqkv + out_off + d1) = v1;
            }
        }
    } else {
        const __nv_bfloat16* src = (qkv_idx == 0) ? dq : dk;
        for (int s = 0; s < TOK_PER_SW; ++s) {
            int tok = tok0 + s;
            if (tok < total_tokens) {
                int64_t out_off = ((int64_t)tok * items_per_tok + rem) * 64;
                int64_t in_off = ((int64_t)tok * num_heads + h) * 64;
                uint4 dy0_v = *reinterpret_cast<const uint4*>(src + in_off + d0);
                uint4 dy1_v = *reinterpret_cast<const uint4*>(src + in_off + d1);
                const __nv_bfloat16* dy0_p = reinterpret_cast<const __nv_bfloat16*>(&dy0_v);
                const __nv_bfloat16* dy1_p = reinterpret_cast<const __nv_bfloat16*>(&dy1_v);

                int cs_base = (tok % seq_per_cam) * 64;
                float4 c0_a = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d0);
                float4 c0_b = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d0 + 4);
                float4 c1_a = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d1);
                float4 c1_b = *reinterpret_cast<const float4*>(cos_f32 + cs_base + d1 + 4);
                float4 s0_a = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d0);
                float4 s0_b = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d0 + 4);
                float4 s1_a = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d1);
                float4 s1_b = *reinterpret_cast<const float4*>(sin_f32 + cs_base + d1 + 4);

                const float c0[8] = {c0_a.x, c0_a.y, c0_a.z, c0_a.w, c0_b.x, c0_b.y, c0_b.z, c0_b.w};
                const float c1[8] = {c1_a.x, c1_a.y, c1_a.z, c1_a.w, c1_b.x, c1_b.y, c1_b.z, c1_b.w};
                const float s0[8] = {s0_a.x, s0_a.y, s0_a.z, s0_a.w, s0_b.x, s0_b.y, s0_b.z, s0_b.w};
                const float s1[8] = {s1_a.x, s1_a.y, s1_a.z, s1_a.w, s1_b.x, s1_b.y, s1_b.z, s1_b.w};

                __nv_bfloat16 dx0_b[8], dx1_b[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    float dy0 = bf16_to_f32(dy0_p[k]);
                    float dy1 = bf16_to_f32(dy1_p[k]);
                    float f0 = dy0 * c0[k] + dy1 * s1[k];
                    float f1 = dy1 * c1[k] - dy0 * s0[k];
                    dx0_b[k] = f32_to_bf16(f0);
                    dx1_b[k] = f32_to_bf16(f1);
                    if constexpr (HAS_DBIAS) {
                        db0[k] += bf16_to_f32(dx0_b[k]);
                        db1[k] += bf16_to_f32(dx1_b[k]);
                    }
                }
                *reinterpret_cast<uint4*>(dqkv + out_off + d0) = *reinterpret_cast<uint4*>(dx0_b);
                *reinterpret_cast<uint4*>(dqkv + out_off + d1) = *reinterpret_cast<uint4*>(dx1_b);
            }
        }
    }

    if constexpr (HAS_DBIAS) {
        float* b_dst = dqkv_bias_f32 + rem * 64;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            atomicAdd(b_dst + d0 + k, db0[k]);
            atomicAdd(b_dst + d1 + k, db1[k]);
        }
    }
}

// --- Qwen D=2560 100%-Occupancy 256-Thread RMSNorm + Quant & Persistent Bwd (with optional fused E5M2 dx_row) ---
__global__ void rmsnorm_amax_calib_k(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ res_in,
    __nv_bfloat16* __restrict__ sum_out,
    const __nv_bfloat16* __restrict__ w,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int D,
    float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;
    const int64_t off = (int64_t)row * D;

    float sq = 0.0f;
    for (int d = threadIdx.x * 2; d < D; d += blockDim.x * 2) {
        float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + off + d));
        if (res_in != nullptr) {
            float2 rv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(res_in + off + d));
            __nv_bfloat162 sb = __floats2bfloat162_rn(v.x + rv.x, v.y + rv.y);
            if (sum_out != nullptr) *reinterpret_cast<__nv_bfloat162*>(sum_out + off + d) = sb;
            v = __bfloat1622float2(sb);
        }
        sq += v.x * v.x + v.y * v.y;
    }
    sq = warp_reduce_sum(sq);
    __shared__ float sdata[8];
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) sdata[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) sdata[0] = rsqrtf(v / (float)D + eps);
    }
    __syncthreads();
    float rstd = sdata[0];

    const __nv_bfloat16* src = (res_in != nullptr && sum_out != nullptr) ? sum_out : x;
    float lmax = 0.0f;
    for (int d = threadIdx.x * 2; d < D; d += blockDim.x * 2) {
        float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(src + off + d));
        if (res_in != nullptr && sum_out == nullptr) {
            float2 rv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(res_in + off + d));
            v = __bfloat1622float2(__floats2bfloat162_rn(v.x + rv.x, v.y + rv.y));
        }
        float2 wv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(w + d));
        lmax = fmaxf(lmax, fmaxf(fabsf(v.x * rstd * wv.x), fabsf(v.y * rstd * wv.y)));
    }
    block_reduce_atomic_max(lmax, next_amax, slot_id);
}

template <bool HAS_RES, bool SAVE_RSTD, bool RECORD_AMAX>
__global__ void rmsnorm_quant_2560_reg_k(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ res_in,
    __nv_bfloat16* __restrict__ sum_out,
    const __nv_bfloat16* __restrict__ w,
    uint8_t* __restrict__ y_e4,
    float* __restrict__ rstd_out,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    float eps
) {
    constexpr int D = 2560;
    constexpr int ITEMS = 5;
    int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int64_t off = (int64_t)row * D;
    const float inv_s = inv_base[slot_id];

    float2 xv[ITEMS];
    float sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int d = (i * 256 + tid) * 2;
        float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + off + d));
        if constexpr (HAS_RES) {
            float2 rv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(res_in + off + d));
            __nv_bfloat162 sb = __floats2bfloat162_rn(v.x + rv.x, v.y + rv.y);
            *reinterpret_cast<__nv_bfloat162*>(sum_out + off + d) = sb;
            v = __bfloat1622float2(sb);
        }
        xv[i] = v;
        sq += v.x * v.x + v.y * v.y;
    }

    sq = warp_reduce_sum(sq);
    __shared__ float sdata[8];
    if (lane == 0) sdata[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 8) ? sdata[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            float r = rsqrtf(v * (1.0f / (float)D) + eps);
            sdata[0] = r;
            if constexpr (SAVE_RSTD) rstd_out[row] = r;
        }
    }
    __syncthreads();
    const float rstd = sdata[0];

    float lmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int d = (i * 256 + tid) * 2;
        float2 wv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(w + d));
        float n0 = xv[i].x * rstd * wv.x;
        float n1 = xv[i].y * rstd * wv.y;
        if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fmaxf(fabsf(n0), fabsf(n1)));
        uint16_t packed = (uint16_t)f32_to_e4m3_raw(n0 * inv_s) | ((uint16_t)f32_to_e4m3_raw(n1 * inv_s) << 8);
        *reinterpret_cast<uint16_t*>(y_e4 + off + d) = packed;
    }

    if constexpr (RECORD_AMAX) {
        lmax = warp_reduce_max(lmax);
        if (lane == 0) sdata[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? sdata[lane] : 0.0f;
            v = warp_reduce_max(v);
            if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
        }
    }
}

template <bool SAVE_RSTD, bool RECORD_AMAX>
__global__ void rmsnorm_quant_generic_k(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ res_in,
    __nv_bfloat16* __restrict__ sum_out,
    const __nv_bfloat16* __restrict__ w,
    uint8_t* __restrict__ y_e4,
    float* __restrict__ rstd_out,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    int D,
    float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;
    const int64_t off = (int64_t)row * D;
    const float inv_s = inv_base[slot_id];

    float sq = 0.0f;
    for (int d = threadIdx.x * 2; d < D; d += blockDim.x * 2) {
        float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + off + d));
        if (res_in != nullptr) {
            float2 rv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(res_in + off + d));
            __nv_bfloat162 sb = __floats2bfloat162_rn(v.x + rv.x, v.y + rv.y);
            *reinterpret_cast<__nv_bfloat162*>(sum_out + off + d) = sb;
            v = __bfloat1622float2(sb);
        }
        sq += v.x * v.x + v.y * v.y;
    }
    sq = warp_reduce_sum(sq);
    __shared__ float sdata[8];
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) sdata[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) {
            float r = rsqrtf(v / (float)D + eps);
            sdata[0] = r;
            if constexpr (SAVE_RSTD) rstd_out[row] = r;
        }
    }
    __syncthreads();
    const float rstd = sdata[0];

    const __nv_bfloat16* src = (res_in != nullptr) ? sum_out : x;
    float lmax = 0.0f;
    for (int d = threadIdx.x * 2; d < D; d += blockDim.x * 2) {
        float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(src + off + d));
        float2 wv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(w + d));
        float n0 = v.x * rstd * wv.x;
        float n1 = v.y * rstd * wv.y;
        if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fmaxf(fabsf(n0), fabsf(n1)));
        uint16_t packed = (uint16_t)f32_to_e4m3_raw(n0 * inv_s) | ((uint16_t)f32_to_e4m3_raw(n1 * inv_s) << 8);
        *reinterpret_cast<uint16_t*>(y_e4 + off + d) = packed;
    }
    if constexpr (RECORD_AMAX) {
        block_reduce_atomic_max(lmax, next_amax, slot_id);
    }
}

template <bool HAS_DRES, bool QUANT_DX>
__global__ void rmsnorm_bwd_2560_opt_k(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ d_res_in,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ dx,
    uint8_t* __restrict__ dx_e5_row,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_dx,
    float* __restrict__ dw_f32,
    int M
) {
    constexpr int D = 2560;
    constexpr int ITEMS = 5;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const float inv_s = QUANT_DX ? inv_base[slot_dx] : 1.0f;
    float lmax = 0.0f;

    float2 w_reg[ITEMS];
    float2 dw_acc[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int d = (i * 256 + tid) * 2;
        w_reg[i] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(w + d));
        dw_acc[i] = make_float2(0.0f, 0.0f);
    }

    __shared__ float ssum[8];

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const int64_t off = (int64_t)row * D;
        const float rstd = rstd_p[row];

        float2 xh[ITEMS];
        float2 g[ITEMS];
        float cdot = 0.0f;

        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            int d = (i * 256 + tid) * 2;
            float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + off + d));
            float2 dyv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(dy + off + d));
            xh[i] = make_float2(xv.x * rstd, xv.y * rstd);
            dw_acc[i].x += dyv.x * xh[i].x;
            dw_acc[i].y += dyv.y * xh[i].y;
            g[i] = make_float2(dyv.x * w_reg[i].x, dyv.y * w_reg[i].y);
            cdot += g[i].x * xh[i].x + g[i].y * xh[i].y;
        }

        cdot = warp_reduce_sum(cdot);
        if (lane == 0) ssum[wid] = cdot;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? ssum[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) ssum[0] = v * (1.0f / (float)D);
        }
        __syncthreads();
        const float mean_gs = ssum[0];

        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            int d = (i * 256 + tid) * 2;
            float dx0 = rstd * (g[i].x - xh[i].x * mean_gs);
            float dx1 = rstd * (g[i].y - xh[i].y * mean_gs);
            if constexpr (HAS_DRES) {
                float2 drv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(d_res_in + off + d));
                float2 drn = __bfloat1622float2(__floats2bfloat162_rn(dx0, dx1));
                dx0 = drv.x + drn.x;
                dx1 = drv.y + drn.y;
            }
            __nv_bfloat162 dxb = __floats2bfloat162_rn(dx0, dx1);
            *reinterpret_cast<__nv_bfloat162*>(dx + off + d) = dxb;
            if constexpr (QUANT_DX) {
                float2 qf = __bfloat1622float2(dxb);
                lmax = fmaxf(lmax, fmaxf(fabsf(qf.x), fabsf(qf.y)));
                uint16_t pe = (uint16_t)f32_to_e5m2_raw(qf.x * inv_s) | ((uint16_t)f32_to_e5m2_raw(qf.y * inv_s) << 8);
                *reinterpret_cast<uint16_t*>(dx_e5_row + off + d) = pe;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int d = (i * 256 + tid) * 2;
        atomicAdd(dw_f32 + d, dw_acc[i].x);
        atomicAdd(dw_f32 + d + 1, dw_acc[i].y);
    }
    if constexpr (QUANT_DX) {
        lmax = warp_reduce_max(lmax);
        if (lane == 0) ssum[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? ssum[lane] : 0.0f;
            v = warp_reduce_max(v);
            if (lane == 0) atomic_max_pos_f32(next_amax + slot_dx, v);
        }
    }
}

// --- MMDiT uint4 Sub-Warp Joint QKV + 4xQK-RMSNorm(48) + RoPE + Pad(64) & Split-Unpad ---
__global__ void mmdit_joint_qkv_norm_rope_pad64_fwd_vec8_k(
    const __nv_bfloat16* __restrict__ act_qkv,
    const __nv_bfloat16* __restrict__ txt_qkv,
    const __nv_bfloat16* __restrict__ nq_w,
    const __nv_bfloat16* __restrict__ nk_w,
    const __nv_bfloat16* __restrict__ naq_w,
    const __nv_bfloat16* __restrict__ nak_w,
    const float* __restrict__ act_cos,
    const float* __restrict__ act_sin,
    const float* __restrict__ txt_cos,
    const float* __restrict__ txt_sin,
    __nv_bfloat16* __restrict__ joint_q,
    __nv_bfloat16* __restrict__ joint_k,
    __nv_bfloat16* __restrict__ joint_v,
    float* __restrict__ rstd_out,
    int B,
    int S_act,
    int S_txt,
    int H,
    float eps
) {
    constexpr int TOK_PER_SW = 4;
    const int S_tot = S_act + S_txt;
    const int total_tok = B * S_tot;
    const int t_groups = (total_tok + TOK_PER_SW - 1) / TOK_PER_SW;
    const int items_per_tok = 3 * H;
    const int sw_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 3;
    if (sw_global >= t_groups * items_per_tok) return;

    const int sub = threadIdx.x & 7;
    const int d = sub * 8;
    const int tg = sw_global / items_per_tok;
    const int rem = sw_global - tg * items_per_tok;
    const int qkv_idx = rem / H;
    const int h = rem - qkv_idx * H;
    const int tok0 = tg * TOK_PER_SW;

    int b = tok0 / S_tot;
    int s_joint = tok0 - b * S_tot;

    if (qkv_idx == 2) {
        #pragma unroll
        for (int k = 0; k < TOK_PER_SW; ++k) {
            int tok = tok0 + k;
            if (tok < total_tok) {
                bool is_act = (s_joint < S_act);
                int s_local = is_act ? s_joint : (s_joint - S_act);
                int S_src = is_act ? S_act : S_txt;
                const __nv_bfloat16* src_base = is_act ? act_qkv : txt_qkv;
                int64_t in_off = (((int64_t)b * S_src + s_local) * items_per_tok + rem) * 48;
                int64_t out_off = (((int64_t)b * S_tot + s_joint) * H + h) * 64;
                uint4 v_val = (sub < 6)
                    ? *reinterpret_cast<const uint4*>(src_base + in_off + d)
                    : make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(joint_v + out_off + d) = v_val;
                ++s_joint;
                if (s_joint == S_tot) { s_joint = 0; ++b; }
            }
        }
        return;
    }

    float wf_act[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float wf_txt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (sub < 6) {
        uint4 wa_v = *reinterpret_cast<const uint4*>(((qkv_idx == 0) ? nq_w : nk_w) + d);
        uint4 wt_v = *reinterpret_cast<const uint4*>(((qkv_idx == 0) ? naq_w : nak_w) + d);
        const __nv_bfloat16* wap = reinterpret_cast<const __nv_bfloat16*>(&wa_v);
        const __nv_bfloat16* wtp = reinterpret_cast<const __nv_bfloat16*>(&wt_v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            wf_act[j] = bf16_to_f32(wap[j]);
            wf_txt[j] = bf16_to_f32(wtp[j]);
        }
    }
    __nv_bfloat16* dst = (qkv_idx == 0) ? joint_q : joint_k;

    #pragma unroll
    for (int k = 0; k < TOK_PER_SW; ++k) {
        int tok = tok0 + k;
        if (tok < total_tok) {
            bool is_act = (s_joint < S_act);
            int s_local = is_act ? s_joint : (s_joint - S_act);
            int S_src = is_act ? S_act : S_txt;
            const __nv_bfloat16* src_base = is_act ? act_qkv : txt_qkv;
            int64_t in_off = (((int64_t)b * S_src + s_local) * items_per_tok + rem) * 48;
            int64_t out_off = (((int64_t)b * S_tot + s_joint) * H + h) * 64;

            float xf[8];
            float sq = 0.0f;
            if (sub < 6) {
                uint4 xv = *reinterpret_cast<const uint4*>(src_base + in_off + d);
                const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    xf[j] = bf16_to_f32(xp[j]);
                    sq += xf[j] * xf[j];
                }
            }
            sq = subwarp8_reduce_sum(sq);
            float rstd = rsqrtf(sq * (1.0f / 48.0f) + eps);
            if (sub == 0 && rstd_out != nullptr) {
                rstd_out[(int64_t)tok * (2 * H) + qkv_idx * H + h] = rstd;
            }

            uint4 y_v = make_uint4(0u, 0u, 0u, 0u);
            if (sub < 6) {
                const float* wf = is_act ? wf_act : wf_txt;
                const float* cos_src = is_act ? act_cos : txt_cos;
                const float* sin_src = is_act ? act_sin : txt_sin;
                float4 ca = *reinterpret_cast<const float4*>(cos_src + s_local * 48 + d);
                float4 cb = *reinterpret_cast<const float4*>(cos_src + s_local * 48 + d + 4);
                float4 sa = *reinterpret_cast<const float4*>(sin_src + s_local * 48 + d);
                float4 sb = *reinterpret_cast<const float4*>(sin_src + s_local * 48 + d + 4);
                const float cs_c[4] = {ca.x, ca.z, cb.x, cb.z};
                const float cs_s[4] = {sa.x, sa.z, sb.x, sb.z};
                __nv_bfloat16 yb[8];
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    float xr = xf[2 * p] * rstd * wf[2 * p];
                    float xi = xf[2 * p + 1] * rstd * wf[2 * p + 1];
                    yb[2 * p] = f32_to_bf16(xr * cs_c[p] - xi * cs_s[p]);
                    yb[2 * p + 1] = f32_to_bf16(xr * cs_s[p] + xi * cs_c[p]);
                }
                y_v = *reinterpret_cast<uint4*>(yb);
            }
            *reinterpret_cast<uint4*>(dst + out_off + d) = y_v;
            ++s_joint;
            if (s_joint == S_tot) { s_joint = 0; ++b; }
        }
    }
}

__global__ void mmdit_joint_qkv_norm_rope_pad64_bwd_vec8_k(
    const __nv_bfloat16* __restrict__ dj_q,
    const __nv_bfloat16* __restrict__ dj_k,
    const __nv_bfloat16* __restrict__ dj_v,
    const __nv_bfloat16* __restrict__ act_qkv,
    const __nv_bfloat16* __restrict__ txt_qkv,
    const __nv_bfloat16* __restrict__ nq_w,
    const __nv_bfloat16* __restrict__ nk_w,
    const __nv_bfloat16* __restrict__ naq_w,
    const __nv_bfloat16* __restrict__ nak_w,
    const float* __restrict__ act_cos,
    const float* __restrict__ act_sin,
    const float* __restrict__ txt_cos,
    const float* __restrict__ txt_sin,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ d_act_qkv,
    __nv_bfloat16* __restrict__ d_txt_qkv,
    float* __restrict__ d_nq_f32,
    float* __restrict__ d_nk_f32,
    float* __restrict__ d_naq_f32,
    float* __restrict__ d_nak_f32,
    int B,
    int S_act,
    int S_txt,
    int H
) {
    const int S_tot = S_act + S_txt;
    const int items_per_tok = 3 * H;
    const int sw = threadIdx.x >> 3;
    const int sub = threadIdx.x & 7;
    const int d = sub * 8;
    const int rem = blockIdx.y * 32 + sw;
    if (rem >= items_per_tok) return;

    const int qkv_idx = rem / H;
    const int h = rem - qkv_idx * H;
    const int total_tok = B * S_tot;

    if (qkv_idx == 2) {
        if (sub < 6) {
            for (int tok = blockIdx.x; tok < total_tok; tok += gridDim.x) {
                int b = tok / S_tot;
                int s_joint = tok - b * S_tot;
                bool is_act = (s_joint < S_act);
                int s_local = is_act ? s_joint : (s_joint - S_act);
                int S_src = is_act ? S_act : S_txt;
                __nv_bfloat16* dst = is_act ? d_act_qkv : d_txt_qkv;
                int64_t out_off = (((int64_t)b * S_src + s_local) * items_per_tok + rem) * 48;
                int64_t in_off = (((int64_t)b * S_tot + s_joint) * H + h) * 64;
                *reinterpret_cast<uint4*>(dst + out_off + d) =
                    *reinterpret_cast<const uint4*>(dj_v + in_off + d);
            }
        }
        return;
    }

    float wf_act[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float wf_txt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (sub < 6) {
        uint4 wa_v = *reinterpret_cast<const uint4*>(((qkv_idx == 0) ? nq_w : nk_w) + d);
        uint4 wt_v = *reinterpret_cast<const uint4*>(((qkv_idx == 0) ? naq_w : nak_w) + d);
        const __nv_bfloat16* wap = reinterpret_cast<const __nv_bfloat16*>(&wa_v);
        const __nv_bfloat16* wtp = reinterpret_cast<const __nv_bfloat16*>(&wt_v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            wf_act[j] = bf16_to_f32(wap[j]);
            wf_txt[j] = bf16_to_f32(wtp[j]);
        }
    }
    float dw_act_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float dw_txt_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const __nv_bfloat16* dj_src = (qkv_idx == 0) ? dj_q : dj_k;

    for (int tok = blockIdx.x; tok < total_tok; tok += gridDim.x) {
        int b = tok / S_tot;
        int s_joint = tok - b * S_tot;
        bool is_act = (s_joint < S_act);
        int s_local = is_act ? s_joint : (s_joint - S_act);
        int S_src = is_act ? S_act : S_txt;

        float cdot = 0.0f;
        float g[8], xh[8];
        float rstd = 0.0f;
        int64_t qkv_off = (((int64_t)b * S_src + s_local) * items_per_tok + rem) * 48;

        if (sub < 6) {
            int64_t dj_off = (((int64_t)b * S_tot + s_joint) * H + h) * 64;
            uint4 dy_v = *reinterpret_cast<const uint4*>(dj_src + dj_off + d);
            const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dy_v);

            const float* cos_src = is_act ? act_cos : txt_cos;
            const float* sin_src = is_act ? act_sin : txt_sin;
            float4 ca = *reinterpret_cast<const float4*>(cos_src + s_local * 48 + d);
            float4 cb = *reinterpret_cast<const float4*>(cos_src + s_local * 48 + d + 4);
            float4 sa = *reinterpret_cast<const float4*>(sin_src + s_local * 48 + d);
            float4 sb = *reinterpret_cast<const float4*>(sin_src + s_local * 48 + d + 4);
            const float cs_c[4] = {ca.x, ca.z, cb.x, cb.z};
            const float cs_s[4] = {sa.x, sa.z, sb.x, sb.z};

            const __nv_bfloat16* x_src = is_act ? act_qkv : txt_qkv;
            uint4 xv = *reinterpret_cast<const uint4*>(x_src + qkv_off + d);
            const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
            rstd = rstd_p[(int64_t)tok * (2 * H) + qkv_idx * H + h];

            const float* wf = is_act ? wf_act : wf_txt;
            float* dw_acc = is_act ? dw_act_acc : dw_txt_acc;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float dyr = bf16_to_f32(dyp[2 * p]);
                float dyi = bf16_to_f32(dyp[2 * p + 1]);
                float dnx_r = dyr * cs_c[p] + dyi * cs_s[p];
                float dnx_i = dyi * cs_c[p] - dyr * cs_s[p];
                float xhr = bf16_to_f32(xp[2 * p]) * rstd;
                float xhi = bf16_to_f32(xp[2 * p + 1]) * rstd;
                xh[2 * p] = xhr;
                xh[2 * p + 1] = xhi;
                dw_acc[2 * p] += dnx_r * xhr;
                dw_acc[2 * p + 1] += dnx_i * xhi;
                g[2 * p] = dnx_r * wf[2 * p];
                g[2 * p + 1] = dnx_i * wf[2 * p + 1];
                cdot += g[2 * p] * xhr + g[2 * p + 1] * xhi;
            }
        }
        cdot = subwarp8_reduce_sum(cdot) * (1.0f / 48.0f);
        if (sub < 6) {
            __nv_bfloat16* dst = is_act ? d_act_qkv : d_txt_qkv;
            __nv_bfloat16 dxb[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                dxb[j] = f32_to_bf16(rstd * (g[j] - xh[j] * cdot));
            }
            *reinterpret_cast<uint4*>(dst + qkv_off + d) = *reinterpret_cast<uint4*>(dxb);
        }
    }

    if (sub < 6) {
        float* dst_act = (qkv_idx == 0) ? d_nq_f32 : d_nk_f32;
        float* dst_txt = (qkv_idx == 0) ? d_naq_f32 : d_nak_f32;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            atomicAdd(dst_act + d + j, dw_act_acc[j]);
            atomicAdd(dst_txt + d + j, dw_txt_acc[j]);
        }
    }
}

__global__ void mmdit_split_unpad_fwd_vec8_k(
    const __nv_bfloat16* __restrict__ joint_out,
    __nv_bfloat16* __restrict__ act_out,
    __nv_bfloat16* __restrict__ txt_out,
    int B,
    int S_act,
    int S_txt,
    int H
) {
    constexpr int TOK_PER_SW = 4;
    const int S_tot = S_act + S_txt;
    const int total_tok = B * S_tot;
    const int t_groups = (total_tok + TOK_PER_SW - 1) / TOK_PER_SW;
    const int sw_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 3;
    if (sw_global >= t_groups * H) return;

    const int sub = threadIdx.x & 7;
    if (sub >= 6) return;
    const int d = sub * 8;

    const int tg = sw_global / H;
    const int h = sw_global - tg * H;
    const int tok0 = tg * TOK_PER_SW;
    int b = tok0 / S_tot;
    int s_joint = tok0 - b * S_tot;

    #pragma unroll
    for (int k = 0; k < TOK_PER_SW; ++k) {
        int tok = tok0 + k;
        if (tok < total_tok) {
            int64_t in_off = (((int64_t)b * S_tot + s_joint) * H + h) * 64;
            if (s_joint < S_act) {
                int64_t out_off = (((int64_t)b * S_act + s_joint) * H + h) * 48;
                *reinterpret_cast<uint4*>(act_out + out_off + d) =
                    *reinterpret_cast<const uint4*>(joint_out + in_off + d);
            } else if (txt_out != nullptr) {
                int s_txt = s_joint - S_act;
                int64_t out_off = (((int64_t)b * S_txt + s_txt) * H + h) * 48;
                *reinterpret_cast<uint4*>(txt_out + out_off + d) =
                    *reinterpret_cast<const uint4*>(joint_out + in_off + d);
            }
            ++s_joint;
            if (s_joint == S_tot) { s_joint = 0; ++b; }
        }
    }
}

__global__ void mmdit_split_unpad_bwd_vec8_k(
    const __nv_bfloat16* __restrict__ d_act_out,
    const __nv_bfloat16* __restrict__ d_txt_out,
    __nv_bfloat16* __restrict__ d_joint_out,
    int B,
    int S_act,
    int S_txt,
    int H
) {
    constexpr int TOK_PER_SW = 4;
    const int S_tot = S_act + S_txt;
    const int total_tok = B * S_tot;
    const int t_groups = (total_tok + TOK_PER_SW - 1) / TOK_PER_SW;
    const int sw_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 3;
    if (sw_global >= t_groups * H) return;

    const int sub = threadIdx.x & 7;
    const int d = sub * 8;
    const int tg = sw_global / H;
    const int h = sw_global - tg * H;
    const int tok0 = tg * TOK_PER_SW;
    int b = tok0 / S_tot;
    int s_joint = tok0 - b * S_tot;

    #pragma unroll
    for (int k = 0; k < TOK_PER_SW; ++k) {
        int tok = tok0 + k;
        if (tok < total_tok) {
            int64_t out_off = (((int64_t)b * S_tot + s_joint) * H + h) * 64;
            uint4 val = make_uint4(0u, 0u, 0u, 0u);
            if (sub < 6) {
                if (s_joint < S_act) {
                    int64_t in_off = (((int64_t)b * S_act + s_joint) * H + h) * 48;
                    val = *reinterpret_cast<const uint4*>(d_act_out + in_off + d);
                } else if (d_txt_out != nullptr) {
                    int s_txt = s_joint - S_act;
                    int64_t in_off = (((int64_t)b * S_txt + s_txt) * H + h) * 48;
                    val = *reinterpret_cast<const uint4*>(d_txt_out + in_off + d);
                }
            }
            *reinterpret_cast<uint4*>(d_joint_out + out_off + d) = val;
            ++s_joint;
            if (s_joint == S_tot) { s_joint = 0; ++b; }
        }
    }
}

// ============================================================================
// 6. CUDA 12.8 cublasLt Autotuned Engine (FP8 & BF16 Vision)
// ============================================================================
struct LtKey {
    int dev, is_bf16, mode, M, N, K;
    int a_e5, b_e5, has_bias, has_beta, fast_accum, sm_target;
    bool operator==(const LtKey& o) const {
        return dev == o.dev && is_bf16 == o.is_bf16 && mode == o.mode
            && M == o.M && N == o.N && K == o.K
            && a_e5 == o.a_e5 && b_e5 == o.b_e5
            && has_bias == o.has_bias && has_beta == o.has_beta
            && fast_accum == o.fast_accum && sm_target == o.sm_target;
    }
};

struct LtKeyHash {
    size_t operator()(const LtKey& k) const {
        size_t h = (size_t)k.M * 1315423911ULL ^ ((size_t)k.N << 16) ^ ((size_t)k.K << 32);
        h ^= ((size_t)k.mode * 733ULL + (size_t)k.is_bf16 * 37ULL + (size_t)k.dev * 13ULL + (size_t)k.sm_target * 97ULL) << 8;
        h ^= (k.a_e5 | (k.b_e5 << 1) | (k.has_bias << 2) | (k.has_beta << 3) | (k.fast_accum << 4)) * 2654435761ULL;
        return h;
    }
};

struct LtPlan {
    cublasLtMatmulDesc_t opDesc;
    cublasLtMatrixLayout_t aDesc, bDesc, cDesc, dDesc;
    cublasLtMatmulAlgo_t bestAlgo;
    int valid;
};

struct LtSerializedAlgo {
    LtKey key;
    cublasLtMatmulAlgo_t algo;
};
static int g_lt_autotune_enabled = 1;
static cublasLtHandle_t g_lt_handles[16] = {nullptr};
static std::unordered_map<LtKey, LtPlan, LtKeyHash> g_lt_cache;
static std::unordered_map<LtKey, cublasLtMatmulAlgo_t, LtKeyHash> g_lt_imported_algos;

inline cublasLtHandle_t get_lt_handle(int dev) {
    if (!g_lt_handles[dev]) {
        cublasLtCreate(&g_lt_handles[dev]);
    }
    return g_lt_handles[dev];
}


// ============================================================================
// 7. v10 Upgrades: Uniform-Head Smem-Transposed Qwen QKV Bwd & MMDiT D=1536 AdaLN
// ============================================================================
template <bool BHSD>
__global__ void qwen_qkv_norm_rope_bwd_smem_e5_k(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ dk,
    const __nv_bfloat16* __restrict__ dv,
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const __nv_bfloat16* __restrict__ cos_p,
    const __nv_bfloat16* __restrict__ sin_p,
    const float* __restrict__ rstd_p,
    uint8_t* __restrict__ dqkv_e5_row,
    uint8_t* __restrict__ dqkv_e5_t,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_dqkv,
    float* __restrict__ dqw_f32,
    float* __restrict__ dkw_f32,
    int B,
    int S,
    int h_q,
    int h_kv
) {
    __shared__ uint8_t smem_b[128][68];  // 17 uint32_t stride -> zero bank conflicts
    __shared__ float s_dw[128];

    const int T = B * S;
    const int h_qk = h_q + h_kv;
    const int h_tot = h_q + 2 * h_kv;
    const int h = blockIdx.y;
    const int m0 = blockIdx.x * 64;
    const int sw = threadIdx.x >> 3;   // 0..31
    const int sub = threadIdx.x & 7;   // 0..7
    const int d0 = sub * 8;
    const int d1 = 64 + d0;
    const float inv_s = inv_base[slot_dqkv];
    float lmax = 0.0f;

    if (h < h_qk && threadIdx.x < 128) {
        s_dw[threadIdx.x] = 0.0f;
    }
    if (m0 + 64 > T) {
        for (int idx = threadIdx.x; idx < 128 * 17; idx += 256) {
            reinterpret_cast<uint32_t*>(&smem_b[0][0])[idx] = 0u;
        }
    }
    __syncthreads();

    if (h >= h_qk) {
        const int vh = h - h_qk;
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            int t = m0 + sw * 2 + k;
            if (t < T) {
                int64_t base_qkv = ((int64_t)t * h_tot + h) * 128;
                int64_t base_v;
                if constexpr (BHSD) {
                    int b = t / S;
                    int s = t - b * S;
                    base_v = (((int64_t)b * h_kv + vh) * S + s) * 128;
                } else {
                    base_v = ((int64_t)t * h_kv + vh) * 128;
                }
                uint4 v0_u4 = *reinterpret_cast<const uint4*>(dv + base_v + d0);
                uint4 v1_u4 = *reinterpret_cast<const uint4*>(dv + base_v + d1);
                const __nv_bfloat16* p0 = reinterpret_cast<const __nv_bfloat16*>(&v0_u4);
                const __nv_bfloat16* p1 = reinterpret_cast<const __nv_bfloat16*>(&v1_u4);
                uint8_t e0[8], e1[8];
                int tk = sw * 2 + k;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float f0 = bf16_to_f32(p0[j]);
                    float f1 = bf16_to_f32(p1[j]);
                    lmax = fmaxf(lmax, fmaxf(fabsf(f0), fabsf(f1)));
                    uint8_t b0 = f32_to_e5m2_raw(f0 * inv_s);
                    uint8_t b1 = f32_to_e5m2_raw(f1 * inv_s);
                    e0[j] = b0;
                    e1[j] = b1;
                    smem_b[d0 + j][tk] = b0;
                    smem_b[d1 + j][tk] = b1;
                }
                *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d0) = *reinterpret_cast<uint64_t*>(e0);
                *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d1) = *reinterpret_cast<uint64_t*>(e1);
            }
        }
    } else {
        const bool is_q = (h < h_q);
        const __nv_bfloat16* w_ptr = is_q ? qw : kw;
        uint4 w0_v = *reinterpret_cast<const uint4*>(w_ptr + d0);
        uint4 w1_v = *reinterpret_cast<const uint4*>(w_ptr + d1);
        const __nv_bfloat16* w0_p = reinterpret_cast<const __nv_bfloat16*>(&w0_v);
        const __nv_bfloat16* w1_p = reinterpret_cast<const __nv_bfloat16*>(&w1_v);
        float wf0[8], wf1[8];
        float dw0_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float dw1_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            wf0[j] = bf16_to_f32(w0_p[j]);
            wf1[j] = bf16_to_f32(w1_p[j]);
        }

        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            int t = m0 + sw * 2 + k;
            if (t < T) {
                int64_t base_qkv = ((int64_t)t * h_tot + h) * 128;
                uint4 dy0_v, dy1_v;
                if (is_q) {
                    int64_t base_q;
                    if constexpr (BHSD) {
                        int b = t / S;
                        int s = t - b * S;
                        base_q = (((int64_t)b * h_q + h) * S + s) * 128;
                    } else {
                        base_q = ((int64_t)t * h_q + h) * 128;
                    }
                    dy0_v = *reinterpret_cast<const uint4*>(dq + base_q + d0);
                    dy1_v = *reinterpret_cast<const uint4*>(dq + base_q + d1);
                } else {
                    int kh = h - h_q;
                    int64_t base_k;
                    if constexpr (BHSD) {
                        int b = t / S;
                        int s = t - b * S;
                        base_k = (((int64_t)b * h_kv + kh) * S + s) * 128;
                    } else {
                        base_k = ((int64_t)t * h_kv + kh) * 128;
                    }
                    dy0_v = *reinterpret_cast<const uint4*>(dk + base_k + d0);
                    dy1_v = *reinterpret_cast<const uint4*>(dk + base_k + d1);
                }
                const __nv_bfloat16* dy0_p = reinterpret_cast<const __nv_bfloat16*>(&dy0_v);
                const __nv_bfloat16* dy1_p = reinterpret_cast<const __nv_bfloat16*>(&dy1_v);

                int64_t base_cs = (int64_t)t * 128;
                uint4 c0_v = *reinterpret_cast<const uint4*>(cos_p + base_cs + d0);
                uint4 c1_v = *reinterpret_cast<const uint4*>(cos_p + base_cs + d1);
                uint4 s0_v = *reinterpret_cast<const uint4*>(sin_p + base_cs + d0);
                uint4 s1_v = *reinterpret_cast<const uint4*>(sin_p + base_cs + d1);
                const __nv_bfloat16* c0_p = reinterpret_cast<const __nv_bfloat16*>(&c0_v);
                const __nv_bfloat16* c1_p = reinterpret_cast<const __nv_bfloat16*>(&c1_v);
                const __nv_bfloat16* s0_p = reinterpret_cast<const __nv_bfloat16*>(&s0_v);
                const __nv_bfloat16* s1_p = reinterpret_cast<const __nv_bfloat16*>(&s1_v);

                uint4 x0_v = *reinterpret_cast<const uint4*>(qkv + base_qkv + d0);
                uint4 x1_v = *reinterpret_cast<const uint4*>(qkv + base_qkv + d1);
                const __nv_bfloat16* x0_p = reinterpret_cast<const __nv_bfloat16*>(&x0_v);
                const __nv_bfloat16* x1_p = reinterpret_cast<const __nv_bfloat16*>(&x1_v);
                float rstd = rstd_p[(int64_t)t * h_qk + h];

                float xh0[8], xh1[8], g0[8], g1[8];
                float cdot = 0.0f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float dy0 = bf16_to_f32(dy0_p[j]);
                    float dy1 = bf16_to_f32(dy1_p[j]);
                    float dnx0 = dy0 * bf16_to_f32(c0_p[j]) + dy1 * bf16_to_f32(s1_p[j]);
                    float dnx1 = dy1 * bf16_to_f32(c1_p[j]) - dy0 * bf16_to_f32(s0_p[j]);
                    xh0[j] = bf16_to_f32(x0_p[j]) * rstd;
                    xh1[j] = bf16_to_f32(x1_p[j]) * rstd;
                    dw0_acc[j] += dnx0 * xh0[j];
                    dw1_acc[j] += dnx1 * xh1[j];
                    g0[j] = dnx0 * wf0[j];
                    g1[j] = dnx1 * wf1[j];
                    cdot += g0[j] * xh0[j] + g1[j] * xh1[j];
                }
                cdot = subwarp8_reduce_sum(cdot) * (1.0f / 128.0f);

                uint8_t e0[8], e1[8];
                int tk = sw * 2 + k;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float dx0 = rstd * (g0[j] - xh0[j] * cdot);
                    float dx1 = rstd * (g1[j] - xh1[j] * cdot);
                    lmax = fmaxf(lmax, fmaxf(fabsf(dx0), fabsf(dx1)));
                    uint8_t b0 = f32_to_e5m2_raw(dx0 * inv_s);
                    uint8_t b1 = f32_to_e5m2_raw(dx1 * inv_s);
                    e0[j] = b0;
                    e1[j] = b1;
                    smem_b[d0 + j][tk] = b0;
                    smem_b[d1 + j][tk] = b1;
                }
                *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d0) = *reinterpret_cast<uint64_t*>(e0);
                *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d1) = *reinterpret_cast<uint64_t*>(e1);
            }
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            atomicAdd(s_dw + d0 + j, dw0_acc[j]);
            atomicAdd(s_dw + d1 + j, dw1_acc[j]);
        }
    }

    __syncthreads();

    if (dqkv_e5_t != nullptr) {
        const int tx16 = threadIdx.x & 15;   // 0..15 -> 64 tokens (4B each)
        const int ty16 = threadIdx.x >> 4;   // 0..15 -> 16 cols per iter
        const int out_m = m0 + tx16 * 4;
        if (out_m < T) {
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                int c = p * 16 + ty16;
                uint32_t v = *reinterpret_cast<const uint32_t*>(&smem_b[c][tx16 * 4]);
                *reinterpret_cast<uint32_t*>(dqkv_e5_t + (int64_t)(h * 128 + c) * T + out_m) = v;
            }
        }
    }

    if (h < h_qk && threadIdx.x < 128) {
        float v = s_dw[threadIdx.x];
        if (v != 0.0f) {
            float* dst = (h < h_q) ? dqw_f32 : dkw_f32;
            atomicAdd(dst + threadIdx.x, v);
        }
    }
    block_reduce_atomic_max(lmax, next_amax, slot_dqkv);
}

// --- MMDiT D=1536 (192 threads = 6 warps, 1 uint4 = 8 BF16 per thread) ---
__global__ void adaln_modulate_amax_calib_1536_k(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ mod_params,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int B, int L, int L_mod, int mod_offset, float eps
) {
    __shared__ float s_sum[6], s_sq[6], s_max[6];
    const int m = blockIdx.x;
    const int tid = threadIdx.x;  // 0..191
    const int col = tid * 8;
    const int b = m / L;
    const int l = m - b * L;
    const int lm = (L_mod == 1) ? 0 : l;
    const __nv_bfloat16* mp = mod_params + ((int64_t)b * L_mod + lm) * 9216 + mod_offset;

    uint4 xv = *reinterpret_cast<const uint4*>(x + (int64_t)m * 1536 + col);
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
    float xf[8];
    float lsum = 0.0f, lsq = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float v = bf16_to_f32(xp[k]);
        xf[k] = v;
        lsum += v;
        lsq += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lsum += __shfl_xor_sync(0xffffffff, lsum, off);
        lsq  += __shfl_xor_sync(0xffffffff, lsq, off);
    }
    const int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) { s_sum[wid] = lsum; s_sq[wid] = lsq; }
    __syncthreads();
    if (wid == 0) {
        float vs = (lane < 6) ? s_sum[lane] : 0.0f;
        float vq = (lane < 6) ? s_sq[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            vs += __shfl_xor_sync(0xffffffff, vs, off);
            vq += __shfl_xor_sync(0xffffffff, vq, off);
        }
        if (lane == 0) {
            float mean = vs * (1.0f / 1536.0f);
            float var = fmaxf(vq * (1.0f / 1536.0f) - mean * mean, 0.0f);
            s_sum[0] = mean;
            s_sq[0] = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    const float mean = s_sum[0], rstd = s_sq[0];

    uint4 shv = *reinterpret_cast<const uint4*>(mp + col);
    uint4 scv = *reinterpret_cast<const uint4*>(mp + 1536 + col);
    const __nv_bfloat16* shp = reinterpret_cast<const __nv_bfloat16*>(&shv);
    const __nv_bfloat16* scp = reinterpret_cast<const __nv_bfloat16*>(&scv);
    float lmax = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float y = (xf[k] - mean) * rstd * (1.0f + bf16_to_f32(scp[k])) + bf16_to_f32(shp[k]);
        lmax = fmaxf(lmax, fabsf(y));
    }
    lmax = warp_reduce_max(lmax);
    if (lane == 0) s_max[wid] = lmax;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 6) ? s_max[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        if (lane == 0 && v > 0.0f) atomicMax(next_amax + slot_id, __float_as_int(v));
    }
}

template <bool SAVE_STATS, bool RECORD_AMAX>
__global__ void adaln_modulate_quant_1536_k(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ mod_params,
    uint8_t* __restrict__ y_row,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int B, int L, int L_mod, int mod_offset, float eps
) {
    __shared__ float s_sum[6], s_sq[6], s_max[6];
    const int m = blockIdx.x;
    const int tid = threadIdx.x;  // 0..191
    const int col = tid * 8;
    const int b = m / L;
    const int l = m - b * L;
    const int lm = (L_mod == 1) ? 0 : l;
    const __nv_bfloat16* mp = mod_params + ((int64_t)b * L_mod + lm) * 9216 + mod_offset;

    uint4 xv = *reinterpret_cast<const uint4*>(x + (int64_t)m * 1536 + col);
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
    float xf[8];
    float lsum = 0.0f, lsq = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float v = bf16_to_f32(xp[k]);
        xf[k] = v;
        lsum += v;
        lsq += v * v;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lsum += __shfl_xor_sync(0xffffffff, lsum, off);
        lsq  += __shfl_xor_sync(0xffffffff, lsq, off);
    }
    const int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) { s_sum[wid] = lsum; s_sq[wid] = lsq; }
    __syncthreads();
    if (wid == 0) {
        float vs = (lane < 6) ? s_sum[lane] : 0.0f;
        float vq = (lane < 6) ? s_sq[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            vs += __shfl_xor_sync(0xffffffff, vs, off);
            vq += __shfl_xor_sync(0xffffffff, vq, off);
        }
        if (lane == 0) {
            float mean = vs * (1.0f / 1536.0f);
            float var = fmaxf(vq * (1.0f / 1536.0f) - mean * mean, 0.0f);
            float rstd = rsqrtf(var + eps);
            s_sum[0] = mean;
            s_sq[0] = rstd;
            if constexpr (SAVE_STATS) {
                mean_out[m] = mean;
                rstd_out[m] = rstd;
            }
        }
    }
    __syncthreads();
    const float mean = s_sum[0], rstd = s_sq[0];
    const float inv_s = inv_base[slot_id];

    uint4 shv = *reinterpret_cast<const uint4*>(mp + col);
    uint4 scv = *reinterpret_cast<const uint4*>(mp + 1536 + col);
    const __nv_bfloat16* shp = reinterpret_cast<const __nv_bfloat16*>(&shv);
    const __nv_bfloat16* scp = reinterpret_cast<const __nv_bfloat16*>(&scv);
    uint8_t e4[8];
    float lmax = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        float y = (xf[k] - mean) * rstd * (1.0f + bf16_to_f32(scp[k])) + bf16_to_f32(shp[k]);
        if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fabsf(y));
        e4[k] = f32_to_e4m3_raw(y * inv_s);
    }
    *reinterpret_cast<uint64_t*>(y_row + (int64_t)m * 1536 + col) = *reinterpret_cast<const uint64_t*>(e4);

    if constexpr (RECORD_AMAX) {
        lmax = warp_reduce_max(lmax);
        if (lane == 0) s_max[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 6) ? s_max[lane] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
            if (lane == 0 && v > 0.0f) atomicMax(next_amax + slot_id, __float_as_int(v));
        }
    }
}

__global__ void gated_add_fwd_1536_k(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ sub_out,
    const __nv_bfloat16* __restrict__ mod_params,
    __nv_bfloat16* __restrict__ out,
    int B, int L, int L_mod, int mod_offset
) {
    const int m = blockIdx.x;
    const int col = threadIdx.x * 8;
    const int b = m / L;
    const int l = m - b * L;
    const int lm = (L_mod == 1) ? 0 : l;
    const __nv_bfloat16* gp = mod_params + ((int64_t)b * L_mod + lm) * 9216 + mod_offset + 3072;

    uint4 xv = *reinterpret_cast<const uint4*>(x + (int64_t)m * 1536 + col);
    uint4 sv = *reinterpret_cast<const uint4*>(sub_out + (int64_t)m * 1536 + col);
    uint4 gv = *reinterpret_cast<const uint4*>(gp + col);
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
    const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(&sv);
    const __nv_bfloat16* g_p = reinterpret_cast<const __nv_bfloat16*>(&gv);
    __nv_bfloat16 res[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        res[k] = f32_to_bf16(bf16_to_f32(xp[k]) + bf16_to_f32(g_p[k]) * bf16_to_f32(sp[k]));
    }
    *reinterpret_cast<uint4*>(out + (int64_t)m * 1536 + col) = *reinterpret_cast<const uint4*>(res);
}

template <bool QUANT_E5, bool HAS_DBIAS>
__global__ void gated_bwd_1536_k(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ sub_out,
    const __nv_bfloat16* __restrict__ mod_params,
    __nv_bfloat16* __restrict__ d_sub_bf16,
    uint8_t* __restrict__ d_sub_e5_row,
    float* __restrict__ dbias_f32_ws,
    float* __restrict__ d_mod_f32,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int B, int L, int L_mod, int mod_offset
) {
    __shared__ float s_max[6];
    const int M = B * L;
    const int tid = threadIdx.x;  // 0..191
    const int col = tid * 8;
    const float inv_s = QUANT_E5 ? inv_base[slot_id] : 1.0f;
    float db_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float lmax = 0.0f;

    for (int m = blockIdx.x; m < M; m += gridDim.x) {
        const int b = m / L;
        const int l = m - b * L;
        const int lm = (L_mod == 1) ? 0 : l;
        const int64_t mod_row_off = ((int64_t)b * L_mod + lm) * 9216 + mod_offset + 3072 + col;

        uint4 dv = *reinterpret_cast<const uint4*>(dout + (int64_t)m * 1536 + col);
        uint4 sv = *reinterpret_cast<const uint4*>(sub_out + (int64_t)m * 1536 + col);
        uint4 gv = *reinterpret_cast<const uint4*>(mod_params + mod_row_off);
        const __nv_bfloat16* dp = reinterpret_cast<const __nv_bfloat16*>(&dv);
        const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(&sv);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);

        float dsub_f[8], dg_f[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float df = bf16_to_f32(dp[k]);
            float sf = bf16_to_f32(sp[k]);
            float gf = bf16_to_f32(gp[k]);
            dsub_f[k] = df * gf;
            dg_f[k] = df * sf;
            if constexpr (HAS_DBIAS) db_acc[k] += dsub_f[k];
        }

        if (L_mod == L) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) d_mod_f32[mod_row_off + k] = dg_f[k];
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) atomicAdd(d_mod_f32 + mod_row_off + k, dg_f[k]);
        }

        if constexpr (QUANT_E5) {
            uint8_t e5[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                lmax = fmaxf(lmax, fabsf(dsub_f[k]));
                e5[k] = f32_to_e5m2_raw(dsub_f[k] * inv_s);
            }
            *reinterpret_cast<uint64_t*>(d_sub_e5_row + (int64_t)m * 1536 + col) = *reinterpret_cast<const uint64_t*>(e5);
        } else {
            __nv_bfloat16 b_out[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) b_out[k] = f32_to_bf16(dsub_f[k]);
            *reinterpret_cast<uint4*>(d_sub_bf16 + (int64_t)m * 1536 + col) = *reinterpret_cast<const uint4*>(b_out);
        }
    }

    if constexpr (HAS_DBIAS) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (db_acc[k] != 0.0f) atomicAdd(dbias_f32_ws + col + k, db_acc[k]);
        }
    }

    if constexpr (QUANT_E5) {
        const int wid = tid >> 5, lane = tid & 31;
        lmax = warp_reduce_max(lmax);
        if (lane == 0) s_max[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 6) ? s_max[lane] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
            if (lane == 0 && v > 0.0f) atomicMax(next_amax + slot_id, __float_as_int(v));
        }
    }
}

template <bool HAS_DRES, bool HAS_DMOD_X>
__global__ void adaln_modulate_bwd_1536_k(
    const __nv_bfloat16* __restrict__ d_mod_x,
    const __nv_bfloat16* __restrict__ d_res_in,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ mod_params,
    const float* __restrict__ mean_p,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ dx_out,
    float* __restrict__ d_mod_f32,
    int B, int L, int L_mod, int mod_offset
) {
    __shared__ float s_dy[2][6], s_dynx[2][6];
    const int M = B * L;
    const int tid = threadIdx.x;  // 0..191
    const int col = tid * 8;
    const int wid = tid >> 5, lane = tid & 31;

    int iter = 0;
    for (int m = blockIdx.x; m < M; m += gridDim.x, ++iter) {
        const int buf = iter & 1;
        const int b = m / L;
        const int l = m - b * L;
        const int lm = (L_mod == 1) ? 0 : l;
        const int64_t mod_base = ((int64_t)b * L_mod + lm) * 9216 + mod_offset;

        if constexpr (!HAS_DMOD_X) {
            if constexpr (HAS_DRES) {
                uint4 drv = *reinterpret_cast<const uint4*>(d_res_in + (int64_t)m * 1536 + col);
                *reinterpret_cast<uint4*>(dx_out + (int64_t)m * 1536 + col) = drv;
            }
            continue;
        }

        uint4 dmv = *reinterpret_cast<const uint4*>(d_mod_x + (int64_t)m * 1536 + col);
        uint4 xv = *reinterpret_cast<const uint4*>(x + (int64_t)m * 1536 + col);
        uint4 scv = *reinterpret_cast<const uint4*>(mod_params + mod_base + 1536 + col);
        const __nv_bfloat16* dmp = reinterpret_cast<const __nv_bfloat16*>(&dmv);
        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
        const __nv_bfloat16* scp = reinterpret_cast<const __nv_bfloat16*>(&scv);

        const float mean = mean_p[m];
        const float rstd = rstd_p[m];

        float nx[8], dy[8], dsh[8], dsc[8];
        float l_dy = 0.0f, l_dynx = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float dm = bf16_to_f32(dmp[k]);
            float xk = bf16_to_f32(xp[k]);
            float sc = bf16_to_f32(scp[k]);
            float n_x = (xk - mean) * rstd;
            nx[k] = n_x;
            dsh[k] = dm;
            dsc[k] = dm * n_x;
            float dyk = dm * (1.0f + sc);
            dy[k] = dyk;
            l_dy += dyk;
            l_dynx += dyk * n_x;
        }

        if (L_mod == L) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                d_mod_f32[mod_base + col + k] = dsh[k];
                d_mod_f32[mod_base + 1536 + col + k] = dsc[k];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                atomicAdd(d_mod_f32 + mod_base + col + k, dsh[k]);
                atomicAdd(d_mod_f32 + mod_base + 1536 + col + k, dsc[k]);
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            l_dy   += __shfl_xor_sync(0xffffffff, l_dy, off);
            l_dynx += __shfl_xor_sync(0xffffffff, l_dynx, off);
        }
        if (lane == 0) {
            s_dy[buf][wid] = l_dy;
            s_dynx[buf][wid] = l_dynx;
        }
        __syncthreads();
        if (wid == 0) {
            float v1 = (lane < 6) ? s_dy[buf][lane] : 0.0f;
            float v2 = (lane < 6) ? s_dynx[buf][lane] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                v1 += __shfl_xor_sync(0xffffffff, v1, off);
                v2 += __shfl_xor_sync(0xffffffff, v2, off);
            }
            if (lane == 0) {
                s_dy[buf][0] = v1 * (1.0f / 1536.0f);
                s_dynx[buf][0] = v2 * (1.0f / 1536.0f);
            }
        }
        __syncthreads();
        const float c1 = s_dy[buf][0];
        const float c2 = s_dynx[buf][0];

        __nv_bfloat16 out_b[8];
        if constexpr (HAS_DRES) {
            uint4 drv = *reinterpret_cast<const uint4*>(d_res_in + (int64_t)m * 1536 + col);
            const __nv_bfloat16* drp = reinterpret_cast<const __nv_bfloat16*>(&drv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                out_b[k] = f32_to_bf16(bf16_to_f32(drp[k]) + (dy[k] - c1 - nx[k] * c2) * rstd);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                out_b[k] = f32_to_bf16((dy[k] - c1 - nx[k] * c2) * rstd);
            }
        }
        *reinterpret_cast<uint4*>(dx_out + (int64_t)m * 1536 + col) = *reinterpret_cast<const uint4*>(out_b);
    }
}


// ============================================================================
// 8. v11 Zero-Barrier 1-Warp/Row Vision LN(1024), Zero-Contention MMDiT Bwd(1536), & HD=48 JointQKV
// ============================================================================
template <bool SAVE_STATS>
__global__ void ln_fwd_1024_w8_k(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int M,
    float eps
) {
    const int wid = threadIdx.x >> 5;   // 0..7 -> 8 rows per CTA
    const int lane = threadIdx.x & 31;  // 0..31
    const int row = blockIdx.x * 8 + wid;
    if (row >= M) return;

    float xf[4][8], gf[4][8], bf[4][8];
    float s = 0.0f, sq = 0.0f;
    const int64_t row_off = (int64_t)row * 1024;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int col = c * 256 + lane * 8;
        uint4 xv = *reinterpret_cast<const uint4*>(x + row_off + col);
        uint4 gv = *reinterpret_cast<const uint4*>(gamma + col);
        uint4 bv = *reinterpret_cast<const uint4*>(beta + col);
        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
        const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&bv);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = bf16_to_f32(xp[k]);
            xf[c][k] = v;
            gf[c][k] = bf16_to_f32(gp[k]);
            bf[c][k] = bf16_to_f32(bp[k]);
            s += v;
            sq += v * v;
        }
    }
    s = warp_reduce_sum(s);
    sq = warp_reduce_sum(sq);
    const float mean = s * (1.0f / 1024.0f);
    const float var = fmaxf(sq * (1.0f / 1024.0f) - mean * mean, 0.0f);
    const float rstd = rsqrtf(var + eps);
    if constexpr (SAVE_STATS) {
        if (lane == 0) {
            mean_out[row] = mean;
            rstd_out[row] = rstd;
        }
    }
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int col = c * 256 + lane * 8;
        __nv_bfloat16 out_b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            out_b[k] = f32_to_bf16((xf[c][k] - mean) * rstd * gf[c][k] + bf[c][k]);
        }
        *reinterpret_cast<uint4*>(y_out + row_off + col) = *reinterpret_cast<const uint4*>(out_b);
    }
}

template <bool HAS_DRES, bool HAS_DRES_SUM>
__global__ void ln_bwd_1024_w8_k(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ d_res_in,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__ mean_p,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ dx,
    float* __restrict__ dgamma_f32,
    float* __restrict__ dbeta_f32,
    float* __restrict__ dres_sum_f32,
    int M
) {
    __shared__ float s_red[8][256];
    const int wid = threadIdx.x >> 5;   // 0..7
    const int lane = threadIdx.x & 31;  // 0..31

    float gf[4][8];
    float dg_acc[4][8] = {{0}};
    float db_acc[4][8] = {{0}};
    float dr_acc[4][8] = {{0}};
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int col = c * 256 + lane * 8;
        uint4 gv = *reinterpret_cast<const uint4*>(gamma + col);
        const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
        #pragma unroll
        for (int k = 0; k < 8; ++k) gf[c][k] = bf16_to_f32(gp[k]);
    }

    const int total_warps = gridDim.x * 8;
    for (int row = blockIdx.x * 8 + wid; row < M; row += total_warps) {
        const int64_t row_off = (int64_t)row * 1024;
        const float mean = mean_p[row];
        const float rstd = rstd_p[row];

        float xh[4][8], g[4][8];
        float s_g = 0.0f, s_gxh = 0.0f;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int col = c * 256 + lane * 8;
            uint4 xv = *reinterpret_cast<const uint4*>(x + row_off + col);
            uint4 dyv = *reinterpret_cast<const uint4*>(dy + row_off + col);
            const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
            const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dyv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dyk = bf16_to_f32(dyp[k]);
                float xhk = (bf16_to_f32(xp[k]) - mean) * rstd;
                xh[c][k] = xhk;
                dg_acc[c][k] += dyk * xhk;
                db_acc[c][k] += dyk;
                float gk = dyk * gf[c][k];
                g[c][k] = gk;
                s_g += gk;
                s_gxh += gk * xhk;
            }
        }
        const float c1 = warp_reduce_sum(s_g) * (1.0f / 1024.0f);
        const float c2 = warp_reduce_sum(s_gxh) * (1.0f / 1024.0f);

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int col = c * 256 + lane * 8;
            uint4 drv;
            if constexpr (HAS_DRES) drv = *reinterpret_cast<const uint4*>(d_res_in + row_off + col);
            const __nv_bfloat16* drp = reinterpret_cast<const __nv_bfloat16*>(&drv);
            __nv_bfloat16 out_b[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dxf = rstd * (g[c][k] - c1 - xh[c][k] * c2);
                if constexpr (HAS_DRES) {
                    float drf = bf16_to_f32(drp[k]);
                    if constexpr (HAS_DRES_SUM) dr_acc[c][k] += drf;
                    dxf = drf + bf16_to_f32(f32_to_bf16(dxf));
                }
                out_b[k] = f32_to_bf16(dxf);
            }
            *reinterpret_cast<uint4*>(dx + row_off + col) = *reinterpret_cast<const uint4*>(out_b);
        }
    }

    // Reduce across 8 warps in CTA before 1 global atomicAdd per CTA (only 264 CTAs!)
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int base_col = c * 256;
        #pragma unroll
        for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = dg_acc[c][k];
        __syncthreads();
        float sum_g = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) sum_g += s_red[w][threadIdx.x];
        atomicAdd(dgamma_f32 + base_col + threadIdx.x, sum_g);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = db_acc[c][k];
        __syncthreads();
        float sum_b = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) sum_b += s_red[w][threadIdx.x];
        atomicAdd(dbeta_f32 + base_col + threadIdx.x, sum_b);
        __syncthreads();

        if constexpr (HAS_DRES && HAS_DRES_SUM) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = dr_acc[c][k];
            __syncthreads();
            float sum_r = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) sum_r += s_red[w][threadIdx.x];
            atomicAdd(dres_sum_f32 + base_col + threadIdx.x, sum_r);
            __syncthreads();
        }
    }
}

// Per-batch (`b = blockIdx.y`), 1-warp/row (`256` threads = 8 warps), register-accumulated MMDiT gated_bwd & adaln_bwd!
template <bool QUANT_E5, bool HAS_DBIAS>
__global__ void gated_bwd_1536_w8_k(
    const __nv_bfloat16* __restrict__ dout,
    const __nv_bfloat16* __restrict__ sub_out,
    const __nv_bfloat16* __restrict__ mod_params,
    __nv_bfloat16* __restrict__ d_sub_bf16,
    uint8_t* __restrict__ d_sub_e5_row,
    float* __restrict__ dbias_f32_ws,
    float* __restrict__ d_mod_f32,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int B, int L, int L_mod, int mod_offset
) {
    __shared__ float s_red[8][256];
    const int wid = threadIdx.x >> 5;   // 0..7
    const int lane = threadIdx.x & 31;  // 0..31
    const int b = blockIdx.y;
    const float inv_s = QUANT_E5 ? inv_base[slot_id] : 1.0f;

    float db_acc[6][8] = {{0}};
    float dg_bcast_acc[6][8] = {{0}};
    float lmax = 0.0f;

    float g_bcast[6][8];
    if (L_mod == 1) {
        const int64_t g_base = (int64_t)b * 9216 + mod_offset + 3072;
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            uint4 gv = *reinterpret_cast<const uint4*>(mod_params + g_base + c * 256 + lane * 8);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) g_bcast[c][k] = bf16_to_f32(gp[k]);
        }
    }

    const int l_stride = gridDim.x * 8;
    for (int l = blockIdx.x * 8 + wid; l < L; l += l_stride) {
        const int m = b * L + l;
        const int64_t row_off = (int64_t)m * 1536;
        const int64_t g_row_off = ((int64_t)b * L_mod + ((L_mod == 1) ? 0 : l)) * 9216 + mod_offset + 3072;

        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            const int col = c * 256 + lane * 8;
            uint4 dv = *reinterpret_cast<const uint4*>(dout + row_off + col);
            uint4 sv = *reinterpret_cast<const uint4*>(sub_out + row_off + col);
            const __nv_bfloat16* dp = reinterpret_cast<const __nv_bfloat16*>(&dv);
            const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(&sv);

            float gf[8];
            if (L_mod == 1) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) gf[k] = g_bcast[c][k];
            } else {
                uint4 gv = *reinterpret_cast<const uint4*>(mod_params + g_row_off + col);
                const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
                #pragma unroll
                for (int k = 0; k < 8; ++k) gf[k] = bf16_to_f32(gp[k]);
            }

            float dsub_f[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float df = bf16_to_f32(dp[k]);
                float sf = bf16_to_f32(sp[k]);
                float ds = df * gf[k];
                float dg = df * sf;
                dsub_f[k] = ds;
                if constexpr (HAS_DBIAS) db_acc[c][k] += ds;
                if (L_mod == 1) {
                    dg_bcast_acc[c][k] += dg;
                } else {
                    d_mod_f32[g_row_off + col + k] = dg;
                }
            }

            if constexpr (QUANT_E5) {
                uint8_t e5[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    lmax = fmaxf(lmax, fabsf(dsub_f[k]));
                    e5[k] = f32_to_e5m2_raw(dsub_f[k] * inv_s);
                }
                *reinterpret_cast<uint64_t*>(d_sub_e5_row + row_off + col) = *reinterpret_cast<const uint64_t*>(e5);
            } else {
                __nv_bfloat16 b_out[8];
                #pragma unroll
                for (int k = 0; k < 8; ++k) b_out[k] = f32_to_bf16(dsub_f[k]);
                *reinterpret_cast<uint4*>(d_sub_bf16 + row_off + col) = *reinterpret_cast<const uint4*>(b_out);
            }
        }
    }

    if (L_mod == 1) {
        const int64_t g_base = (int64_t)b * 9216 + mod_offset + 3072;
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = dg_bcast_acc[c][k];
            __syncthreads();
            float sum_dg = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) sum_dg += s_red[w][threadIdx.x];
            if (gridDim.x == 1) {
                d_mod_f32[g_base + c * 256 + threadIdx.x] = sum_dg;
            } else if (sum_dg != 0.0f) {
                atomicAdd(d_mod_f32 + g_base + c * 256 + threadIdx.x, sum_dg);
            }
            __syncthreads();
        }
    }

    if constexpr (HAS_DBIAS) {
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = db_acc[c][k];
            __syncthreads();
            float sum_db = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) sum_db += s_red[w][threadIdx.x];
            if (sum_db != 0.0f) atomicAdd(dbias_f32_ws + c * 256 + threadIdx.x, sum_db);
            __syncthreads();
        }
    }

    if constexpr (QUANT_E5) {
        block_reduce_atomic_max(lmax, next_amax, slot_id);
    }
}

template <bool HAS_DRES>
__global__ void adaln_modulate_bwd_1536_w8_k(
    const __nv_bfloat16* __restrict__ d_mod_x,
    const __nv_bfloat16* __restrict__ d_res_in,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ mod_params,
    const float* __restrict__ mean_p,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ dx_out,
    float* __restrict__ d_mod_f32,
    int B, int L, int L_mod, int mod_offset
) {
    __shared__ float s_red[8][256];
    const int wid = threadIdx.x >> 5;   // 0..7
    const int lane = threadIdx.x & 31;  // 0..31
    const int b = blockIdx.y;

    float dsh_bcast_acc[6][8] = {{0}};
    float dsc_bcast_acc[6][8] = {{0}};
    float sc_bcast[6][8];

    if (L_mod == 1) {
        const int64_t sc_base = (int64_t)b * 9216 + mod_offset + 1536;
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            uint4 scv = *reinterpret_cast<const uint4*>(mod_params + sc_base + c * 256 + lane * 8);
            const __nv_bfloat16* scp = reinterpret_cast<const __nv_bfloat16*>(&scv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) sc_bcast[c][k] = bf16_to_f32(scp[k]);
        }
    }

    const int l_stride = gridDim.x * 8;
    for (int l = blockIdx.x * 8 + wid; l < L; l += l_stride) {
        const int m = b * L + l;
        const int64_t row_off = (int64_t)m * 1536;
        const int64_t mod_base = ((int64_t)b * L_mod + ((L_mod == 1) ? 0 : l)) * 9216 + mod_offset;
        const float mean = mean_p[m];
        const float rstd = rstd_p[m];

        float nx[6][8], dy[6][8];
        float l_dy = 0.0f, l_dynx = 0.0f;

        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            const int col = c * 256 + lane * 8;
            uint4 dmv = *reinterpret_cast<const uint4*>(d_mod_x + row_off + col);
            uint4 xv = *reinterpret_cast<const uint4*>(x + row_off + col);
            const __nv_bfloat16* dmp = reinterpret_cast<const __nv_bfloat16*>(&dmv);
            const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);

            float scf[8];
            if (L_mod == 1) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) scf[k] = sc_bcast[c][k];
            } else {
                uint4 scv = *reinterpret_cast<const uint4*>(mod_params + mod_base + 1536 + col);
                const __nv_bfloat16* scp = reinterpret_cast<const __nv_bfloat16*>(&scv);
                #pragma unroll
                for (int k = 0; k < 8; ++k) scf[k] = bf16_to_f32(scp[k]);
            }

            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dm = bf16_to_f32(dmp[k]);
                float n_x = (bf16_to_f32(xp[k]) - mean) * rstd;
                nx[c][k] = n_x;
                float dyk = dm * (1.0f + scf[k]);
                dy[c][k] = dyk;
                l_dy += dyk;
                l_dynx += dyk * n_x;
                if (L_mod == 1) {
                    dsh_bcast_acc[c][k] += dm;
                    dsc_bcast_acc[c][k] += dm * n_x;
                } else {
                    d_mod_f32[mod_base + col + k] = dm;
                    d_mod_f32[mod_base + 1536 + col + k] = dm * n_x;
                }
            }
        }

        const float c1 = warp_reduce_sum(l_dy) * (1.0f / 1536.0f);
        const float c2 = warp_reduce_sum(l_dynx) * (1.0f / 1536.0f);

        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            const int col = c * 256 + lane * 8;
            __nv_bfloat16 out_b[8];
            if constexpr (HAS_DRES) {
                uint4 drv = *reinterpret_cast<const uint4*>(d_res_in + row_off + col);
                const __nv_bfloat16* drp = reinterpret_cast<const __nv_bfloat16*>(&drv);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    out_b[k] = f32_to_bf16(bf16_to_f32(drp[k]) + (dy[c][k] - c1 - nx[c][k] * c2) * rstd);
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    out_b[k] = f32_to_bf16((dy[c][k] - c1 - nx[c][k] * c2) * rstd);
                }
            }
            *reinterpret_cast<uint4*>(dx_out + row_off + col) = *reinterpret_cast<const uint4*>(out_b);
        }
    }

    if (L_mod == 1) {
        const int64_t mod_base = (int64_t)b * 9216 + mod_offset;
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = dsh_bcast_acc[c][k];
            __syncthreads();
            float s1 = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) s1 += s_red[w][threadIdx.x];
            if (gridDim.x == 1) d_mod_f32[mod_base + c * 256 + threadIdx.x] = s1;
            else if (s1 != 0.0f) atomicAdd(d_mod_f32 + mod_base + c * 256 + threadIdx.x, s1);
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = dsc_bcast_acc[c][k];
            __syncthreads();
            float s2 = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) s2 += s_red[w][threadIdx.x];
            if (gridDim.x == 1) d_mod_f32[mod_base + 1536 + c * 256 + threadIdx.x] = s2;
            else if (s2 != 0.0f) atomicAdd(d_mod_f32 + 1536 + c * 256 + threadIdx.x, s2);
            __syncthreads();
        }
    }
}

// Native HD=48 (zero-pad!) MMDiT Joint QKV & Split for cuDNN 9.12 SDPA
__global__ void mmdit_joint_qkv_norm_rope_48_fwd_vec8_k(
    const __nv_bfloat16* __restrict__ act_qkv,
    const __nv_bfloat16* __restrict__ txt_qkv,
    const __nv_bfloat16* __restrict__ nq_w,
    const __nv_bfloat16* __restrict__ nk_w,
    const __nv_bfloat16* __restrict__ naq_w,
    const __nv_bfloat16* __restrict__ nak_w,
    const float* __restrict__ act_cos,
    const float* __restrict__ act_sin,
    const float* __restrict__ txt_cos,
    const float* __restrict__ txt_sin,
    __nv_bfloat16* __restrict__ joint_q,
    __nv_bfloat16* __restrict__ joint_k,
    __nv_bfloat16* __restrict__ joint_v,
    float* __restrict__ rstd_out,
    int B, int S_act, int S_txt, int H, float eps
) {
    constexpr int TOK_PER_SW = 4;
    const int S_tot = S_act + S_txt;
    const int total_tok = B * S_tot;
    const int t_groups = (total_tok + TOK_PER_SW - 1) / TOK_PER_SW;
    const int items_per_tok = 3 * H;
    const int sw_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 3;
    if (sw_global >= t_groups * items_per_tok) return;

    const int sub = threadIdx.x & 7;
    const int d = sub * 8;
    const int tg = sw_global / items_per_tok;
    const int rem = sw_global - tg * items_per_tok;
    const int qkv_idx = rem / H;
    const int h = rem - qkv_idx * H;
    const int tok0 = tg * TOK_PER_SW;

    int b = tok0 / S_tot;
    int s_joint = tok0 - b * S_tot;

    if (qkv_idx == 2) {
        if (sub < 6) {
            #pragma unroll
            for (int k = 0; k < TOK_PER_SW; ++k) {
                int tok = tok0 + k;
                if (tok < total_tok) {
                    bool is_act = (s_joint < S_act);
                    int s_local = is_act ? s_joint : (s_joint - S_act);
                    int S_src = is_act ? S_act : S_txt;
                    const __nv_bfloat16* src_base = is_act ? act_qkv : txt_qkv;
                    int64_t in_off = (((int64_t)b * S_src + s_local) * items_per_tok + rem) * 48;
                    int64_t out_off = (((int64_t)b * S_tot + s_joint) * H + h) * 48;
                    *reinterpret_cast<uint4*>(joint_v + out_off + d) =
                        *reinterpret_cast<const uint4*>(src_base + in_off + d);
                    ++s_joint;
                    if (s_joint == S_tot) { s_joint = 0; ++b; }
                }
            }
        }
        return;
    }

    float wf_act[8] = {0}, wf_txt[8] = {0};
    if (sub < 6) {
        uint4 wa_v = *reinterpret_cast<const uint4*>(((qkv_idx == 0) ? nq_w : nk_w) + d);
        uint4 wt_v = *reinterpret_cast<const uint4*>(((qkv_idx == 0) ? naq_w : nak_w) + d);
        const __nv_bfloat16* wap = reinterpret_cast<const __nv_bfloat16*>(&wa_v);
        const __nv_bfloat16* wtp = reinterpret_cast<const __nv_bfloat16*>(&wt_v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            wf_act[j] = bf16_to_f32(wap[j]);
            wf_txt[j] = bf16_to_f32(wtp[j]);
        }
    }
    __nv_bfloat16* dst = (qkv_idx == 0) ? joint_q : joint_k;

    #pragma unroll
    for (int k = 0; k < TOK_PER_SW; ++k) {
        int tok = tok0 + k;
        if (tok < total_tok) {
            bool is_act = (s_joint < S_act);
            int s_local = is_act ? s_joint : (s_joint - S_act);
            int S_src = is_act ? S_act : S_txt;
            const __nv_bfloat16* src_base = is_act ? act_qkv : txt_qkv;
            int64_t in_off = (((int64_t)b * S_src + s_local) * items_per_tok + rem) * 48;
            int64_t out_off = (((int64_t)b * S_tot + s_joint) * H + h) * 48;

            float xf[8];
            float sq = 0.0f;
            if (sub < 6) {
                uint4 xv = *reinterpret_cast<const uint4*>(src_base + in_off + d);
                const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    xf[j] = bf16_to_f32(xp[j]);
                    sq += xf[j] * xf[j];
                }
            }
            sq = subwarp8_reduce_sum(sq);
            float rstd = rsqrtf(sq * (1.0f / 48.0f) + eps);
            if (sub == 0 && rstd_out != nullptr) {
                rstd_out[(int64_t)tok * (2 * H) + qkv_idx * H + h] = rstd;
            }

            if (sub < 6) {
                const float* wf = is_act ? wf_act : wf_txt;
                const float* cos_src = is_act ? act_cos : txt_cos;
                const float* sin_src = is_act ? act_sin : txt_sin;
                float4 ca = *reinterpret_cast<const float4*>(cos_src + s_local * 48 + d);
                float4 cb = *reinterpret_cast<const float4*>(cos_src + s_local * 48 + d + 4);
                float4 sa = *reinterpret_cast<const float4*>(sin_src + s_local * 48 + d);
                float4 sb = *reinterpret_cast<const float4*>(sin_src + s_local * 48 + d + 4);
                const float cs_c[4] = {ca.x, ca.z, cb.x, cb.z};
                const float cs_s[4] = {sa.x, sa.z, sb.x, sb.z};
                __nv_bfloat16 yb[8];
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    float xr = xf[2 * p] * rstd * wf[2 * p];
                    float xi = xf[2 * p + 1] * rstd * wf[2 * p + 1];
                    yb[2 * p] = f32_to_bf16(xr * cs_c[p] - xi * cs_s[p]);
                    yb[2 * p + 1] = f32_to_bf16(xr * cs_s[p] + xi * cs_c[p]);
                }
                *reinterpret_cast<uint4*>(dst + out_off + d) = *reinterpret_cast<const uint4*>(yb);
            }
            ++s_joint;
            if (s_joint == S_tot) { s_joint = 0; ++b; }
        }
    }
}

__global__ void mmdit_joint_qkv_norm_rope_48_bwd_vec8_k(
    const __nv_bfloat16* __restrict__ dj_q,
    const __nv_bfloat16* __restrict__ dj_k,
    const __nv_bfloat16* __restrict__ dj_v,
    const __nv_bfloat16* __restrict__ act_qkv,
    const __nv_bfloat16* __restrict__ txt_qkv,
    const __nv_bfloat16* __restrict__ nq_w,
    const __nv_bfloat16* __restrict__ nk_w,
    const __nv_bfloat16* __restrict__ naq_w,
    const __nv_bfloat16* __restrict__ nak_w,
    const float* __restrict__ act_cos,
    const float* __restrict__ act_sin,
    const float* __restrict__ txt_cos,
    const float* __restrict__ txt_sin,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ d_act_qkv,
    __nv_bfloat16* __restrict__ d_txt_qkv,
    float* __restrict__ d_nq_f32,
    float* __restrict__ d_nk_f32,
    float* __restrict__ d_naq_f32,
    float* __restrict__ d_nak_f32,
    int B, int S_act, int S_txt, int H
) {
    const int S_tot = S_act + S_txt;
    const int items_per_tok = 3 * H;
    const int sw = threadIdx.x >> 3;
    const int sub = threadIdx.x & 7;
    const int d = sub * 8;
    const int rem = blockIdx.y * 32 + sw;
    if (rem >= items_per_tok) return;

    const int qkv_idx = rem / H;
    const int h = rem - qkv_idx * H;
    const int total_tok = B * S_tot;

    if (qkv_idx == 2) {
        if (sub < 6) {
            for (int tok = blockIdx.x; tok < total_tok; tok += gridDim.x) {
                int b = tok / S_tot;
                int s_joint = tok - b * S_tot;
                bool is_act = (s_joint < S_act);
                int s_local = is_act ? s_joint : (s_joint - S_act);
                int S_src = is_act ? S_act : S_txt;
                __nv_bfloat16* dst = is_act ? d_act_qkv : d_txt_qkv;
                int64_t out_off = (((int64_t)b * S_src + s_local) * items_per_tok + rem) * 48;
                int64_t in_off = (((int64_t)b * S_tot + s_joint) * H + h) * 48;
                *reinterpret_cast<uint4*>(dst + out_off + d) =
                    *reinterpret_cast<const uint4*>(dj_v + in_off + d);
            }
        }
        return;
    }

    float wf_act[8] = {0}, wf_txt[8] = {0};
    if (sub < 6) {
        uint4 wa_v = *reinterpret_cast<const uint4*>(((qkv_idx == 0) ? nq_w : nk_w) + d);
        uint4 wt_v = *reinterpret_cast<const uint4*>(((qkv_idx == 0) ? naq_w : nak_w) + d);
        const __nv_bfloat16* wap = reinterpret_cast<const __nv_bfloat16*>(&wa_v);
        const __nv_bfloat16* wtp = reinterpret_cast<const __nv_bfloat16*>(&wt_v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            wf_act[j] = bf16_to_f32(wap[j]);
            wf_txt[j] = bf16_to_f32(wtp[j]);
        }
    }
    float dw_act_acc[8] = {0}, dw_txt_acc[8] = {0};
    const __nv_bfloat16* dj_src = (qkv_idx == 0) ? dj_q : dj_k;

    for (int tok = blockIdx.x; tok < total_tok; tok += gridDim.x) {
        int b = tok / S_tot;
        int s_joint = tok - b * S_tot;
        bool is_act = (s_joint < S_act);
        int s_local = is_act ? s_joint : (s_joint - S_act);
        int S_src = is_act ? S_act : S_txt;

        float cdot = 0.0f;
        float g[8], xh[8];
        float rstd = 0.0f;
        int64_t qkv_off = (((int64_t)b * S_src + s_local) * items_per_tok + rem) * 48;

        if (sub < 6) {
            int64_t dj_off = (((int64_t)b * S_tot + s_joint) * H + h) * 48;
            uint4 dy_v = *reinterpret_cast<const uint4*>(dj_src + dj_off + d);
            const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dy_v);

            const float* cos_src = is_act ? act_cos : txt_cos;
            const float* sin_src = is_act ? act_sin : txt_sin;
            float4 ca = *reinterpret_cast<const float4*>(cos_src + s_local * 48 + d);
            float4 cb = *reinterpret_cast<const float4*>(cos_src + s_local * 48 + d + 4);
            float4 sa = *reinterpret_cast<const float4*>(sin_src + s_local * 48 + d);
            float4 sb = *reinterpret_cast<const float4*>(sin_src + s_local * 48 + d + 4);
            const float cs_c[4] = {ca.x, ca.z, cb.x, cb.z};
            const float cs_s[4] = {sa.x, sa.z, sb.x, sb.z};

            const __nv_bfloat16* x_src = is_act ? act_qkv : txt_qkv;
            uint4 xv = *reinterpret_cast<const uint4*>(x_src + qkv_off + d);
            const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
            rstd = rstd_p[(int64_t)tok * (2 * H) + qkv_idx * H + h];

            const float* wf = is_act ? wf_act : wf_txt;
            float* dw_acc = is_act ? dw_act_acc : dw_txt_acc;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                float dyr = bf16_to_f32(dyp[2 * p]);
                float dyi = bf16_to_f32(dyp[2 * p + 1]);
                float dnx_r = dyr * cs_c[p] + dyi * cs_s[p];
                float dnx_i = dyi * cs_c[p] - dyr * cs_s[p];
                float xhr = bf16_to_f32(xp[2 * p]) * rstd;
                float xhi = bf16_to_f32(xp[2 * p + 1]) * rstd;
                xh[2 * p] = xhr;
                xh[2 * p + 1] = xhi;
                dw_acc[2 * p] += dnx_r * xhr;
                dw_acc[2 * p + 1] += dnx_i * xhi;
                g[2 * p] = dnx_r * wf[2 * p];
                g[2 * p + 1] = dnx_i * wf[2 * p + 1];
                cdot += g[2 * p] * xhr + g[2 * p + 1] * xhi;
            }
        }
        cdot = subwarp8_reduce_sum(cdot) * (1.0f / 48.0f);
        if (sub < 6) {
            __nv_bfloat16* dst = is_act ? d_act_qkv : d_txt_qkv;
            __nv_bfloat16 dxb[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                dxb[j] = f32_to_bf16(rstd * (g[j] - xh[j] * cdot));
            }
            *reinterpret_cast<uint4*>(dst + qkv_off + d) = *reinterpret_cast<const uint4*>(dxb);
        }
    }

    if (sub < 6) {
        float* dst_act = (qkv_idx == 0) ? d_nq_f32 : d_nk_f32;
        float* dst_txt = (qkv_idx == 0) ? d_naq_f32 : d_nak_f32;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            atomicAdd(dst_act + d + j, dw_act_acc[j]);
            atomicAdd(dst_txt + d + j, dw_txt_acc[j]);
        }
    }
}

__global__ void mmdit_split_48_fwd_k(
    const __nv_bfloat16* __restrict__ joint_out,
    __nv_bfloat16* __restrict__ act_out,
    __nv_bfloat16* __restrict__ txt_out,
    int B, int S_act, int S_txt
) {
    const int S_tot = S_act + S_txt;
    const int m = blockIdx.x;
    const int col = threadIdx.x * 8;
    const int b = m / S_tot;
    const int s = m - b * S_tot;
    uint4 v = *reinterpret_cast<const uint4*>(joint_out + (int64_t)m * 1536 + col);
    if (s < S_act) {
        *reinterpret_cast<uint4*>(act_out + (int64_t)(b * S_act + s) * 1536 + col) = v;
    } else if (txt_out != nullptr) {
        *reinterpret_cast<uint4*>(txt_out + (int64_t)(b * S_txt + (s - S_act)) * 1536 + col) = v;
    }
}

__global__ void mmdit_split_48_bwd_k(
    const __nv_bfloat16* __restrict__ d_act_out,
    const __nv_bfloat16* __restrict__ d_txt_out,
    __nv_bfloat16* __restrict__ d_joint_out,
    int B, int S_act, int S_txt
) {
    const int S_tot = S_act + S_txt;
    const int m = blockIdx.x;
    const int col = threadIdx.x * 8;
    const int b = m / S_tot;
    const int s = m - b * S_tot;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (s < S_act) {
        v = *reinterpret_cast<const uint4*>(d_act_out + (int64_t)(b * S_act + s) * 1536 + col);
    } else if (d_txt_out != nullptr) {
        v = *reinterpret_cast<const uint4*>(d_txt_out + (int64_t)(b * S_txt + (s - S_act)) * 1536 + col);
    }
    *reinterpret_cast<uint4*>(d_joint_out + (int64_t)m * 1536 + col) = v;
}


// ============================================================================
// 9. v13 L1-Reload 2-CTA/SM Vision LN Bwd, Zero-Spill AdaLN Bwd, Zero-Copy Split48Quant, & Fused ZeRO-2 AdamW
// ============================================================================
template <bool HAS_DRES, bool HAS_DRES_SUM>
__global__ void ln_bwd_1024_w8_l1_k(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ d_res_in,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__ mean_p,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ dx,
    float* __restrict__ dgamma_f32,
    float* __restrict__ dbeta_f32,
    float* __restrict__ dres_sum_f32,
    int M
) {
    __shared__ float s_red[8][256];
    const int wid = threadIdx.x >> 5;   // 0..7
    const int lane = threadIdx.x & 31;  // 0..31

    float dg_acc[4][8] = {{0}};
    float db_acc[4][8] = {{0}};
    float dr_acc[4][8] = {{0}};

    const int total_warps = gridDim.x * 8;
    for (int row = blockIdx.x * 8 + wid; row < M; row += total_warps) {
        const int64_t row_off = (int64_t)row * 1024;
        const float mean = mean_p[row];
        const float rstd = rstd_p[row];

        float s_g = 0.0f, s_gxh = 0.0f;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int col = c * 256 + lane * 8;
            uint4 xv = *reinterpret_cast<const uint4*>(x + row_off + col);
            uint4 dyv = *reinterpret_cast<const uint4*>(dy + row_off + col);
            uint4 gv = __ldg(reinterpret_cast<const uint4*>(gamma + col));
            const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
            const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dyv);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dyk = bf16_to_f32(dyp[k]);
                float xhk = (bf16_to_f32(xp[k]) - mean) * rstd;
                dg_acc[c][k] += dyk * xhk;
                db_acc[c][k] += dyk;
                float gk = dyk * bf16_to_f32(gp[k]);
                s_g += gk;
                s_gxh += gk * xhk;
            }
        }
        const float c1 = warp_reduce_sum(s_g) * (1.0f / 1024.0f);
        const float c2 = warp_reduce_sum(s_gxh) * (1.0f / 1024.0f);

        // Pass 2: reload x, dy & gamma from hot L1/RO cache (frees 96 regs -> 2 CTAs/SM = 528 CTAs, 0 spills, 0 barriers!)
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int col = c * 256 + lane * 8;
            uint4 xv = *reinterpret_cast<const uint4*>(x + row_off + col);
            uint4 dyv = *reinterpret_cast<const uint4*>(dy + row_off + col);
            uint4 gv = __ldg(reinterpret_cast<const uint4*>(gamma + col));
            uint4 drv;
            if constexpr (HAS_DRES) drv = *reinterpret_cast<const uint4*>(d_res_in + row_off + col);
            const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
            const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dyv);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
            const __nv_bfloat16* drp = reinterpret_cast<const __nv_bfloat16*>(&drv);
            __nv_bfloat16 out_b[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dyk = bf16_to_f32(dyp[k]);
                float xhk = (bf16_to_f32(xp[k]) - mean) * rstd;
                float gk = dyk * bf16_to_f32(gp[k]);
                float dxf = rstd * (gk - c1 - xhk * c2);
                if constexpr (HAS_DRES) {
                    float drf = bf16_to_f32(drp[k]);
                    if constexpr (HAS_DRES_SUM) dr_acc[c][k] += drf;
                    dxf = drf + bf16_to_f32(f32_to_bf16(dxf));
                }
                out_b[k] = f32_to_bf16(dxf);
            }
            *reinterpret_cast<uint4*>(dx + row_off + col) = *reinterpret_cast<const uint4*>(out_b);
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int base_col = c * 256;
        #pragma unroll
        for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = dg_acc[c][k];
        __syncthreads();
        float sum_g = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) sum_g += s_red[w][threadIdx.x];
        atomicAdd(dgamma_f32 + base_col + threadIdx.x, sum_g);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = db_acc[c][k];
        __syncthreads();
        float sum_b = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) sum_b += s_red[w][threadIdx.x];
        atomicAdd(dbeta_f32 + base_col + threadIdx.x, sum_b);
        __syncthreads();

        if constexpr (HAS_DRES && HAS_DRES_SUM) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) s_red[wid][lane * 8 + k] = dr_acc[c][k];
            __syncthreads();
            float sum_r = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) sum_r += s_red[w][threadIdx.x];
            atomicAdd(dres_sum_f32 + base_col + threadIdx.x, sum_r);
            __syncthreads();
        }
    }
}

template <bool HAS_DRES>
__global__ void adaln_modulate_bwd_1536_r1_k(
    const __nv_bfloat16* __restrict__ d_mod_x,
    const __nv_bfloat16* __restrict__ d_res_in,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ mod_params,
    const float* __restrict__ mean_p,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ dx_out,
    float* __restrict__ d_mod_f32,
    int B, int L, int L_mod, int mod_offset
) {
    __shared__ float s_c1[6], s_c2[6];
    const int tid = threadIdx.x;  // 0..191
    const int wid = tid >> 5;     // 0..5
    const int lane = tid & 31;
    const int col = tid * 8;
    const int b = blockIdx.y;

    float dsh_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float dsc_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float sc_bcast[8];
    if (L_mod == 1) {
        const int64_t m_base = (int64_t)b * 9216 + mod_offset + 1536 + col;
        uint4 sv = __ldg(reinterpret_cast<const uint4*>(mod_params + m_base));
        const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(&sv);
        #pragma unroll
        for (int k = 0; k < 8; ++k) sc_bcast[k] = 1.0f + bf16_to_f32(sp[k]);
    }

    for (int l = blockIdx.x; l < L; l += gridDim.x) {
        const int m = b * L + l;
        const int64_t row_off = (int64_t)m * 1536 + col;
        const int64_t m_row_off = ((int64_t)b * L_mod + ((L_mod == 1) ? 0 : l)) * 9216 + mod_offset;

        const float mean = mean_p[m];
        const float rstd = rstd_p[m];

        uint4 xv = *reinterpret_cast<const uint4*>(x + row_off);
        uint4 dmv = *reinterpret_cast<const uint4*>(d_mod_x + row_off);
        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
        const __nv_bfloat16* dmp = reinterpret_cast<const __nv_bfloat16*>(&dmv);

        float scf[8];
        if (L_mod == 1) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) scf[k] = sc_bcast[k];
        } else {
            uint4 sv = *reinterpret_cast<const uint4*>(mod_params + m_row_off + 1536 + col);
            const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(&sv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) scf[k] = 1.0f + bf16_to_f32(sp[k]);
        }

        float xh[8], g[8];
        float sg = 0.0f, sgxh = 0.0f;
        float dsh_l[8], dsc_l[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float dm = bf16_to_f32(dmp[k]);
            float xhk = (bf16_to_f32(xp[k]) - mean) * rstd;
            xh[k] = xhk;
            float gk = dm * scf[k];
            g[k] = gk;
            sg += gk;
            sgxh += gk * xhk;
            dsh_l[k] = dm;
            dsc_l[k] = dm * xhk;
        }
        if (L_mod == 1) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                dsh_acc[k] += dsh_l[k];
                dsc_acc[k] += dsc_l[k];
            }
        } else {
            *reinterpret_cast<float4*>(d_mod_f32 + m_row_off + col) = make_float4(dsh_l[0], dsh_l[1], dsh_l[2], dsh_l[3]);
            *reinterpret_cast<float4*>(d_mod_f32 + m_row_off + col + 4) = make_float4(dsh_l[4], dsh_l[5], dsh_l[6], dsh_l[7]);
            *reinterpret_cast<float4*>(d_mod_f32 + m_row_off + 1536 + col) = make_float4(dsc_l[0], dsc_l[1], dsc_l[2], dsc_l[3]);
            *reinterpret_cast<float4*>(d_mod_f32 + m_row_off + 1536 + col + 4) = make_float4(dsc_l[4], dsc_l[5], dsc_l[6], dsc_l[7]);
        }

        sg = warp_reduce_sum(sg);
        sgxh = warp_reduce_sum(sgxh);
        if (lane == 0) { s_c1[wid] = sg; s_c2[wid] = sgxh; }
        __syncthreads();
        if (wid == 0) {
            float v1 = (lane < 6) ? s_c1[lane] : 0.0f;
            float v2 = (lane < 6) ? s_c2[lane] : 0.0f;
            v1 = warp_reduce_sum(v1);
            v2 = warp_reduce_sum(v2);
            if (lane == 0) {
                s_c1[0] = v1 * (1.0f / 1536.0f);
                s_c2[0] = v2 * (1.0f / 1536.0f);
            }
        }
        __syncthreads();
        const float c1 = s_c1[0];
        const float c2 = s_c2[0];

        uint4 drv;
        if constexpr (HAS_DRES) drv = *reinterpret_cast<const uint4*>(d_res_in + row_off);
        const __nv_bfloat16* drp = reinterpret_cast<const __nv_bfloat16*>(&drv);
        __nv_bfloat16 out_b[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float dxf = rstd * (g[k] - c1 - xh[k] * c2);
            if constexpr (HAS_DRES) {
                dxf = bf16_to_f32(drp[k]) + bf16_to_f32(f32_to_bf16(dxf));
            }
            out_b[k] = f32_to_bf16(dxf);
        }
        *reinterpret_cast<uint4*>(dx_out + row_off) = *reinterpret_cast<const uint4*>(out_b);
        __syncthreads();
    }

    if (L_mod == 1) {
        const int64_t m_base = (int64_t)b * 9216 + mod_offset;
        if (gridDim.x == 1) {
            *reinterpret_cast<float4*>(d_mod_f32 + m_base + col) = make_float4(dsh_acc[0], dsh_acc[1], dsh_acc[2], dsh_acc[3]);
            *reinterpret_cast<float4*>(d_mod_f32 + m_base + col + 4) = make_float4(dsh_acc[4], dsh_acc[5], dsh_acc[6], dsh_acc[7]);
            *reinterpret_cast<float4*>(d_mod_f32 + m_base + 1536 + col) = make_float4(dsc_acc[0], dsc_acc[1], dsc_acc[2], dsc_acc[3]);
            *reinterpret_cast<float4*>(d_mod_f32 + m_base + 1536 + col + 4) = make_float4(dsc_acc[4], dsc_acc[5], dsc_acc[6], dsc_acc[7]);
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                atomicAdd(d_mod_f32 + m_base + col + k, dsh_acc[k]);
                atomicAdd(d_mod_f32 + m_base + 1536 + col + k, dsc_acc[k]);
            }
        }
    }
}

// Zero-copy MMDiT Split + Quant (supports BOTH hd=48 cuDNN [bhsd OR bshd] AND hd=64 FA4 [bshd]!)
// layout_mode: 0 = bshd_pad64 [B, S_tot, 32, 64], 1 = bshd_48 [B, S_tot, 32, 48], 2 = bhsd_48 [B, 32, S_tot, 48]
template <int LAYOUT_MODE, bool DUAL, bool RECORD_AMAX>
__global__ void mmdit_split_quant_64x64_k(
    const __nv_bfloat16* __restrict__ j_out,
    uint8_t* __restrict__ y_row,
    uint8_t* __restrict__ y_col,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int B, int S_sub, int S_tot, int s_offset
) {
    __shared__ uint32_t smem_u32[64][17];
    __shared__ float smax[8];

    const int M = B * S_sub;
    const float inv_s = inv_base[slot_id];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int n0 = blockIdx.x * 64;
    const int m0 = blockIdx.y * 64;
    const int col = n0 + tx * 4;
    const int h = col / 48;
    const int d = col - h * 48;
    const int row0 = m0 + ty * 4;

    float lmax = 0.0f;
    uint32_t r_u32[4] = {0u, 0u, 0u, 0u};

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int row = row0 + k;
        if (row < M) {
            int b = row / S_sub;
            int s = row - b * S_sub;
            int64_t j_off;
            if constexpr (LAYOUT_MODE == 0) {
                j_off = ((int64_t)b * S_tot + s_offset + s) * 2048 + h * 64 + d;
            } else if constexpr (LAYOUT_MODE == 1) {
                j_off = ((int64_t)b * S_tot + s_offset + s) * 1536 + col;
            } else {
                j_off = (((int64_t)b * 32 + h) * S_tot + s_offset + s) * 48 + d;
            }
            uint2 jv = *reinterpret_cast<const uint2*>(j_out + j_off);
            const __nv_bfloat16* jp = reinterpret_cast<const __nv_bfloat16*>(&jv);
            uint8_t e4[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float fv = bf16_to_f32(jp[j]);
                if constexpr (RECORD_AMAX) lmax = fmaxf(lmax, fabsf(fv));
                e4[j] = f32_to_e4m3_raw(fv * inv_s);
            }
            uint32_t packed = *reinterpret_cast<const uint32_t*>(e4);
            r_u32[k] = packed;
            *reinterpret_cast<uint32_t*>(y_row + (int64_t)row * 1536 + col) = packed;
        }
    }

    if constexpr (DUAL) {
        uint32_t c0, c1, c2, c3;
        transpose_4x4_bytes(r_u32[0], r_u32[1], r_u32[2], r_u32[3], c0, c1, c2, c3);
        smem_u32[tx * 4 + 0][ty] = c0;
        smem_u32[tx * 4 + 1][ty] = c1;
        smem_u32[tx * 4 + 2][ty] = c2;
        smem_u32[tx * 4 + 3][ty] = c3;
        __syncthreads();

        const int out_m = m0 + tx * 4;
        const int out_n0 = n0 + ty * 4;
        if (out_m < M) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int out_n = out_n0 + k;
                *reinterpret_cast<uint32_t*>(y_col + (int64_t)out_n * M + out_m) = smem_u32[ty * 4 + k][tx];
            }
        }
    }

    if constexpr (RECORD_AMAX) {
        lmax = warp_reduce_max(lmax);
        int wid = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        if (lane == 0) smax[wid] = lmax;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? smax[lane] : 0.0f;
            v = warp_reduce_max(v);
            if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
        }
    }
}

template <int LAYOUT_MODE>
__global__ void mmdit_split_quant_amax_k(
    const __nv_bfloat16* __restrict__ j_out,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int B, int S_sub, int S_tot, int s_offset
) {
    const int M = B * S_sub;
    const int64_t total_vec4 = (int64_t)M * 384;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float lmax = 0.0f;
    for (int64_t idx = tid; idx < total_vec4; idx += stride) {
        int row = (int)(idx / 384);
        int col = (int)(idx - (int64_t)row * 384) * 4;
        int h = col / 48;
        int d = col - h * 48;
        int b = row / S_sub;
        int s = row - b * S_sub;
        int64_t j_off;
        if constexpr (LAYOUT_MODE == 0) j_off = ((int64_t)b * S_tot + s_offset + s) * 2048 + h * 64 + d;
        else if constexpr (LAYOUT_MODE == 1) j_off = ((int64_t)b * S_tot + s_offset + s) * 1536 + col;
        else j_off = (((int64_t)b * 32 + h) * S_tot + s_offset + s) * 48 + d;
        uint2 jv = *reinterpret_cast<const uint2*>(j_out + j_off);
        const __nv_bfloat16* jp = reinterpret_cast<const __nv_bfloat16*>(&jv);
        #pragma unroll
        for (int j = 0; j < 4; ++j) lmax = fmaxf(lmax, fabsf(bf16_to_f32(jp[j])));
    }
    block_reduce_atomic_max(lmax, next_amax, slot_id);
}

// Direct bhsd [B, 32, S_tot, 48] output for cuDNN SDPA backward (eliminates dj_out.transpose(1,2) copy!)
__global__ void mmdit_split_48_bwd_bhsd_k(
    const __nv_bfloat16* __restrict__ d_act_out,
    const __nv_bfloat16* __restrict__ d_txt_out,
    __nv_bfloat16* __restrict__ dj_bh_out,
    int B, int S_act, int S_txt
) {
    const int S_tot = S_act + S_txt;
    const int tok = blockIdx.x;
    if (tok >= B * S_tot) return;
    const int b = tok / S_tot;
    const int s = tok - b * S_tot;
    const int col = threadIdx.x * 8;  // 192 threads * 8 = 1536
    const int h = col / 48;
    const int d = col - h * 48;       // col % 8 == 0, 48 % 8 == 0 -> d is 0..40 Inside single head h!

    uint4 v;
    if (s < S_act) {
        v = *reinterpret_cast<const uint4*>(d_act_out + ((int64_t)b * S_act + s) * 1536 + col);
    } else if (d_txt_out != nullptr) {
        v = *reinterpret_cast<const uint4*>(d_txt_out + ((int64_t)b * S_txt + (s - S_act)) * 1536 + col);
    } else {
        v = make_uint4(0, 0, 0, 0);
    }
    *reinterpret_cast<uint4*>(dj_bh_out + (((int64_t)b * 32 + h) * S_tot + s) * 48 + d) = v;
}

// Fused ZeRO-2 FP64 Norm^2 + Device InvScale + Single-Pass FP32-Master / BF16-IO AdamW
__global__ void bf16_sum_sq_f64_k(
    const __nv_bfloat16* __restrict__ g,
    double* __restrict__ out_sum_sq,
    int64_t N
) {
    __shared__ double s_sum[8];
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t n8 = N >> 3;

    double acc = 0.0;
    for (int64_t i = tid; i < n8; i += stride) {
        uint4 v = *reinterpret_cast<const uint4*>(g + (i << 3));
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            double d = (double)bf16_to_f32(p[k]);
            acc += d * d;
        }
    }
    for (int64_t idx = (n8 << 3) + tid; idx < N; idx += stride) {
        double d = (double)bf16_to_f32(g[idx]);
        acc += d * d;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        int hi = __double2hiint(acc);
        int lo = __double2loint(acc);
        hi = __shfl_xor_sync(0xffffffff, hi, off);
        lo = __shfl_xor_sync(0xffffffff, lo, off);
        acc += __hiloint2double(hi, lo);
    }
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) s_sum[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        double v = (lane < 8) ? s_sum[lane] : 0.0;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            int hi = __double2hiint(v);
            int lo = __double2loint(v);
            hi = __shfl_xor_sync(0xffffffff, hi, off);
            lo = __shfl_xor_sync(0xffffffff, lo, off);
            v += __hiloint2double(hi, lo);
        }
        if (lane == 0) atomicAdd(out_sum_sq, v);
    }
}

__global__ void compute_adamw_inv_scale_k(
    const double* __restrict__ norm_sq_f64,
    float* __restrict__ global_norm_out,
    float* __restrict__ inv_scale_out,
    float loss_scale,
    float clip_grad,
    float dp_ws,
    int use_deferred_div
) {
    double sq = norm_sq_f64[0];
    float total_norm = sqrtf((float)sq);
    if (use_deferred_div) total_norm *= (1.0f / dp_ws);
    if (!isfinite(total_norm)) total_norm = INFINITY;
    global_norm_out[0] = total_norm / loss_scale;

    float combined = loss_scale;
    if (clip_grad > 0.0f) {
        float clip = ((total_norm / loss_scale) + 1e-6f) / clip_grad;
        if (clip < 1.0f) clip = 1.0f;
        combined = clip * loss_scale;
    }
    if (use_deferred_div) combined *= dp_ws;
    inv_scale_out[0] = (isfinite(total_norm) && combined > 0.0f) ? (1.0f / combined) : 0.0f;
}

__global__ void fused_adamw_fp32_master_bf16_io_k(
    float* __restrict__ fp32_p,
    float* __restrict__ fp32_m,
    float* __restrict__ fp32_v,
    const __nv_bfloat16* __restrict__ bf16_g,
    __nv_bfloat16* __restrict__ bf16_p_out,
    const float* __restrict__ inv_scale_ptr,
    float lr,
    float beta1,
    float beta2,
    float step_size,
    float inv_bc2_sqrt,
    float eps,
    float decay_factor,
    int64_t N
) {
    const float inv_s = inv_scale_ptr[0];
    if (inv_s == 0.0f) return;

    const float one_m_b1 = 1.0f - beta1;
    const float one_m_b2 = 1.0f - beta2;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const bool aligned = ((((uintptr_t)fp32_p | (uintptr_t)fp32_m | (uintptr_t)fp32_v) & 15) == 0)
                      && ((((uintptr_t)bf16_g | (uintptr_t)bf16_p_out) & 7) == 0);

    if (aligned) {
        const int64_t n4 = N >> 2;
        for (int64_t i4 = tid; i4 < n4; i4 += stride) {
            const int64_t off = i4 << 2;
            float4 pv = *reinterpret_cast<const float4*>(fp32_p + off);
            float4 mv = *reinterpret_cast<const float4*>(fp32_m + off);
            float4 vv = *reinterpret_cast<const float4*>(fp32_v + off);
            uint2 gv  = *reinterpret_cast<const uint2*>(bf16_g + off);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);

            float p_arr[4] = {pv.x, pv.y, pv.z, pv.w};
            float m_arr[4] = {mv.x, mv.y, mv.z, mv.w};
            float v_arr[4] = {vv.x, vv.y, vv.z, vv.w};
            __nv_bfloat16 out_b[4];

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float g = bf16_to_f32(gp[k]) * inv_s;
                float p = p_arr[k] * decay_factor;
                float m = beta1 * m_arr[k] + one_m_b1 * g;
                float v = beta2 * v_arr[k] + one_m_b2 * g * g;
                p -= step_size * (m / (sqrtf(v) * inv_bc2_sqrt + eps));
                p_arr[k] = p;
                m_arr[k] = m;
                v_arr[k] = v;
                out_b[k] = f32_to_bf16(p);
            }
            *reinterpret_cast<float4*>(fp32_p + off) = make_float4(p_arr[0], p_arr[1], p_arr[2], p_arr[3]);
            *reinterpret_cast<float4*>(fp32_m + off) = make_float4(m_arr[0], m_arr[1], m_arr[2], m_arr[3]);
            *reinterpret_cast<float4*>(fp32_v + off) = make_float4(v_arr[0], v_arr[1], v_arr[2], v_arr[3]);
            *reinterpret_cast<uint2*>(bf16_p_out + off) = *reinterpret_cast<const uint2*>(out_b);
        }
        for (int64_t idx = (n4 << 2) + tid; idx < N; idx += stride) {
            float g = bf16_to_f32(bf16_g[idx]) * inv_s;
            float p = fp32_p[idx] * decay_factor;
            float m = beta1 * fp32_m[idx] + one_m_b1 * g;
            float v = beta2 * fp32_v[idx] + one_m_b2 * g * g;
            p -= step_size * (m / (sqrtf(v) * inv_bc2_sqrt + eps));
            fp32_p[idx] = p;
            fp32_m[idx] = m;
            fp32_v[idx] = v;
            bf16_p_out[idx] = f32_to_bf16(p);
        }
    } else {
        for (int64_t idx = tid; idx < N; idx += stride) {
            float g = bf16_to_f32(bf16_g[idx]) * inv_s;
            float p = fp32_p[idx] * decay_factor;
            float m = beta1 * fp32_m[idx] + one_m_b1 * g;
            float v = beta2 * fp32_v[idx] + one_m_b2 * g * g;
            p -= step_size * (m / (sqrtf(v) * inv_bc2_sqrt + eps));
            fp32_p[idx] = p;
            fp32_m[idx] = m;
            fp32_v[idx] = v;
            bf16_p_out[idx] = f32_to_bf16(p);
        }
    }
}


// ============================================================================
// 10. v14 Zero-Spill Coalesced Dual Kernels & Zero-DtoH Flat int32 Scatter/DeepStack
// ============================================================================
template <bool BHSD>
__global__ void qwen_qkv_norm_rope_bwd_pers64_dual_k(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ dk,
    const __nv_bfloat16* __restrict__ dv,
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const __nv_bfloat16* __restrict__ cos_p,
    const __nv_bfloat16* __restrict__ sin_p,
    const float* __restrict__ rstd_p,
    uint8_t* __restrict__ dqkv_e5_row,
    uint8_t* __restrict__ dqkv_e5_t,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_dqkv,
    float* __restrict__ dqw_f32,
    float* __restrict__ dkw_f32,
    int B, int S, int h_q, int h_kv
) {
    __shared__ uint8_t smem_t[128][65];
    __shared__ float s_dw[128];

    const int T = B * S;
    const int h_qk = h_q + h_kv;
    const int h_tot = h_q + 2 * h_kv;
    const int h = blockIdx.x;
    const int sw = threadIdx.x >> 3;   // 0..31
    const int sub = threadIdx.x & 7;   // 0..7
    const int d0 = sub * 8;
    const int d1 = 64 + d0;
    const int col_base = h * 128;
    const float inv_s = inv_base[slot_dqkv];
    const bool aligned16_T = ((T & 15) == 0);

    if (threadIdx.x < 128) s_dw[threadIdx.x] = 0.0f;
    float lmax = 0.0f;

    if (h >= h_qk) {
        const int vh = h - h_qk;
        for (int t0 = blockIdx.y * 64; t0 < T; t0 += gridDim.y * 64) {
            #pragma unroll
            for (int half_r = 0; half_r < 64; half_r += 32) {
                const int t = t0 + half_r + sw;
                if (t < T) {
                    int64_t base_qkv = ((int64_t)t * h_tot + h) * 128;
                    int64_t base_v;
                    if constexpr (BHSD) {
                        int b = t / S;
                        int s = t - b * S;
                        base_v = (((int64_t)b * h_kv + vh) * S + s) * 128;
                    } else {
                        base_v = ((int64_t)t * h_kv + vh) * 128;
                    }
                    uint4 v0_u4 = *reinterpret_cast<const uint4*>(dv + base_v + d0);
                    uint4 v1_u4 = *reinterpret_cast<const uint4*>(dv + base_v + d1);
                    const __nv_bfloat16* p0 = reinterpret_cast<const __nv_bfloat16*>(&v0_u4);
                    const __nv_bfloat16* p1 = reinterpret_cast<const __nv_bfloat16*>(&v1_u4);
                    uint8_t e0[8], e1[8];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float f0 = bf16_to_f32(p0[j]);
                        float f1 = bf16_to_f32(p1[j]);
                        lmax = fmaxf(lmax, fmaxf(fabsf(f0), fabsf(f1)));
                        uint8_t q0 = f32_to_e5m2_raw(f0 * inv_s);
                        uint8_t q1 = f32_to_e5m2_raw(f1 * inv_s);
                        e0[j] = q0;
                        e1[j] = q1;
                        smem_t[d0 + j][half_r + sw] = q0;
                        smem_t[d1 + j][half_r + sw] = q1;
                    }
                    *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d0) = *reinterpret_cast<const uint64_t*>(e0);
                    *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d1) = *reinterpret_cast<const uint64_t*>(e1);
                }
            }
            __syncthreads();
            if (dqkv_e5_t != nullptr) {
                const int cl0 = threadIdx.x >> 2;
                const int seg = (threadIdx.x & 3) * 16;
                #pragma unroll
                for (int c_pass = 0; c_pass < 128; c_pass += 64) {
                    const int cl = c_pass + cl0;
                    const int64_t dst_off = (int64_t)(col_base + cl) * T + t0 + seg;
                    if (aligned16_T && (t0 + 64 <= T)) {
                        uint8_t b16[16];
                        #pragma unroll
                        for (int r = 0; r < 16; ++r) b16[r] = smem_t[cl][seg + r];
                        *reinterpret_cast<uint4*>(dqkv_e5_t + dst_off) = *reinterpret_cast<const uint4*>(b16);
                    } else {
                        for (int r = 0; r < 16; ++r) {
                            if (t0 + seg + r < T) dqkv_e5_t[dst_off + r] = smem_t[cl][seg + r];
                        }
                    }
                }
            }
            __syncthreads();
        }
    } else {
        const bool is_q = (h < h_q);
        const __nv_bfloat16* w_ptr = is_q ? qw : kw;
        uint4 w0_v = __ldg(reinterpret_cast<const uint4*>(w_ptr + d0));
        uint4 w1_v = __ldg(reinterpret_cast<const uint4*>(w_ptr + d1));
        const __nv_bfloat16* w0_p = reinterpret_cast<const __nv_bfloat16*>(&w0_v);
        const __nv_bfloat16* w1_p = reinterpret_cast<const __nv_bfloat16*>(&w1_v);
        float wf0[8], wf1[8];
        float dw0_acc[8] = {0}, dw1_acc[8] = {0};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            wf0[j] = bf16_to_f32(w0_p[j]);
            wf1[j] = bf16_to_f32(w1_p[j]);
        }
        const __nv_bfloat16* dy_src = is_q ? dq : dk;
        const int src_heads = is_q ? h_q : h_kv;
        const int src_h = is_q ? h : (h - h_q);

        for (int t0 = blockIdx.y * 64; t0 < T; t0 += gridDim.y * 64) {
            #pragma unroll
            for (int half_r = 0; half_r < 64; half_r += 32) {
                const int t = t0 + half_r + sw;
                if (t < T) {
                    int64_t base_qkv = ((int64_t)t * h_tot + h) * 128;
                    int64_t base_dy;
                    if constexpr (BHSD) {
                        int b = t / S;
                        int s = t - b * S;
                        base_dy = (((int64_t)b * src_heads + src_h) * S + s) * 128;
                    } else {
                        base_dy = ((int64_t)t * src_heads + src_h) * 128;
                    }
                    int64_t cs_off = (int64_t)t * 128;

                    uint4 dy0_v = *reinterpret_cast<const uint4*>(dy_src + base_dy + d0);
                    uint4 dy1_v = *reinterpret_cast<const uint4*>(dy_src + base_dy + d1);
                    uint4 x0_v  = *reinterpret_cast<const uint4*>(qkv + base_qkv + d0);
                    uint4 x1_v  = *reinterpret_cast<const uint4*>(qkv + base_qkv + d1);
                    uint4 c0_v  = __ldg(reinterpret_cast<const uint4*>(cos_p + cs_off + d0));
                    uint4 c1_v  = __ldg(reinterpret_cast<const uint4*>(cos_p + cs_off + d1));
                    uint4 s0_v  = __ldg(reinterpret_cast<const uint4*>(sin_p + cs_off + d0));
                    uint4 s1_v  = __ldg(reinterpret_cast<const uint4*>(sin_p + cs_off + d1));

                    const __nv_bfloat16* dy0_p = reinterpret_cast<const __nv_bfloat16*>(&dy0_v);
                    const __nv_bfloat16* dy1_p = reinterpret_cast<const __nv_bfloat16*>(&dy1_v);
                    const __nv_bfloat16* x0_p  = reinterpret_cast<const __nv_bfloat16*>(&x0_v);
                    const __nv_bfloat16* x1_p  = reinterpret_cast<const __nv_bfloat16*>(&x1_v);
                    const __nv_bfloat16* c0_p  = reinterpret_cast<const __nv_bfloat16*>(&c0_v);
                    const __nv_bfloat16* c1_p  = reinterpret_cast<const __nv_bfloat16*>(&c1_v);
                    const __nv_bfloat16* s0_p  = reinterpret_cast<const __nv_bfloat16*>(&s0_v);
                    const __nv_bfloat16* s1_p  = reinterpret_cast<const __nv_bfloat16*>(&s1_v);

                    const float rstd = rstd_p[(int64_t)t * h_qk + h];
                    float g0[8], g1[8], xh0[8], xh1[8];
                    float cdot = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float dy0 = bf16_to_f32(dy0_p[j]);
                        float dy1 = bf16_to_f32(dy1_p[j]);
                        float dnx0 = dy0 * bf16_to_f32(c0_p[j]) + dy1 * bf16_to_f32(s1_p[j]);
                        float dnx1 = dy1 * bf16_to_f32(c1_p[j]) - dy0 * bf16_to_f32(s0_p[j]);
                        xh0[j] = bf16_to_f32(x0_p[j]) * rstd;
                        xh1[j] = bf16_to_f32(x1_p[j]) * rstd;
                        dw0_acc[j] += dnx0 * xh0[j];
                        dw1_acc[j] += dnx1 * xh1[j];
                        g0[j] = dnx0 * wf0[j];
                        g1[j] = dnx1 * wf1[j];
                        cdot += g0[j] * xh0[j] + g1[j] * xh1[j];
                    }
                    cdot = subwarp8_reduce_sum(cdot) * (1.0f / 128.0f);

                    uint8_t e0[8], e1[8];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float dx0 = rstd * (g0[j] - xh0[j] * cdot);
                        float dx1 = rstd * (g1[j] - xh1[j] * cdot);
                        lmax = fmaxf(lmax, fmaxf(fabsf(dx0), fabsf(dx1)));
                        uint8_t q0 = f32_to_e5m2_raw(dx0 * inv_s);
                        uint8_t q1 = f32_to_e5m2_raw(dx1 * inv_s);
                        e0[j] = q0;
                        e1[j] = q1;
                        smem_t[d0 + j][half_r + sw] = q0;
                        smem_t[d1 + j][half_r + sw] = q1;
                    }
                    *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d0) = *reinterpret_cast<const uint64_t*>(e0);
                    *reinterpret_cast<uint64_t*>(dqkv_e5_row + base_qkv + d1) = *reinterpret_cast<const uint64_t*>(e1);
                }
            }
            __syncthreads();
            if (dqkv_e5_t != nullptr) {
                const int cl0 = threadIdx.x >> 2;
                const int seg = (threadIdx.x & 3) * 16;
                #pragma unroll
                for (int c_pass = 0; c_pass < 128; c_pass += 64) {
                    const int cl = c_pass + cl0;
                    const int64_t dst_off = (int64_t)(col_base + cl) * T + t0 + seg;
                    if (aligned16_T && (t0 + 64 <= T)) {
                        uint8_t b16[16];
                        #pragma unroll
                        for (int r = 0; r < 16; ++r) b16[r] = smem_t[cl][seg + r];
                        *reinterpret_cast<uint4*>(dqkv_e5_t + dst_off) = *reinterpret_cast<const uint4*>(b16);
                    } else {
                        for (int r = 0; r < 16; ++r) {
                            if (t0 + seg + r < T) dqkv_e5_t[dst_off + r] = smem_t[cl][seg + r];
                        }
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            atomicAdd(s_dw + d0 + j, dw0_acc[j]);
            atomicAdd(s_dw + d1 + j, dw1_acc[j]);
        }
        __syncthreads();
        if (threadIdx.x < 128) {
            atomicAdd((is_q ? dqw_f32 : dkw_f32) + threadIdx.x, s_dw[threadIdx.x]);
        }
    }

    lmax = warp_reduce_max(lmax);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) s_dw[wid] = lmax;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 8) ? s_dw[lane] : 0.0f;
        v = warp_reduce_max(v);
        if (lane == 0) atomic_max_pos_f32(next_amax + slot_dqkv, v);
    }
}

__global__ void vis_qkv_rope_bwd_pers_k(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ dk,
    const __nv_bfloat16* __restrict__ dv,
    const float* __restrict__ cos_f32,
    const float* __restrict__ sin_f32,
    __nv_bfloat16* __restrict__ dqkv,
    float* __restrict__ dqkv_bias_f32,
    int total_tokens,
    int seq_per_cam,
    int num_heads
) {
    __shared__ float s_db[64];
    if (threadIdx.x < 64) s_db[threadIdx.x] = 0.0f;
    __syncthreads();

    const int items_per_tok = 3 * num_heads;
    const int rem = blockIdx.x;        // 0..47
    const int qkv_idx = rem / num_heads;
    const int h = rem - qkv_idx * num_heads;
    const int sw = threadIdx.x >> 2;   // 0..63
    const int sub = threadIdx.x & 3;   // 0..3
    const int d0 = sub * 8;
    const int d1 = 32 + d0;

    float db0[8] = {0}, db1[8] = {0};

    if (qkv_idx == 2) {
        for (int tok = blockIdx.y * 64 + sw; tok < total_tokens; tok += gridDim.y * 64) {
            int64_t out_off = ((int64_t)tok * items_per_tok + rem) * 64;
            int64_t in_off = ((int64_t)tok * num_heads + h) * 64;
            uint4 v0 = *reinterpret_cast<const uint4*>(dv + in_off + d0);
            uint4 v1 = *reinterpret_cast<const uint4*>(dv + in_off + d1);
            *reinterpret_cast<uint4*>(dqkv + out_off + d0) = v0;
            *reinterpret_cast<uint4*>(dqkv + out_off + d1) = v1;
            const __nv_bfloat16* p0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
            const __nv_bfloat16* p1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                db0[k] += bf16_to_f32(p0[k]);
                db1[k] += bf16_to_f32(p1[k]);
            }
        }
    } else {
        const __nv_bfloat16* src = (qkv_idx == 0) ? dq : dk;
        for (int tok = blockIdx.y * 64 + sw; tok < total_tokens; tok += gridDim.y * 64) {
            int64_t out_off = ((int64_t)tok * items_per_tok + rem) * 64;
            int64_t in_off = ((int64_t)tok * num_heads + h) * 64;
            uint4 dy0_v = *reinterpret_cast<const uint4*>(src + in_off + d0);
            uint4 dy1_v = *reinterpret_cast<const uint4*>(src + in_off + d1);
            const __nv_bfloat16* dy0_p = reinterpret_cast<const __nv_bfloat16*>(&dy0_v);
            const __nv_bfloat16* dy1_p = reinterpret_cast<const __nv_bfloat16*>(&dy1_v);

            int cs_base = (tok % seq_per_cam) * 64;
            float4 c0_a = __ldg(reinterpret_cast<const float4*>(cos_f32 + cs_base + d0));
            float4 c0_b = __ldg(reinterpret_cast<const float4*>(cos_f32 + cs_base + d0 + 4));
            float4 c1_a = __ldg(reinterpret_cast<const float4*>(cos_f32 + cs_base + d1));
            float4 c1_b = __ldg(reinterpret_cast<const float4*>(cos_f32 + cs_base + d1 + 4));
            float4 s0_a = __ldg(reinterpret_cast<const float4*>(sin_f32 + cs_base + d0));
            float4 s0_b = __ldg(reinterpret_cast<const float4*>(sin_f32 + cs_base + d0 + 4));
            float4 s1_a = __ldg(reinterpret_cast<const float4*>(sin_f32 + cs_base + d1));
            float4 s1_b = __ldg(reinterpret_cast<const float4*>(sin_f32 + cs_base + d1 + 4));

            const float c0[8] = {c0_a.x, c0_a.y, c0_a.z, c0_a.w, c0_b.x, c0_b.y, c0_b.z, c0_b.w};
            const float c1[8] = {c1_a.x, c1_a.y, c1_a.z, c1_a.w, c1_b.x, c1_b.y, c1_b.z, c1_b.w};
            const float s0[8] = {s0_a.x, s0_a.y, s0_a.z, s0_a.w, s0_b.x, s0_b.y, s0_b.z, s0_b.w};
            const float s1[8] = {s1_a.x, s1_a.y, s1_a.z, s1_a.w, s1_b.x, s1_b.y, s1_b.z, s1_b.w};

            __nv_bfloat16 dx0_b[8], dx1_b[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dy0 = bf16_to_f32(dy0_p[k]);
                float dy1 = bf16_to_f32(dy1_p[k]);
                dx0_b[k] = f32_to_bf16(dy0 * c0[k] + dy1 * s1[k]);
                dx1_b[k] = f32_to_bf16(dy1 * c1[k] - dy0 * s0[k]);
                db0[k] += bf16_to_f32(dx0_b[k]);
                db1[k] += bf16_to_f32(dx1_b[k]);
            }
            *reinterpret_cast<uint4*>(dqkv + out_off + d0) = *reinterpret_cast<const uint4*>(dx0_b);
            *reinterpret_cast<uint4*>(dqkv + out_off + d1) = *reinterpret_cast<const uint4*>(dx1_b);
        }
    }

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        atomicAdd(s_db + d0 + k, db0[k]);
        atomicAdd(s_db + d1 + k, db1[k]);
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        atomicAdd(dqkv_bias_f32 + rem * 64 + threadIdx.x, s_db[threadIdx.x]);
    }
}

template <bool HAS_DRES, bool HAS_DRES_SUM>
__global__ void ln_bwd_1024_w4_l1_k(
    const __nv_bfloat16* __restrict__ dy,
    const __nv_bfloat16* __restrict__ d_res_in,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    const float* __restrict__ mean_p,
    const float* __restrict__ rstd_p,
    __nv_bfloat16* __restrict__ dx,
    float* __restrict__ dgamma_f32,
    float* __restrict__ dbeta_f32,
    float* __restrict__ dres_sum_f32,
    int M
) {
    __shared__ float s_red[4][128];
    const int wid = threadIdx.x >> 5;   // 0..3
    const int lane = threadIdx.x & 31;  // 0..31

    float dg_acc[4][8] = {{0}};
    float db_acc[4][8] = {{0}};
    float dr_acc[4][8] = {{0}};

    const int total_warps = gridDim.x * 4;
    for (int row = blockIdx.x * 4 + wid; row < M; row += total_warps) {
        const int64_t row_off = (int64_t)row * 1024;
        const float mean = mean_p[row];
        const float rstd = rstd_p[row];

        float s_g = 0.0f, s_gxh = 0.0f;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int col = c * 256 + lane * 8;
            uint4 xv = *reinterpret_cast<const uint4*>(x + row_off + col);
            uint4 dyv = *reinterpret_cast<const uint4*>(dy + row_off + col);
            uint4 gv = __ldg(reinterpret_cast<const uint4*>(gamma + col));
            const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
            const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dyv);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dyk = bf16_to_f32(dyp[k]);
                float xhk = (bf16_to_f32(xp[k]) - mean) * rstd;
                dg_acc[c][k] += dyk * xhk;
                db_acc[c][k] += dyk;
                float gk = dyk * bf16_to_f32(gp[k]);
                s_g += gk;
                s_gxh += gk * xhk;
            }
        }
        const float c1 = warp_reduce_sum(s_g) * (1.0f / 1024.0f);
        const float c2 = warp_reduce_sum(s_gxh) * (1.0f / 1024.0f);

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int col = c * 256 + lane * 8;
            uint4 xv = *reinterpret_cast<const uint4*>(x + row_off + col);
            uint4 dyv = *reinterpret_cast<const uint4*>(dy + row_off + col);
            uint4 gv = __ldg(reinterpret_cast<const uint4*>(gamma + col));
            uint4 drv;
            if constexpr (HAS_DRES) drv = *reinterpret_cast<const uint4*>(d_res_in + row_off + col);
            const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
            const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dyv);
            const __nv_bfloat16* gp = reinterpret_cast<const __nv_bfloat16*>(&gv);
            const __nv_bfloat16* drp = reinterpret_cast<const __nv_bfloat16*>(&drv);
            __nv_bfloat16 out_b[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float dyk = bf16_to_f32(dyp[k]);
                float xhk = (bf16_to_f32(xp[k]) - mean) * rstd;
                float gk = dyk * bf16_to_f32(gp[k]);
                float dxf = rstd * (gk - c1 - xhk * c2);
                if constexpr (HAS_DRES) {
                    float drf = bf16_to_f32(drp[k]);
                    if constexpr (HAS_DRES_SUM) dr_acc[c][k] += drf;
                    dxf = drf + bf16_to_f32(f32_to_bf16(dxf));
                }
                out_b[k] = f32_to_bf16(dxf);
            }
            *reinterpret_cast<uint4*>(dx + row_off + col) = *reinterpret_cast<const uint4*>(out_b);
        }
    }

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int base_col = c * 256 + half * 128;
            if ((lane * 8) >= half * 128 && (lane * 8) < (half + 1) * 128) {
                int loc = lane * 8 - half * 128;
                #pragma unroll
                for (int k = 0; k < 8; ++k) s_red[wid][loc + k] = dg_acc[c][k];
            }
            __syncthreads();
            atomicAdd(dgamma_f32 + base_col + threadIdx.x, s_red[0][threadIdx.x] + s_red[1][threadIdx.x] + s_red[2][threadIdx.x] + s_red[3][threadIdx.x]);
            __syncthreads();

            if ((lane * 8) >= half * 128 && (lane * 8) < (half + 1) * 128) {
                int loc = lane * 8 - half * 128;
                #pragma unroll
                for (int k = 0; k < 8; ++k) s_red[wid][loc + k] = db_acc[c][k];
            }
            __syncthreads();
            atomicAdd(dbeta_f32 + base_col + threadIdx.x, s_red[0][threadIdx.x] + s_red[1][threadIdx.x] + s_red[2][threadIdx.x] + s_red[3][threadIdx.x]);
            __syncthreads();

            if constexpr (HAS_DRES && HAS_DRES_SUM) {
                if ((lane * 8) >= half * 128 && (lane * 8) < (half + 1) * 128) {
                    int loc = lane * 8 - half * 128;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) s_red[wid][loc + k] = dr_acc[c][k];
                }
                __syncthreads();
                atomicAdd(dres_sum_f32 + base_col + threadIdx.x, s_red[0][threadIdx.x] + s_red[1][threadIdx.x] + s_red[2][threadIdx.x] + s_red[3][threadIdx.x]);
                __syncthreads();
            }
        }
    }
}

template <bool BHSD>
__global__ void mmdit_joint_qkv_48_bwd_pers64_e5_k(
    const __nv_bfloat16* __restrict__ dj_q,
    const __nv_bfloat16* __restrict__ dj_k,
    const __nv_bfloat16* __restrict__ dj_v,
    const __nv_bfloat16* __restrict__ act_qkv,
    const __nv_bfloat16* __restrict__ txt_qkv,
    const __nv_bfloat16* __restrict__ nq_w,
    const __nv_bfloat16* __restrict__ nk_w,
    const __nv_bfloat16* __restrict__ naq_w,
    const __nv_bfloat16* __restrict__ nak_w,
    const float* __restrict__ act_cos,
    const float* __restrict__ act_sin,
    const float* __restrict__ txt_cos,
    const float* __restrict__ txt_sin,
    const float* __restrict__ rstd_p,
    uint8_t* __restrict__ da_row, uint8_t* __restrict__ da_t, float* __restrict__ da_dbias_f32,
    uint8_t* __restrict__ dt_row, uint8_t* __restrict__ dt_t, float* __restrict__ dt_dbias_f32,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_act, int slot_txt,
    float* __restrict__ d_nq_f32,
    float* __restrict__ d_nk_f32,
    float* __restrict__ d_naq_f32,
    float* __restrict__ d_nak_f32,
    int B, int S_act, int S_txt, int H
) {
    __shared__ uint8_t smem_t[48][65];
    __shared__ float s_acc[4][48];

    if (threadIdx.x < 192) (&s_acc[0][0])[threadIdx.x] = 0.0f;

    const int rem = blockIdx.x;  // 0 .. 3*H - 1 (96)
    const int qkv_idx = rem / H;
    const int h = rem - qkv_idx * H;
    const int sw = threadIdx.x >> 3;   // 0..31
    const int sub = threadIdx.x & 7;   // 0..7 (sub < 6 active -> 48 cols)
    const int d = sub * 8;
    const int col_base = rem * 48;
    const int S_tot = S_act + S_txt;
    const int M_act = B * S_act;
    const int M_txt = B * S_txt;
    const float inv_a = inv_base[slot_act];
    const float inv_t = inv_base[slot_txt];

    float lmax_a = 0.0f, lmax_t = 0.0f;
    float dba_acc[8] = {0}, dbt_acc[8] = {0};
    float dwa_acc[8] = {0}, dwt_acc[8] = {0};

    float wf_a[8] = {0}, wf_t[8] = {0};
    if (qkv_idx < 2 && sub < 6) {
        uint4 wa_v = __ldg(reinterpret_cast<const uint4*>(((qkv_idx == 0) ? nq_w : nk_w) + d));
        uint4 wt_v = __ldg(reinterpret_cast<const uint4*>(((qkv_idx == 0) ? naq_w : nak_w) + d));
        const __nv_bfloat16* wap = reinterpret_cast<const __nv_bfloat16*>(&wa_v);
        const __nv_bfloat16* wtp = reinterpret_cast<const __nv_bfloat16*>(&wt_v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            wf_a[j] = bf16_to_f32(wap[j]);
            wf_t[j] = bf16_to_f32(wtp[j]);
        }
    }
    const __nv_bfloat16* dj_src = (qkv_idx == 0) ? dj_q : ((qkv_idx == 1) ? dj_k : dj_v);

    // Phase A: Action tokens [0, M_act)
    const bool aligned16_Ma = ((M_act & 15) == 0);
    for (int m0 = blockIdx.y * 64; m0 < M_act; m0 += gridDim.y * 64) {
        #pragma unroll
        for (int half_r = 0; half_r < 64; half_r += 32) {
            const int m = m0 + half_r + sw;
            if (m < M_act) {
                int b = m / S_act;
                int s = m - b * S_act;
                float cdot = 0.0f;
                float g[8], xh[8];
                float rstd = 0.0f;
                int64_t qkv_off = (int64_t)m * 4608 + col_base + d;
                uint4 dy_v;
                if (sub < 6) {
                    int64_t dj_off = BHSD
                        ? ((((int64_t)b * H + h) * S_tot + s) * 48 + d)
                        : ((((int64_t)b * S_tot + s) * H + h) * 48 + d);
                    dy_v = *reinterpret_cast<const uint4*>(dj_src + dj_off);
                    if (qkv_idx < 2) {
                        const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dy_v);
                        float4 ca = __ldg(reinterpret_cast<const float4*>(act_cos + s * 48 + d));
                        float4 cb = __ldg(reinterpret_cast<const float4*>(act_cos + s * 48 + d + 4));
                        float4 sa = __ldg(reinterpret_cast<const float4*>(act_sin + s * 48 + d));
                        float4 sb = __ldg(reinterpret_cast<const float4*>(act_sin + s * 48 + d + 4));
                        const float cc[4] = {ca.x, ca.z, cb.x, cb.z};
                        const float ss[4] = {sa.x, sa.z, sb.x, sb.z};
                        uint4 xv = *reinterpret_cast<const uint4*>(act_qkv + qkv_off);
                        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
                        int tok_joint = b * S_tot + s;
                        rstd = rstd_p[((int64_t)tok_joint * 2 + qkv_idx) * H + h];
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            float dy0 = bf16_to_f32(dyp[2 * p]);
                            float dy1 = bf16_to_f32(dyp[2 * p + 1]);
                            float dn0 = dy0 * cc[p] + dy1 * ss[p];
                            float dn1 = dy1 * cc[p] - dy0 * ss[p];
                            xh[2 * p]     = bf16_to_f32(xp[2 * p]) * rstd;
                            xh[2 * p + 1] = bf16_to_f32(xp[2 * p + 1]) * rstd;
                            dwa_acc[2 * p]     += dn0 * xh[2 * p];
                            dwa_acc[2 * p + 1] += dn1 * xh[2 * p + 1];
                            g[2 * p]     = dn0 * wf_a[2 * p];
                            g[2 * p + 1] = dn1 * wf_a[2 * p + 1];
                            cdot += g[2 * p] * xh[2 * p] + g[2 * p + 1] * xh[2 * p + 1];
                        }
                    }
                }
                if (qkv_idx < 2) {
                    cdot = subwarp8_reduce_sum(cdot) * (1.0f / 48.0f);
                }
                if (sub < 6) {
                    const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dy_v);
                    uint8_t eq[8];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float fv = (qkv_idx == 2)
                            ? bf16_to_f32(dyp[j])
                            : bf16_to_f32(f32_to_bf16(rstd * (g[j] - xh[j] * cdot)));
                        dba_acc[j] += fv;
                        lmax_a = fmaxf(lmax_a, fabsf(fv));
                        uint8_t q = f32_to_e5m2_raw(fv * inv_a);
                        eq[j] = q;
                        smem_t[d + j][half_r + sw] = q;
                    }
                    *reinterpret_cast<uint64_t*>(da_row + qkv_off) = *reinterpret_cast<const uint64_t*>(eq);
                }
            }
        }
        __syncthreads();
        if (da_t != nullptr && threadIdx.x < 192) {
            const int cl = threadIdx.x >> 2;
            const int seg = (threadIdx.x & 3) * 16;
            const int64_t dst_off = (int64_t)(col_base + cl) * M_act + m0 + seg;
            if (aligned16_Ma && (m0 + 64 <= M_act)) {
                uint8_t b16[16];
                #pragma unroll
                for (int r = 0; r < 16; ++r) b16[r] = smem_t[cl][seg + r];
                *reinterpret_cast<uint4*>(da_t + dst_off) = *reinterpret_cast<const uint4*>(b16);
            } else {
                for (int r = 0; r < 16; ++r) {
                    if (m0 + seg + r < M_act) da_t[dst_off + r] = smem_t[cl][seg + r];
                }
            }
        }
        __syncthreads();
    }

    // Phase B: Text tokens [0, M_txt)
    const bool aligned16_Mt = ((M_txt & 15) == 0);
    for (int m0 = blockIdx.y * 64; m0 < M_txt; m0 += gridDim.y * 64) {
        #pragma unroll
        for (int half_r = 0; half_r < 64; half_r += 32) {
            const int m = m0 + half_r + sw;
            if (m < M_txt) {
                int b = m / S_txt;
                int s = m - b * S_txt;
                int s_joint = S_act + s;
                float cdot = 0.0f;
                float g[8], xh[8];
                float rstd = 0.0f;
                int64_t qkv_off = (int64_t)m * 4608 + col_base + d;
                uint4 dy_v;
                if (sub < 6) {
                    int64_t dj_off = BHSD
                        ? ((((int64_t)b * H + h) * S_tot + s_joint) * 48 + d)
                        : ((((int64_t)b * S_tot + s_joint) * H + h) * 48 + d);
                    dy_v = *reinterpret_cast<const uint4*>(dj_src + dj_off);
                    if (qkv_idx < 2) {
                        const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dy_v);
                        float4 ca = __ldg(reinterpret_cast<const float4*>(txt_cos + s * 48 + d));
                        float4 cb = __ldg(reinterpret_cast<const float4*>(txt_cos + s * 48 + d + 4));
                        float4 sa = __ldg(reinterpret_cast<const float4*>(txt_sin + s * 48 + d));
                        float4 sb = __ldg(reinterpret_cast<const float4*>(txt_sin + s * 48 + d + 4));
                        const float cc[4] = {ca.x, ca.z, cb.x, cb.z};
                        const float ss[4] = {sa.x, sa.z, sb.x, sb.z};
                        uint4 xv = *reinterpret_cast<const uint4*>(txt_qkv + qkv_off);
                        const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
                        int tok_joint = b * S_tot + s_joint;
                        rstd = rstd_p[((int64_t)tok_joint * 2 + qkv_idx) * H + h];
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            float dy0 = bf16_to_f32(dyp[2 * p]);
                            float dy1 = bf16_to_f32(dyp[2 * p + 1]);
                            float dn0 = dy0 * cc[p] + dy1 * ss[p];
                            float dn1 = dy1 * cc[p] - dy0 * ss[p];
                            xh[2 * p]     = bf16_to_f32(xp[2 * p]) * rstd;
                            xh[2 * p + 1] = bf16_to_f32(xp[2 * p + 1]) * rstd;
                            dwt_acc[2 * p]     += dn0 * xh[2 * p];
                            dwt_acc[2 * p + 1] += dn1 * xh[2 * p + 1];
                            g[2 * p]     = dn0 * wf_t[2 * p];
                            g[2 * p + 1] = dn1 * wf_t[2 * p + 1];
                            cdot += g[2 * p] * xh[2 * p] + g[2 * p + 1] * xh[2 * p + 1];
                        }
                    }
                }
                if (qkv_idx < 2) {
                    cdot = subwarp8_reduce_sum(cdot) * (1.0f / 48.0f);
                }
                if (sub < 6) {
                    const __nv_bfloat16* dyp = reinterpret_cast<const __nv_bfloat16*>(&dy_v);
                    uint8_t eq[8];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float fv = (qkv_idx == 2)
                            ? bf16_to_f32(dyp[j])
                            : bf16_to_f32(f32_to_bf16(rstd * (g[j] - xh[j] * cdot)));
                        dbt_acc[j] += fv;
                        lmax_t = fmaxf(lmax_t, fabsf(fv));
                        uint8_t q = f32_to_e5m2_raw(fv * inv_t);
                        eq[j] = q;
                        smem_t[d + j][half_r + sw] = q;
                    }
                    *reinterpret_cast<uint64_t*>(dt_row + qkv_off) = *reinterpret_cast<const uint64_t*>(eq);
                }
            }
        }
        __syncthreads();
        if (dt_t != nullptr && threadIdx.x < 192) {
            const int cl = threadIdx.x >> 2;
            const int seg = (threadIdx.x & 3) * 16;
            const int64_t dst_off = (int64_t)(col_base + cl) * M_txt + m0 + seg;
            if (aligned16_Mt && (m0 + 64 <= M_txt)) {
                uint8_t b16[16];
                #pragma unroll
                for (int r = 0; r < 16; ++r) b16[r] = smem_t[cl][seg + r];
                *reinterpret_cast<uint4*>(dt_t + dst_off) = *reinterpret_cast<const uint4*>(b16);
            } else {
                for (int r = 0; r < 16; ++r) {
                    if (m0 + seg + r < M_txt) dt_t[dst_off + r] = smem_t[cl][seg + r];
                }
            }
        }
        __syncthreads();
    }

    if (sub < 6) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            atomicAdd(&s_acc[0][d + j], dba_acc[j]);
            atomicAdd(&s_acc[1][d + j], dbt_acc[j]);
            if (qkv_idx < 2) {
                atomicAdd(&s_acc[2][d + j], dwa_acc[j]);
                atomicAdd(&s_acc[3][d + j], dwt_acc[j]);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < 48) {
        atomicAdd(da_dbias_f32 + col_base + threadIdx.x, s_acc[0][threadIdx.x]);
        atomicAdd(dt_dbias_f32 + col_base + threadIdx.x, s_acc[1][threadIdx.x]);
        if (qkv_idx < 2) {
            atomicAdd(((qkv_idx == 0) ? d_nq_f32 : d_nk_f32) + threadIdx.x, s_acc[2][threadIdx.x]);
            atomicAdd(((qkv_idx == 0) ? d_naq_f32 : d_nak_f32) + threadIdx.x, s_acc[3][threadIdx.x]);
        }
    }

    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    lmax_a = warp_reduce_max(lmax_a);
    lmax_t = warp_reduce_max(lmax_t);
    __syncthreads();
    if (lane == 0) { s_acc[0][wid] = lmax_a; s_acc[0][8 + wid] = lmax_t; }
    __syncthreads();
    if (wid == 0) {
        float va = (lane < 8) ? s_acc[0][lane] : 0.0f;
        float vt = (lane < 8) ? s_acc[0][8 + lane] : 0.0f;
        va = warp_reduce_max(va);
        vt = warp_reduce_max(vt);
        if (lane == 0) {
            atomic_max_pos_f32(next_amax + slot_act, va);
            atomic_max_pos_f32(next_amax + slot_txt, vt);
        }
    }
}

__global__ void scatter_rows_2560_bf16_k(
    __nv_bfloat16* __restrict__ dst_2d,
    const __nv_bfloat16* __restrict__ src_2d,
    const int32_t* __restrict__ idx_i32,
    int N_rows
) {
    const int r = blockIdx.x;
    if (r >= N_rows) return;
    const int dst_row = idx_i32[r];
    const int col = threadIdx.x * 8;
    *reinterpret_cast<uint4*>(dst_2d + (int64_t)dst_row * 2560 + col) =
        *reinterpret_cast<const uint4*>(src_2d + (int64_t)r * 2560 + col);
}

__global__ void gather_and_zero_rows_2560_bf16_k(
    __nv_bfloat16* __restrict__ d_main_2d,
    __nv_bfloat16* __restrict__ d_src_out,
    const int32_t* __restrict__ idx_i32,
    int N_rows,
    int zero_main
) {
    const int r = blockIdx.x;
    if (r >= N_rows) return;
    const int src_row = idx_i32[r];
    const int col = threadIdx.x * 8;
    int64_t m_off = (int64_t)src_row * 2560 + col;
    uint4 v = *reinterpret_cast<const uint4*>(d_main_2d + m_off);
    if (d_src_out != nullptr) {
        *reinterpret_cast<uint4*>(d_src_out + (int64_t)r * 2560 + col) = v;
    }
    if (zero_main) {
        *reinterpret_cast<uint4*>(d_main_2d + m_off) = make_uint4(0, 0, 0, 0);
    }
}

__global__ void deepstack_add_2560_bf16_k(
    __nv_bfloat16* __restrict__ dst_2d,
    const __nv_bfloat16* __restrict__ vis_2d,
    const int32_t* __restrict__ idx_i32,
    int N_rows
) {
    const int r = blockIdx.x;
    if (r >= N_rows) return;
    const int dst_row = idx_i32[r];
    const int col = threadIdx.x * 8;
    int64_t d_off = (int64_t)dst_row * 2560 + col;
    uint4 dv = *reinterpret_cast<const uint4*>(dst_2d + d_off);
    uint4 vv = *reinterpret_cast<const uint4*>(vis_2d + (int64_t)r * 2560 + col);
    const __nv_bfloat16* dp = reinterpret_cast<const __nv_bfloat16*>(&dv);
    const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(&vv);
    __nv_bfloat16 out[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) out[k] = f32_to_bf16(bf16_to_f32(dp[k]) + bf16_to_f32(vp[k]));
    *reinterpret_cast<uint4*>(dst_2d + d_off) = *reinterpret_cast<const uint4*>(out);
}

template <bool SAVE_RSTD>
__global__ __launch_bounds__(512, 1) void rmsnorm_quant_2560_pers64_dual_k(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    uint8_t* __restrict__ y_e4_row,
    uint8_t* __restrict__ y_e4_col,
    float* __restrict__ rstd_out,
    const float* __restrict__ inv_base,
    int32_t* __restrict__ next_amax,
    int slot_id,
    int M,
    float eps
) {
    extern __shared__ uint8_t smem_raw[];
    uint8_t (*smem_t)[65] = reinterpret_cast<uint8_t(*)[65]>(smem_raw);
    float* s_amax = reinterpret_cast<float*>(smem_raw + 2560 * 65);

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const float inv_s = inv_base[slot_id];
    const bool aligned16_M = ((M & 15) == 0);
    float lmax = 0.0f;

    for (int m0 = blockIdx.x * 64; m0 < M; m0 += gridDim.x * 64) {
        #pragma unroll
        for (int sub_r = 0; sub_r < 64; sub_r += 16) {
            const int row = m0 + sub_r + wid;
            if (row < M) {
                const int64_t off = (int64_t)row * 2560;
                float sq = 0.0f;
                #pragma unroll
                for (int c = 0; c < 10; ++c) {
                    const int col = c * 256 + lane * 8;
                    uint4 xv = *reinterpret_cast<const uint4*>(x + off + col);
                    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        float vf = bf16_to_f32(xp[k]);
                        sq += vf * vf;
                    }
                }
                sq = warp_reduce_sum(sq);
                const float rstd = rsqrtf(sq * (1.0f / 2560.0f) + eps);
                if constexpr (SAVE_RSTD) {
                    if (lane == 0) rstd_out[row] = rstd;
                }

                #pragma unroll
                for (int c = 0; c < 10; ++c) {
                    const int col = c * 256 + lane * 8;
                    uint4 xv = *reinterpret_cast<const uint4*>(x + off + col);
                    uint4 wv = __ldg(reinterpret_cast<const uint4*>(w + col));
                    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(&xv);
                    const __nv_bfloat16* wp = reinterpret_cast<const __nv_bfloat16*>(&wv);
                    uint8_t q8[8];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        float nf = bf16_to_f32(xp[k]) * rstd * bf16_to_f32(wp[k]);
                        lmax = fmaxf(lmax, fabsf(nf));
                        uint8_t q = f32_to_e4m3_raw(nf * inv_s);
                        q8[k] = q;
                        smem_t[col + k][sub_r + wid] = q;
                    }
                    *reinterpret_cast<uint64_t*>(y_e4_row + off + col) = *reinterpret_cast<const uint64_t*>(q8);
                }
            }
        }
        __syncthreads();
        const int cl0 = tid >> 2;
        const int seg = (tid & 3) * 16;
        #pragma unroll
        for (int cp = 0; cp < 2560; cp += 128) {
            const int cl = cp + cl0;
            const int64_t dst_off = (int64_t)cl * M + m0 + seg;
            if (aligned16_M && (m0 + 64 <= M)) {
                uint8_t b16[16];
                #pragma unroll
                for (int r = 0; r < 16; ++r) b16[r] = smem_t[cl][seg + r];
                *reinterpret_cast<uint4*>(y_e4_col + dst_off) = *reinterpret_cast<const uint4*>(b16);
            } else {
                for (int r = 0; r < 16; ++r) {
                    if (m0 + seg + r < M) y_e4_col[dst_off + r] = smem_t[cl][seg + r];
                }
            }
        }
        __syncthreads();
    }

    lmax = warp_reduce_max(lmax);
    if (lane == 0) s_amax[wid] = lmax;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 16) ? s_amax[lane] : 0.0f;
        v = warp_reduce_max(v);
        if (lane == 0) atomic_max_pos_f32(next_amax + slot_id, v);
    }
}

}  // namespace

extern "C" {

void launch_finalize_delayed_scales(void* next_amax, const void* fp8_max, void* curr_scale, void* curr_inv_scale, int n, cudaStream_t stream) {
    finalize_delayed_scales_k<<<(n + 255) / 256, 256, 0, stream>>>(
        (int32_t*)next_amax, (const float*)fp8_max, (float*)curr_scale, (float*)curr_inv_scale, n);
}

void launch_sum_rows_bf16(const void* x, void* dbias_bf16, void* dbias_f32_ws, int M, int N, cudaStream_t stream) {
    cudaMemsetAsync(dbias_f32_ws, 0, (size_t)N * sizeof(float), stream);
    if (((N & 7) == 0) && (((uintptr_t)x & 15) == 0)) {
        int grid_x = (N + 63) / 64;
        int grid_y = min(max(1, M / 64), 32);
        dim3 grid(grid_x, grid_y);
        sum_rows_bf16_vec8_k<<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, (float*)dbias_f32_ws, M, N);
    } else {
        dim3 g_sum((N + 255) / 256, min(M, 128));
        sum_rows_bf16_f32_k<<<g_sum, 256, 0, stream>>>((const __nv_bfloat16*)x, (float*)dbias_f32_ws, M, N);
    }
    cast_f32_to_bf16_k<<<(N + 255) / 256, 256, 0, stream>>>((const float*)dbias_f32_ws, (__nv_bfloat16*)dbias_bf16, N);
}

void launch_transpose_fp8(const void* x, void* y_t, int M, int N, cudaStream_t stream) {
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    if (((M & 3) == 0) && ((N & 3) == 0)) {
        transpose_fp8_64x64_vec4_k<<<grid, 256, 0, stream>>>((const uint8_t*)x, (uint8_t*)y_t, M, N);
    } else {
        transpose_fp8_64x64_scalar_k<<<grid, 256, 0, stream>>>((const uint8_t*)x, (uint8_t*)y_t, M, N);
    }
}

void launch_quant_1d_bf16(const void* x, void* y, const void* curr_scale, const void* curr_inv, void* next_amax,
                          int slot_id, int64_t numel, int is_e5m2, int warmed_up, cudaStream_t stream) {
    float fmax = is_e5m2 ? 57344.0f : 448.0f;
    int blocks = (int)min((int64_t)2112, (numel + 2047) / 2048);
    if (!warmed_up) {
        fast_amax_bf16_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)x, (int32_t*)next_amax, slot_id, numel);
        calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, fmax);
        if (is_e5m2) quant_1d_bf16_k<true, false><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y, (const float*)curr_inv, (int32_t*)next_amax, slot_id, numel);
        else         quant_1d_bf16_k<false, false><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y, (const float*)curr_inv, (int32_t*)next_amax, slot_id, numel);
    } else {
        if (is_e5m2) quant_1d_bf16_k<true, true><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y, (const float*)curr_inv, (int32_t*)next_amax, slot_id, numel);
        else         quant_1d_bf16_k<false, true><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y, (const float*)curr_inv, (int32_t*)next_amax, slot_id, numel);
    }
}

void launch_quant_bhsd_to_bshd(const void* x_bhsd, void* y_row, void* y_col,
                               const void* curr_scale, const void* curr_inv, void* next_amax,
                               int slot_id, int B, int H, int S, int dual, int warmed_up, cudaStream_t stream) {
    int M = B * S;
    int N = H * 128;
    int64_t numel = (int64_t)M * N;
    int blocks = (int)min((int64_t)2112, (numel + 2047) / 2048);
    if (!warmed_up) {
        fast_amax_bf16_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)x_bhsd, (int32_t*)next_amax, slot_id, numel);
        calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 448.0f);
    }
    if (dual && ((M & 3) == 0)) {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        if (!warmed_up) quant_bhsd_to_bshd_dual_64x64_prmt_k<false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x_bhsd, (uint8_t*)y_row, (uint8_t*)y_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, H, S);
        else            quant_bhsd_to_bshd_dual_64x64_prmt_k<true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x_bhsd, (uint8_t*)y_row, (uint8_t*)y_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, H, S);
        return;
    }
    if (!warmed_up) quant_bhsd_to_bshd_1d_k<false><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)x_bhsd, (uint8_t*)y_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, H, S);
    else            quant_bhsd_to_bshd_1d_k<true><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)x_bhsd, (uint8_t*)y_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, H, S);
    if (dual) {
        launch_transpose_fp8(y_row, y_col, M, N, stream);
    }
}

void launch_quant_dual_bf16(const void* x, void* y_row, void* y_col, void* dbias_bf16, void* dbias_f32_ws,
                            const void* curr_scale, const void* curr_inv, void* next_amax,
                            int slot_id, int M, int N, int is_e5m2, int warmed_up, cudaStream_t stream) {
    bool has_db = (dbias_bf16 != nullptr);
    if (has_db) {
        cudaMemsetAsync(dbias_f32_ws, 0, (size_t)N * sizeof(float), stream);
    }
    if (((M & 3) == 0) && ((N & 3) == 0)) {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        if (!warmed_up) {
            int64_t numel = (int64_t)M * N;
            int blocks = (int)min((int64_t)2112, (numel + 2047) / 2048);
            fast_amax_bf16_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)x, (int32_t*)next_amax, slot_id, numel);
            calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, is_e5m2 ? 57344.0f : 448.0f);
            if (is_e5m2) {
                if (has_db) quant_dual_64x64_prmt_k<true, false, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y_row, (uint8_t*)y_col, (float*)dbias_f32_ws, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
                else        quant_dual_64x64_prmt_k<true, false, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y_row, (uint8_t*)y_col, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
            } else {
                if (has_db) quant_dual_64x64_prmt_k<false, false, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y_row, (uint8_t*)y_col, (float*)dbias_f32_ws, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
                else        quant_dual_64x64_prmt_k<false, false, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y_row, (uint8_t*)y_col, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
            }
        } else {
            if (is_e5m2) {
                if (has_db) quant_dual_64x64_prmt_k<true, true, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y_row, (uint8_t*)y_col, (float*)dbias_f32_ws, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
                else        quant_dual_64x64_prmt_k<true, true, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y_row, (uint8_t*)y_col, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
            } else {
                if (has_db) quant_dual_64x64_prmt_k<false, true, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y_row, (uint8_t*)y_col, (float*)dbias_f32_ws, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
                else        quant_dual_64x64_prmt_k<false, true, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)x, (uint8_t*)y_row, (uint8_t*)y_col, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
            }
        }
    } else {
        launch_quant_1d_bf16(x, y_row, curr_scale, curr_inv, next_amax, slot_id, (int64_t)M * N, is_e5m2, warmed_up, stream);
        launch_transpose_fp8(y_row, y_col, M, N, stream);
        if (has_db) {
            dim3 g_sum((N + 255) / 256, min(M, 128));
            sum_rows_bf16_f32_k<<<g_sum, 256, 0, stream>>>((const __nv_bfloat16*)x, (float*)dbias_f32_ws, M, N);
        }
    }
    if (has_db) {
        cast_f32_to_bf16_k<<<(N + 255) / 256, 256, 0, stream>>>((const float*)dbias_f32_ws, (__nv_bfloat16*)dbias_bf16, N);
    }
}

void launch_swiglu_delayed(const void* gu, void* h_row, void* h_col, const void* curr_scale, const void* curr_inv, void* next_amax,
                           int slot_id, int M, int N, int dual, int warmed_up, cudaStream_t stream) {
    int64_t total_vec8 = (int64_t)M * (N >> 3);
    int blocks = (int)min((int64_t)2112, (total_vec8 + 255) / 256);
    if (dual && ((M & 3) == 0) && ((N & 3) == 0)) {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        if (!warmed_up) {
            swiglu_amax_calib_vec8_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)gu, (int32_t*)next_amax, slot_id, M, N);
            calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 448.0f);
            swiglu_dual_64x64_prmt_k<false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)gu, (uint8_t*)h_row, (uint8_t*)h_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        } else {
            swiglu_dual_64x64_prmt_k<true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)gu, (uint8_t*)h_row, (uint8_t*)h_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        }
        return;
    }
    if (!warmed_up) {
        swiglu_amax_calib_vec8_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)gu, (int32_t*)next_amax, slot_id, M, N);
        calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 448.0f);
        swiglu_1d_vec8_k<false><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)gu, (uint8_t*)h_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
    } else {
        swiglu_1d_vec8_k<true><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)gu, (uint8_t*)h_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
    }
    if (dual) {
        launch_transpose_fp8(h_row, h_col, M, N, stream);
    }
}

void launch_dswiglu_dual_delayed(const void* dh, const void* gu, void* dgu_row, void* dgu_t, const void* curr_scale, const void* curr_inv, void* next_amax,
                                 int slot_id, int M, int N, int warmed_up, cudaStream_t stream) {
    int64_t total_vec8 = (int64_t)M * (N >> 3);
    int blocks = (int)min((int64_t)2112, (total_vec8 + 255) / 256);
    if (((M & 3) == 0) && ((N & 3) == 0)) {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        if (!warmed_up) {
            dswiglu_amax_calib_vec8_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)gu, (int32_t*)next_amax, slot_id, M, N);
            calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 57344.0f);
            dswiglu_dual_64x64_prmt_k<false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)gu, (uint8_t*)dgu_row, (uint8_t*)dgu_t, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        } else {
            dswiglu_dual_64x64_prmt_k<true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)gu, (uint8_t*)dgu_row, (uint8_t*)dgu_t, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        }
        return;
    }
    if (!warmed_up) {
        dswiglu_amax_calib_vec8_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)gu, (int32_t*)next_amax, slot_id, M, N);
        calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 57344.0f);
        dswiglu_1d_vec8_k<false><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)gu, (uint8_t*)dgu_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
    } else {
        dswiglu_1d_vec8_k<true><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)gu, (uint8_t*)dgu_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
    }
    launch_transpose_fp8(dgu_row, dgu_t, M, 2 * N, stream);
}

void launch_gelu_bias_delayed(const void* z, const void* b, void* zb_out, void* h_row, void* h_col, const void* curr_scale, const void* curr_inv, void* next_amax,
                              int slot_id, int M, int N, int dual, int warmed_up, cudaStream_t stream) {
    int64_t total_vec8 = (int64_t)M * (N >> 3);
    int blocks = (int)min((int64_t)2112, (total_vec8 + 255) / 256);
    if (dual && ((M & 3) == 0) && ((N & 3) == 0)) {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        if (!warmed_up) {
            gelu_bias_amax_calib_vec8_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)z, (const __nv_bfloat16*)b, (int32_t*)next_amax, slot_id, M, N);
            calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 448.0f);
            gelu_bias_dual_64x64_prmt_k<false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)z, (const __nv_bfloat16*)b, (__nv_bfloat16*)zb_out, (uint8_t*)h_row, (uint8_t*)h_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        } else {
            gelu_bias_dual_64x64_prmt_k<true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)z, (const __nv_bfloat16*)b, (__nv_bfloat16*)zb_out, (uint8_t*)h_row, (uint8_t*)h_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        }
        return;
    }
    if (!warmed_up) {
        gelu_bias_amax_calib_vec8_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)z, (const __nv_bfloat16*)b, (int32_t*)next_amax, slot_id, M, N);
        calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 448.0f);
        if (dual) gelu_bias_1d_vec8_k<true, false><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)z, (const __nv_bfloat16*)b, (__nv_bfloat16*)zb_out, (uint8_t*)h_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        else      gelu_bias_1d_vec8_k<false, false><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)z, (const __nv_bfloat16*)b, nullptr, (uint8_t*)h_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
    } else {
        if (dual) gelu_bias_1d_vec8_k<true, true><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)z, (const __nv_bfloat16*)b, (__nv_bfloat16*)zb_out, (uint8_t*)h_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        else      gelu_bias_1d_vec8_k<false, true><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)z, (const __nv_bfloat16*)b, nullptr, (uint8_t*)h_row, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
    }
    if (dual) {
        launch_transpose_fp8(h_row, h_col, M, N, stream);
    }
}

void launch_dgelu_dual_delayed(const void* dh, const void* zb, void* dz_row, void* dz_t, void* db1_bf16, void* db1_f32_ws,
                               const void* curr_scale, const void* curr_inv, void* next_amax,
                               int slot_id, int M, int N, int warmed_up, cudaStream_t stream) {
    cudaMemsetAsync(db1_f32_ws, 0, (size_t)N * sizeof(float), stream);
    int n8 = N >> 3;
    int64_t total_vec8 = (int64_t)M * n8;
    int blocks = (int)min((int64_t)2112, (total_vec8 + 255) / 256);
    if (((M & 3) == 0) && ((N & 3) == 0)) {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        if (!warmed_up) {
            dgelu_amax_calib_vec8_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)zb, (int32_t*)next_amax, slot_id, M, N);
            calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 57344.0f);
            dgelu_dual_64x64_prmt_k<false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)zb, (uint8_t*)dz_row, (uint8_t*)dz_t, (float*)db1_f32_ws, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        } else {
            dgelu_dual_64x64_prmt_k<true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)zb, (uint8_t*)dz_row, (uint8_t*)dz_t, (float*)db1_f32_ws, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
        }
        cast_f32_to_bf16_k<<<(N + 255) / 256, 256, 0, stream>>>((const float*)db1_f32_ws, (__nv_bfloat16*)db1_bf16, N);
        return;
    }
    dim3 grid_pers((n8 + 255) / 256, min(M, 352));
    if (!warmed_up) {
        dgelu_amax_calib_vec8_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)zb, (int32_t*)next_amax, slot_id, M, N);
        calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 57344.0f);
        dgelu_1d_vec8_k<false><<<grid_pers, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)zb, (uint8_t*)dz_row, (float*)db1_f32_ws, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
    } else {
        dgelu_1d_vec8_k<true><<<grid_pers, 256, 0, stream>>>((const __nv_bfloat16*)dh, (const __nv_bfloat16*)zb, (uint8_t*)dz_row, (float*)db1_f32_ws, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, N);
    }
    launch_transpose_fp8(dz_row, dz_t, M, N, stream);
    cast_f32_to_bf16_k<<<(N + 255) / 256, 256, 0, stream>>>((const float*)db1_f32_ws, (__nv_bfloat16*)db1_bf16, N);
}

void launch_ln_fwd_1024(const void* x, const void* gamma, const void* beta, void* y_out, void* mean_out, void* rstd_out,
                        int M, float eps, cudaStream_t stream) {
    int blocks = (M + 7) / 8;
    if (mean_out != nullptr) {
        ln_fwd_1024_w8_k<true><<<blocks, 256, 0, stream>>>(
            (const __nv_bfloat16*)x, (const __nv_bfloat16*)gamma, (const __nv_bfloat16*)beta,
            (__nv_bfloat16*)y_out, (float*)mean_out, (float*)rstd_out, M, eps);
    } else {
        ln_fwd_1024_w8_k<false><<<blocks, 256, 0, stream>>>(
            (const __nv_bfloat16*)x, (const __nv_bfloat16*)gamma, (const __nv_bfloat16*)beta,
            (__nv_bfloat16*)y_out, nullptr, nullptr, M, eps);
    }
}

void launch_ln_bwd_1024(const void* dy, const void* d_res_in, const void* x, const void* gamma,
                        const void* mean_p, const void* rstd_p,
                        void* dx, void* dgamma_f32, void* dbeta_f32,
                        void* dres_sum_bf16, void* dres_sum_f32_ws,
                        int M, cudaStream_t stream) {
    int blocks = min((M + 3) / 4, 528);
    if (d_res_in != nullptr) {
        if (dres_sum_bf16 != nullptr) {
            cudaMemsetAsync(dres_sum_f32_ws, 0, 1024 * sizeof(float), stream);
            ln_bwd_1024_w4_l1_k<true, true><<<blocks, 128, 0, stream>>>(
                (const __nv_bfloat16*)dy, (const __nv_bfloat16*)d_res_in, (const __nv_bfloat16*)x,
                (const __nv_bfloat16*)gamma, (const float*)mean_p, (const float*)rstd_p,
                (__nv_bfloat16*)dx, (float*)dgamma_f32, (float*)dbeta_f32, (float*)dres_sum_f32_ws, M);
            cast_f32_to_bf16_k<<<4, 256, 0, stream>>>((const float*)dres_sum_f32_ws, (__nv_bfloat16*)dres_sum_bf16, 1024);
        } else {
            ln_bwd_1024_w4_l1_k<true, false><<<blocks, 128, 0, stream>>>(
                (const __nv_bfloat16*)dy, (const __nv_bfloat16*)d_res_in, (const __nv_bfloat16*)x,
                (const __nv_bfloat16*)gamma, (const float*)mean_p, (const float*)rstd_p,
                (__nv_bfloat16*)dx, (float*)dgamma_f32, (float*)dbeta_f32, nullptr, M);
        }
    } else {
        ln_bwd_1024_w4_l1_k<false, false><<<blocks, 128, 0, stream>>>(
            (const __nv_bfloat16*)dy, nullptr, (const __nv_bfloat16*)x,
            (const __nv_bfloat16*)gamma, (const float*)mean_p, (const float*)rstd_p,
            (__nv_bfloat16*)dx, (float*)dgamma_f32, (float*)dbeta_f32, nullptr, M);
    }
}

void launch_qwen_qkv_norm_rope_fwd(const void* qkv, const void* qw, const void* kw, const void* cos_p, const void* sin_p,
                                   void* q_out, void* k_out, void* v_out, void* rstd_out,
                                   int B, int S, int h_q, int h_kv, int bhsd, float eps, cudaStream_t stream) {
    int T = B * S;
    int t_groups = (T + 3) / 4;
    int total_sw = t_groups * (h_q + 2 * h_kv);
    if (bhsd) {
        qwen_qkv_norm_rope_fwd_vec8_k<true><<<(total_sw + 31) / 32, 256, 0, stream>>>(
            (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
            (const __nv_bfloat16*)cos_p, (const __nv_bfloat16*)sin_p,
            (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out, (float*)rstd_out,
            B, S, h_q, h_kv, eps);
    } else {
        qwen_qkv_norm_rope_fwd_vec8_k<false><<<(total_sw + 31) / 32, 256, 0, stream>>>(
            (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
            (const __nv_bfloat16*)cos_p, (const __nv_bfloat16*)sin_p,
            (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out, (float*)rstd_out,
            B, S, h_q, h_kv, eps);
    }
}

void launch_qwen_qkv_norm_rope_bwd(const void* dq, const void* dk, const void* dv, const void* qkv,
                                   const void* qw, const void* kw, const void* cos_p, const void* sin_p, const void* rstd_p,
                                   void* dqkv_bf16, void* dqkv_e5_row, void* dqkv_e5_t,
                                   const void* curr_inv, void* next_amax, int slot_dqkv,
                                   void* dqw_f32, void* dkw_f32,
                                   int B, int S, int h_q, int h_kv, int bhsd, cudaStream_t stream) {
    int T = B * S;
    int h_tot = h_q + 2 * h_kv;
    bool quant_e5 = (dqkv_e5_row != nullptr);
    if (quant_e5) {
        int gy = min(max(1, (T + 63) / 64), 22);
        dim3 grid(h_tot, gy);
        if (bhsd) {
            qwen_qkv_norm_rope_bwd_pers64_dual_k<true><<<grid, 256, 0, stream>>>(
                (const __nv_bfloat16*)dq, (const __nv_bfloat16*)dk, (const __nv_bfloat16*)dv,
                (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
                (const __nv_bfloat16*)cos_p, (const __nv_bfloat16*)sin_p, (const float*)rstd_p,
                (uint8_t*)dqkv_e5_row, (uint8_t*)dqkv_e5_t, (const float*)curr_inv, (int32_t*)next_amax, slot_dqkv,
                (float*)dqw_f32, (float*)dkw_f32, B, S, h_q, h_kv);
        } else {
            qwen_qkv_norm_rope_bwd_pers64_dual_k<false><<<grid, 256, 0, stream>>>(
                (const __nv_bfloat16*)dq, (const __nv_bfloat16*)dk, (const __nv_bfloat16*)dv,
                (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
                (const __nv_bfloat16*)cos_p, (const __nv_bfloat16*)sin_p, (const float*)rstd_p,
                (uint8_t*)dqkv_e5_row, (uint8_t*)dqkv_e5_t, (const float*)curr_inv, (int32_t*)next_amax, slot_dqkv,
                (float*)dqw_f32, (float*)dkw_f32, B, S, h_q, h_kv);
        }
        return;
    }

    int t_groups = (T + 7) / 8;
    int total_sw = t_groups * h_tot;
    int blocks = (total_sw + 31) / 32;
    if (bhsd) {
        qwen_qkv_norm_rope_bwd_opt_k<true, false><<<blocks, 256, 0, stream>>>(
            (const __nv_bfloat16*)dq, (const __nv_bfloat16*)dk, (const __nv_bfloat16*)dv,
            (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
            (const __nv_bfloat16*)cos_p, (const __nv_bfloat16*)sin_p, (const float*)rstd_p,
            (__nv_bfloat16*)dqkv_bf16, nullptr, nullptr, nullptr, 0,
            (float*)dqw_f32, (float*)dkw_f32, B, S, h_q, h_kv);
    } else {
        qwen_qkv_norm_rope_bwd_opt_k<false, false><<<blocks, 256, 0, stream>>>(
            (const __nv_bfloat16*)dq, (const __nv_bfloat16*)dk, (const __nv_bfloat16*)dv,
            (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
            (const __nv_bfloat16*)cos_p, (const __nv_bfloat16*)sin_p, (const float*)rstd_p,
            (__nv_bfloat16*)dqkv_bf16, nullptr, nullptr, nullptr, 0,
            (float*)dqw_f32, (float*)dkw_f32, B, S, h_q, h_kv);
    }
}

void launch_vis_qkv_rope_fwd(const void* qkv, const void* cos_f32, const void* sin_f32,
                             void* q_out, void* k_out, void* v_out,
                             int total_tokens, int seq_per_cam, int num_heads, cudaStream_t stream) {
    int total_heads = total_tokens * 3 * num_heads;
    int blocks = (total_heads + 63) / 64;
    vis_qkv_rope_fwd_vec8_k<<<blocks, 256, 0, stream>>>(
        (const __nv_bfloat16*)qkv, (const float*)cos_f32, (const float*)sin_f32,
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        total_tokens, seq_per_cam, num_heads);
}

void launch_vis_qkv_rope_bwd(const void* dq, const void* dk, const void* dv,
                             const void* cos_f32, const void* sin_f32, void* dqkv,
                             void* dqkv_bias_bf16, void* dqkv_bias_f32_ws,
                             int total_tokens, int seq_per_cam, int num_heads, cudaStream_t stream) {
    int cols = 3 * num_heads * 64;
    if (dqkv_bias_bf16 != nullptr) {
        cudaMemsetAsync(dqkv_bias_f32_ws, 0, (size_t)cols * sizeof(float), stream);
        int gy = min(max(1, (total_tokens + 63) / 64), 22);
        dim3 grid(3 * num_heads, gy);
        vis_qkv_rope_bwd_pers_k<<<grid, 256, 0, stream>>>(
            (const __nv_bfloat16*)dq, (const __nv_bfloat16*)dk, (const __nv_bfloat16*)dv,
            (const float*)cos_f32, (const float*)sin_f32, (__nv_bfloat16*)dqkv, (float*)dqkv_bias_f32_ws,
            total_tokens, seq_per_cam, num_heads);
        cast_f32_to_bf16_k<<<(cols + 255) / 256, 256, 0, stream>>>((const float*)dqkv_bias_f32_ws, (__nv_bfloat16*)dqkv_bias_bf16, cols);
    } else {
        int total_heads = total_tokens * 3 * num_heads;
        int blocks_fast = (total_heads + 63) / 64;
        vis_qkv_rope_bwd_vec8_k<<<blocks_fast, 256, 0, stream>>>(
            (const __nv_bfloat16*)dq, (const __nv_bfloat16*)dk, (const __nv_bfloat16*)dv,
            (const float*)cos_f32, (const float*)sin_f32, (__nv_bfloat16*)dqkv,
            total_tokens, seq_per_cam, num_heads);
    }
}

void launch_rmsnorm_quant_delayed(const void* x, const void* res_in, void* sum_out, const void* w,
                                  void* y_row, void* y_col, void* rstd_out,
                                  const void* curr_scale, const void* curr_inv, void* next_amax,
                                  int slot_id, int M, int D, float eps, int dual, int warmed_up, cudaStream_t stream) {
    bool has_res = (res_in != nullptr);
    if (!warmed_up) {
        rmsnorm_amax_calib_k<<<M, 256, 0, stream>>>(
            (const __nv_bfloat16*)x, (const __nv_bfloat16*)res_in, (__nv_bfloat16*)sum_out,
            (const __nv_bfloat16*)w, (int32_t*)next_amax, slot_id, M, D, eps);
        calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 448.0f);
        if (D == 2560) {
            const __nv_bfloat16* src = has_res ? (const __nv_bfloat16*)sum_out : (const __nv_bfloat16*)x;
            if (dual) rmsnorm_quant_2560_reg_k<false, true, false><<<M, 256, 0, stream>>>(src, nullptr, nullptr, (const __nv_bfloat16*)w, (uint8_t*)y_row, (float*)rstd_out, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, eps);
            else      rmsnorm_quant_2560_reg_k<false, false, false><<<M, 256, 0, stream>>>(src, nullptr, nullptr, (const __nv_bfloat16*)w, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, eps);
        } else {
            const __nv_bfloat16* src = has_res ? (const __nv_bfloat16*)sum_out : (const __nv_bfloat16*)x;
            if (dual) rmsnorm_quant_generic_k<true, false><<<M, 256, 0, stream>>>(src, nullptr, nullptr, (const __nv_bfloat16*)w, (uint8_t*)y_row, (float*)rstd_out, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, D, eps);
            else      rmsnorm_quant_generic_k<false, false><<<M, 256, 0, stream>>>(src, nullptr, nullptr, (const __nv_bfloat16*)w, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, D, eps);
        }
    } else {
        if (D == 2560) {
            if (has_res) {
                if (dual) rmsnorm_quant_2560_reg_k<true, true, true><<<M, 256, 0, stream>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)res_in, (__nv_bfloat16*)sum_out, (const __nv_bfloat16*)w, (uint8_t*)y_row, (float*)rstd_out, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, eps);
                else      rmsnorm_quant_2560_reg_k<true, false, true><<<M, 256, 0, stream>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)res_in, (__nv_bfloat16*)sum_out, (const __nv_bfloat16*)w, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, eps);
            } else {
                if (dual && y_col != nullptr) {
                    static bool smem_init[16] = {false};
                    int dev_id = 0;
                    cudaGetDevice(&dev_id);
                    constexpr int smem_bytes = 2560 * 65 + 256;
                    if (dev_id >= 0 && dev_id < 16 && !smem_init[dev_id]) {
                        cudaFuncSetAttribute(rmsnorm_quant_2560_pers64_dual_k<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                        cudaFuncSetAttribute(rmsnorm_quant_2560_pers64_dual_k<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
                        smem_init[dev_id] = true;
                    }
                    int blocks = min((M + 63) / 64, 132);
                    if (rstd_out != nullptr) {
                        rmsnorm_quant_2560_pers64_dual_k<true><<<blocks, 512, smem_bytes, stream>>>(
                            (const __nv_bfloat16*)x, (const __nv_bfloat16*)w,
                            (uint8_t*)y_row, (uint8_t*)y_col, (float*)rstd_out,
                            (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, eps);
                    } else {
                        rmsnorm_quant_2560_pers64_dual_k<false><<<blocks, 512, smem_bytes, stream>>>(
                            (const __nv_bfloat16*)x, (const __nv_bfloat16*)w,
                            (uint8_t*)y_row, (uint8_t*)y_col, nullptr,
                            (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, eps);
                    }
                    return;
                }
                if (dual) rmsnorm_quant_2560_reg_k<false, true, true><<<M, 256, 0, stream>>>((const __nv_bfloat16*)x, nullptr, nullptr, (const __nv_bfloat16*)w, (uint8_t*)y_row, (float*)rstd_out, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, eps);
                else      rmsnorm_quant_2560_reg_k<false, false, true><<<M, 256, 0, stream>>>((const __nv_bfloat16*)x, nullptr, nullptr, (const __nv_bfloat16*)w, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, eps);
            }
        } else {
            if (dual) rmsnorm_quant_generic_k<true, true><<<M, 256, 0, stream>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)res_in, (__nv_bfloat16*)sum_out, (const __nv_bfloat16*)w, (uint8_t*)y_row, (float*)rstd_out, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, D, eps);
            else      rmsnorm_quant_generic_k<false, true><<<M, 256, 0, stream>>>((const __nv_bfloat16*)x, (const __nv_bfloat16*)res_in, (__nv_bfloat16*)sum_out, (const __nv_bfloat16*)w, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, M, D, eps);
        }
    }
    if (dual) {
        launch_transpose_fp8(y_row, y_col, M, D, stream);
    }
}

void launch_rmsnorm_bwd_2560(const void* dy, const void* d_res_in, const void* x, const void* w, const void* rstd_p,
                             void* dx, void* dx_e5_row, void* dx_e5_t,
                             const void* curr_inv, void* next_amax, int slot_dx,
                             void* dw_f32, int M, cudaStream_t stream) {
    int blocks = min(M, 792);  // 6 CTAs/SM = 1536 threads/SM
    bool quant_dx = (dx_e5_row != nullptr);
    if (d_res_in != nullptr) {
        if (quant_dx) {
            rmsnorm_bwd_2560_opt_k<true, true><<<blocks, 256, 0, stream>>>(
                (const __nv_bfloat16*)dy, (const __nv_bfloat16*)d_res_in, (const __nv_bfloat16*)x,
                (const __nv_bfloat16*)w, (const float*)rstd_p, (__nv_bfloat16*)dx,
                (uint8_t*)dx_e5_row, (const float*)curr_inv, (int32_t*)next_amax, slot_dx,
                (float*)dw_f32, M);
        } else {
            rmsnorm_bwd_2560_opt_k<true, false><<<blocks, 256, 0, stream>>>(
                (const __nv_bfloat16*)dy, (const __nv_bfloat16*)d_res_in, (const __nv_bfloat16*)x,
                (const __nv_bfloat16*)w, (const float*)rstd_p, (__nv_bfloat16*)dx,
                nullptr, nullptr, nullptr, 0, (float*)dw_f32, M);
        }
    } else {
        if (quant_dx) {
            rmsnorm_bwd_2560_opt_k<false, true><<<blocks, 256, 0, stream>>>(
                (const __nv_bfloat16*)dy, nullptr, (const __nv_bfloat16*)x,
                (const __nv_bfloat16*)w, (const float*)rstd_p, (__nv_bfloat16*)dx,
                (uint8_t*)dx_e5_row, (const float*)curr_inv, (int32_t*)next_amax, slot_dx,
                (float*)dw_f32, M);
        } else {
            rmsnorm_bwd_2560_opt_k<false, false><<<blocks, 256, 0, stream>>>(
                (const __nv_bfloat16*)dy, nullptr, (const __nv_bfloat16*)x,
                (const __nv_bfloat16*)w, (const float*)rstd_p, (__nv_bfloat16*)dx,
                nullptr, nullptr, nullptr, 0, (float*)dw_f32, M);
        }
    }
    if (quant_dx && dx_e5_t != nullptr) {
        launch_transpose_fp8(dx_e5_row, dx_e5_t, M, 2560, stream);
    }
}

void launch_mmdit_joint_qkv_fwd(const void* act_qkv, const void* txt_qkv,
                                const void* nq_w, const void* nk_w, const void* naq_w, const void* nak_w,
                                const void* act_cos, const void* act_sin, const void* txt_cos, const void* txt_sin,
                                void* joint_q, void* joint_k, void* joint_v, void* rstd_out,
                                int B, int S_act, int S_txt, int H, float eps, cudaStream_t stream) {
    int t_groups = (B * (S_act + S_txt) + 3) / 4;
    int total_sw = t_groups * 3 * H;
    mmdit_joint_qkv_norm_rope_pad64_fwd_vec8_k<<<(total_sw + 31) / 32, 256, 0, stream>>>(
        (const __nv_bfloat16*)act_qkv, (const __nv_bfloat16*)txt_qkv,
        (const __nv_bfloat16*)nq_w, (const __nv_bfloat16*)nk_w, (const __nv_bfloat16*)naq_w, (const __nv_bfloat16*)nak_w,
        (const float*)act_cos, (const float*)act_sin, (const float*)txt_cos, (const float*)txt_sin,
        (__nv_bfloat16*)joint_q, (__nv_bfloat16*)joint_k, (__nv_bfloat16*)joint_v, (float*)rstd_out,
        B, S_act, S_txt, H, eps);
}

void launch_mmdit_joint_qkv_bwd(const void* dj_q, const void* dj_k, const void* dj_v,
                                const void* act_qkv, const void* txt_qkv,
                                const void* nq_w, const void* nk_w, const void* naq_w, const void* nak_w,
                                const void* act_cos, const void* act_sin, const void* txt_cos, const void* txt_sin,
                                const void* rstd_p,
                                void* d_act_qkv, void* d_txt_qkv,
                                void* d_nq_f32, void* d_nk_f32, void* d_naq_f32, void* d_nak_f32,
                                int B, int S_act, int S_txt, int H, cudaStream_t stream) {
    int total_tok = B * (S_act + S_txt);
    dim3 grid(min(total_tok, 176), (3 * H + 31) / 32);
    mmdit_joint_qkv_norm_rope_pad64_bwd_vec8_k<<<grid, 256, 0, stream>>>(
        (const __nv_bfloat16*)dj_q, (const __nv_bfloat16*)dj_k, (const __nv_bfloat16*)dj_v,
        (const __nv_bfloat16*)act_qkv, (const __nv_bfloat16*)txt_qkv,
        (const __nv_bfloat16*)nq_w, (const __nv_bfloat16*)nk_w, (const __nv_bfloat16*)naq_w, (const __nv_bfloat16*)nak_w,
        (const float*)act_cos, (const float*)act_sin, (const float*)txt_cos, (const float*)txt_sin,
        (const float*)rstd_p,
        (__nv_bfloat16*)d_act_qkv, (__nv_bfloat16*)d_txt_qkv,
        (float*)d_nq_f32, (float*)d_nk_f32, (float*)d_naq_f32, (float*)d_nak_f32,
        B, S_act, S_txt, H);
}

void launch_mmdit_split_unpad_fwd(const void* joint_out, void* act_out, void* txt_out,
                                  int B, int S_act, int S_txt, int H, cudaStream_t stream) {
    int t_groups = (B * (S_act + S_txt) + 3) / 4;
    int total_sw = t_groups * H;
    mmdit_split_unpad_fwd_vec8_k<<<(total_sw + 31) / 32, 256, 0, stream>>>(
        (const __nv_bfloat16*)joint_out, (__nv_bfloat16*)act_out, (__nv_bfloat16*)txt_out, B, S_act, S_txt, H);
}

void launch_mmdit_split_unpad_bwd(const void* d_act_out, const void* d_txt_out, void* d_joint_out,
                                  int B, int S_act, int S_txt, int H, cudaStream_t stream) {
    int t_groups = (B * (S_act + S_txt) + 3) / 4;
    int total_sw = t_groups * H;
    mmdit_split_unpad_bwd_vec8_k<<<(total_sw + 31) / 32, 256, 0, stream>>>(
        (const __nv_bfloat16*)d_act_out, (const __nv_bfloat16*)d_txt_out, (__nv_bfloat16*)d_joint_out, B, S_act, S_txt, H);
}

// --- Autotuned FP8 cublasLt Matmul ---
// Computes row-major D(M,N) = (scale_a * A(M,K)) @ (scale_b * B(N,K))^T + beta * C(M,N) + bias(N)
int lt_fp8_matmul(
    const void* A_row,
    const void* B_row,
    const void* C_row,
    void* D_row,
    const void* scale_a,
    const void* scale_b,
    const void* bias,
    void* workspace,
    size_t workspace_bytes,
    int M, int N, int K,
    int a_e5, int b_e5, int fast_accum, int sm_target,
    cudaStream_t stream
) {
    int dev = 0;
    cudaGetDevice(&dev);
    cublasLtHandle_t lt = get_lt_handle(dev);

    int has_bias = (bias != nullptr) ? 1 : 0;
    int has_beta = (C_row != nullptr) ? 1 : 0;
    LtKey key{dev, 0, 0, M, N, K, a_e5, b_e5, has_bias, has_beta, fast_accum, sm_target};
    auto it = g_lt_cache.find(key);
    const float alpha = 1.0f;
    const float beta = has_beta ? 1.0f : 0.0f;

    if (it == g_lt_cache.end()) {
        LtPlan plan{};
        cublasLtMatmulDescCreate(&plan.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
        int8_t fa = (int8_t)fast_accum;
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fa, sizeof(fa));
        if (sm_target > 0) {
            int32_t smt = sm_target;
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &smt, sizeof(smt));
        }
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_b, sizeof(scale_b));
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_a, sizeof(scale_a));
        if (has_bias) {
            cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
            cudaDataType_t btype = CUDA_R_16BF;
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype));
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        }

        cudaDataType_t t_first = b_e5 ? CUDA_R_8F_E5M2 : CUDA_R_8F_E4M3;
        cudaDataType_t t_second = a_e5 ? CUDA_R_8F_E5M2 : CUDA_R_8F_E4M3;
        cublasLtMatrixLayoutCreate(&plan.aDesc, t_first, K, N, K);
        cublasLtMatrixLayoutCreate(&plan.bDesc, t_second, K, M, K);
        cublasLtMatrixLayoutCreate(&plan.cDesc, CUDA_R_16BF, N, M, N);
        cublasLtMatrixLayoutCreate(&plan.dDesc, CUDA_R_16BF, N, M, N);

        LtKey key0 = key;
        key0.dev = 0;
        auto imp_it = g_lt_imported_algos.find(key0);
        if (imp_it != g_lt_imported_algos.end()) {
            cublasLtMatmulHeuristicResult_t chk{};
            if (cublasLtMatmulAlgoCheck(lt, plan.opDesc, plan.aDesc, plan.bDesc, plan.cDesc, plan.dDesc, &imp_it->second, &chk) == CUBLAS_STATUS_SUCCESS
                && chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= workspace_bytes) {
                plan.bestAlgo = imp_it->second;
                plan.valid = 1;
                it = g_lt_cache.emplace(key, plan).first;
            }
        }
        if (it == g_lt_cache.end()) {
            cublasLtMatmulPreference_t pref;
            cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
            cublasLtMatmulHeuristicResult_t results[64];
            int returned = 0;
            cublasLtMatmulAlgoGetHeuristic(lt, plan.opDesc, plan.aDesc, plan.bDesc, plan.cDesc, plan.dDesc, pref, 64, results, &returned);
            cublasLtMatmulPreferenceDestroy(pref);

            if (returned == 0) {
                plan.valid = 0;
                g_lt_cache.emplace(key, plan);
                return -1;
            }

            if (!g_lt_autotune_enabled) {
                int found = 0;
                for (int i = 0; i < returned; ++i) {
                    if (results[i].state == CUBLAS_STATUS_SUCCESS && results[i].workspaceSize <= workspace_bytes) {
                        plan.bestAlgo = results[i].algo;
                        found = 1;
                        break;
                    }
                }
                plan.valid = found;
            } else {
                void* bench_D = D_row;
                bool free_tmp = false;
                if (has_beta && C_row == D_row) {
                    cudaMallocAsync(&bench_D, (size_t)M * N * sizeof(__nv_bfloat16), stream);
                    free_tmp = true;
                }
                cudaDeviceSynchronize();
                float best_ms = 1e9f;
                int found = 0;
                void* l2_flush = nullptr;
                const size_t l2_bytes = 56ull * 1024ull * 1024ull;
                cudaMallocAsync(&l2_flush, l2_bytes, stream);
                cudaEvent_t e0[3], e1[3];
                for (int r = 0; r < 3; ++r) { cudaEventCreate(&e0[r]); cudaEventCreate(&e1[r]); }
                for (int i = 0; i < returned; ++i) {
                    if (results[i].state != CUBLAS_STATUS_SUCCESS || results[i].workspaceSize > workspace_bytes) continue;
                    cublasStatus_t st = cublasLtMatmul(
                        lt, plan.opDesc, &alpha, B_row, plan.aDesc, A_row, plan.bDesc,
                        &beta, has_beta ? C_row : bench_D, plan.cDesc, bench_D, plan.dDesc,
                        &results[i].algo, workspace, workspace_bytes, stream
                    );
                    if (st != CUBLAS_STATUS_SUCCESS) continue;
                    for (int r = 0; r < 3; ++r) {
                        if (l2_flush) cudaMemsetAsync(l2_flush, 0, l2_bytes, stream);
                        cudaEventRecord(e0[r], stream);
                        cublasLtMatmul(
                            lt, plan.opDesc, &alpha, B_row, plan.aDesc, A_row, plan.bDesc,
                            &beta, has_beta ? C_row : bench_D, plan.cDesc, bench_D, plan.dDesc,
                            &results[i].algo, workspace, workspace_bytes, stream
                        );
                        cudaEventRecord(e1[r], stream);
                    }
                    cudaEventSynchronize(e1[2]);
                    float sum_ms = 0.0f;
                    for (int r = 0; r < 3; ++r) {
                        float r_ms = 0.0f;
                        cudaEventElapsedTime(&r_ms, e0[r], e1[r]);
                        sum_ms += r_ms;
                    }
                    if (sum_ms < best_ms) {
                        best_ms = sum_ms;
                        plan.bestAlgo = results[i].algo;
                        found = 1;
                    }
                }
                for (int r = 0; r < 3; ++r) { cudaEventDestroy(e0[r]); cudaEventDestroy(e1[r]); }
                if (l2_flush) cudaFreeAsync(l2_flush, stream);
                if (free_tmp) cudaFreeAsync(bench_D, stream);
                plan.valid = found;
            }
            it = g_lt_cache.emplace(key, plan).first;
        }
    }

    LtPlan& p = it->second;
    if (!p.valid) return -1;
    cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_b, sizeof(scale_b));
    cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_a, sizeof(scale_a));
    if (has_bias) {
        cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    }
    cublasStatus_t st = cublasLtMatmul(
        lt, p.opDesc, &alpha, B_row, p.aDesc, A_row, p.bDesc,
        &beta, has_beta ? C_row : D_row, p.cDesc, D_row, p.dDesc,
        &p.bestAlgo, workspace, workspace_bytes, stream
    );
    return (st == CUBLAS_STATUS_SUCCESS) ? 0 : (int)st;
}

// --- Autotuned BF16 Vision cublasLt Matmul (Modes 0..6) ---
// Modes 0..3 (Fwd): M1=B_row(N,K)[OP_T], M2=A_row(M,K)[OP_N] -> D_col(N,M) = D_row(M,N)
// Modes 4..5 (dX) : M1=B_row=W(N,K)[OP_N], M2=A_row=dY(M,N)[OP_N] -> D_col(K,M) = D_row(M,K)
// Mode  6    (dW) : M1=B_row=X(M,K)[OP_N], M2=A_row=dY(M,N)[OP_T] -> D_col(K,N) = dW_row(N,K), bias_ptr=dbias(N)
int lt_bf16_matmul(
    int mode,
    const void* A_row,
    const void* B_row,
    const void* C_row,
    void* D_row,
    void* bias_or_dbias,
    void* aux_buf,
    void* workspace,
    size_t workspace_bytes,
    int M, int N, int K,
    int sm_target,
    cudaStream_t stream
) {
    int dev = 0;
    cudaGetDevice(&dev);
    cublasLtHandle_t lt = get_lt_handle(dev);

    int has_beta = (C_row != nullptr) ? 1 : 0;
    int has_bias = (bias_or_dbias != nullptr) ? 1 : 0;
    LtKey key{dev, 1, mode, M, N, K, 0, 0, has_bias, has_beta, 0, sm_target};
    auto it = g_lt_cache.find(key);
    const float alpha = 1.0f;
    const float beta = has_beta ? 1.0f : 0.0f;

    if (it == g_lt_cache.end()) {
        LtPlan plan{};
        cublasLtMatmulDescCreate(&plan.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (sm_target > 0) {
            int32_t smt = sm_target;
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &smt, sizeof(smt));
        }

        cublasOperation_t op1, op2;
        if (mode <= 3) {
            op1 = CUBLAS_OP_T;
            op2 = CUBLAS_OP_N;
            cublasLtMatrixLayoutCreate(&plan.aDesc, CUDA_R_16BF, K, N, K);
            cublasLtMatrixLayoutCreate(&plan.bDesc, CUDA_R_16BF, K, M, K);
            cublasLtMatrixLayoutCreate(&plan.cDesc, CUDA_R_16BF, N, M, N);
            cublasLtMatrixLayoutCreate(&plan.dDesc, CUDA_R_16BF, N, M, N);
        } else if (mode <= 5 || mode == 7) {
            op1 = CUBLAS_OP_N;
            op2 = CUBLAS_OP_N;
            cublasLtMatrixLayoutCreate(&plan.aDesc, CUDA_R_16BF, K, N, K);
            cublasLtMatrixLayoutCreate(&plan.bDesc, CUDA_R_16BF, N, M, N);
            cublasLtMatrixLayoutCreate(&plan.cDesc, CUDA_R_16BF, K, M, K);
            cublasLtMatrixLayoutCreate(&plan.dDesc, CUDA_R_16BF, K, M, K);
        } else {
            op1 = CUBLAS_OP_N;
            op2 = CUBLAS_OP_T;
            cublasLtMatrixLayoutCreate(&plan.aDesc, CUDA_R_16BF, K, M, K);
            cublasLtMatrixLayoutCreate(&plan.bDesc, CUDA_R_16BF, N, M, N);
            cublasLtMatrixLayoutCreate(&plan.cDesc, CUDA_R_16BF, K, N, K);
            cublasLtMatrixLayoutCreate(&plan.dDesc, CUDA_R_16BF, K, N, K);
        }
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op1, sizeof(op1));
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op2, sizeof(op2));

        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DEFAULT;
        cudaDataType_t btype = CUDA_R_16BF;
        if (mode == 1) {
            epi = CUBLASLT_EPILOGUE_BIAS;
        } else if (mode == 2) {
            epi = CUBLASLT_EPILOGUE_GELU_BIAS;
        } else if (mode == 3) {
            epi = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
        } else if (mode == 5) {
            epi = CUBLASLT_EPILOGUE_DGELU;
        } else if (mode == 7) {
            epi = CUBLASLT_EPILOGUE_DGELU_BGRAD;
        } else if (mode == 6 && has_bias) {
            epi = CUBLASLT_EPILOGUE_BGRADB;
        }
        cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));

        if (mode == 1 || mode == 2 || mode == 3 || mode == 7 || (mode == 6 && has_bias)) {
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype));
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_or_dbias, sizeof(bias_or_dbias));
        }
        if (mode == 3 || mode == 5 || mode == 7) {
            int64_t aux_ld = (mode == 3) ? (int64_t)N : (int64_t)K;
            cudaDataType_t aux_dt = CUDA_R_16BF;
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, &aux_dt, sizeof(aux_dt));
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &aux_ld, sizeof(aux_ld));
            cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_buf, sizeof(aux_buf));
        }

        LtKey key0 = key;
        key0.dev = 0;
        auto imp_it = g_lt_imported_algos.find(key0);
        if (imp_it != g_lt_imported_algos.end()) {
            cublasLtMatmulHeuristicResult_t chk{};
            if (cublasLtMatmulAlgoCheck(lt, plan.opDesc, plan.aDesc, plan.bDesc, plan.cDesc, plan.dDesc, &imp_it->second, &chk) == CUBLAS_STATUS_SUCCESS
                && chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= workspace_bytes) {
                plan.bestAlgo = imp_it->second;
                plan.valid = 1;
                it = g_lt_cache.emplace(key, plan).first;
            }
        }
        if (it == g_lt_cache.end()) {
            cublasLtMatmulPreference_t pref;
            cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
            cublasLtMatmulHeuristicResult_t results[64];
            int returned = 0;
            cublasLtMatmulAlgoGetHeuristic(lt, plan.opDesc, plan.aDesc, plan.bDesc, plan.cDesc, plan.dDesc, pref, 64, results, &returned);
            cublasLtMatmulPreferenceDestroy(pref);

            if (returned == 0) {
                plan.valid = 0;
                g_lt_cache.emplace(key, plan);
                return -1;
            }

            if (!g_lt_autotune_enabled) {
                int found = 0;
                for (int i = 0; i < returned; ++i) {
                    if (results[i].state == CUBLAS_STATUS_SUCCESS && results[i].workspaceSize <= workspace_bytes) {
                        plan.bestAlgo = results[i].algo;
                        found = 1;
                        break;
                    }
                }
                plan.valid = found;
            } else {
                size_t d_elems = (mode <= 3) ? ((size_t)M * N) : ((mode <= 5 || mode == 7) ? ((size_t)M * K) : ((size_t)N * K));
                void* bench_D = D_row;
                bool free_tmp = false;
                if (has_beta && C_row == D_row) {
                    cudaMallocAsync(&bench_D, d_elems * sizeof(__nv_bfloat16), stream);
                    free_tmp = true;
                }
                cudaDeviceSynchronize();
                float best_ms = 1e9f;
                int found = 0;
                void* l2_flush = nullptr;
                const size_t l2_bytes = 56ull * 1024ull * 1024ull;
                cudaMallocAsync(&l2_flush, l2_bytes, stream);
                cudaEvent_t e0[3], e1[3];
                for (int r = 0; r < 3; ++r) { cudaEventCreate(&e0[r]); cudaEventCreate(&e1[r]); }
                for (int i = 0; i < returned; ++i) {
                    if (results[i].state != CUBLAS_STATUS_SUCCESS || results[i].workspaceSize > workspace_bytes) continue;
                    cublasStatus_t st = cublasLtMatmul(
                        lt, plan.opDesc, &alpha, B_row, plan.aDesc, A_row, plan.bDesc,
                        &beta, has_beta ? C_row : bench_D, plan.cDesc, bench_D, plan.dDesc,
                        &results[i].algo, workspace, workspace_bytes, stream
                    );
                    if (st != CUBLAS_STATUS_SUCCESS) continue;
                    for (int r = 0; r < 3; ++r) {
                        if (l2_flush) cudaMemsetAsync(l2_flush, 0, l2_bytes, stream);
                        cudaEventRecord(e0[r], stream);
                        cublasLtMatmul(
                            lt, plan.opDesc, &alpha, B_row, plan.aDesc, A_row, plan.bDesc,
                            &beta, has_beta ? C_row : bench_D, plan.cDesc, bench_D, plan.dDesc,
                            &results[i].algo, workspace, workspace_bytes, stream
                        );
                        cudaEventRecord(e1[r], stream);
                    }
                    cudaEventSynchronize(e1[2]);
                    float sum_ms = 0.0f;
                    for (int r = 0; r < 3; ++r) {
                        float r_ms = 0.0f;
                        cudaEventElapsedTime(&r_ms, e0[r], e1[r]);
                        sum_ms += r_ms;
                    }
                    if (sum_ms < best_ms) {
                        best_ms = sum_ms;
                        plan.bestAlgo = results[i].algo;
                        found = 1;
                    }
                }
                for (int r = 0; r < 3; ++r) { cudaEventDestroy(e0[r]); cudaEventDestroy(e1[r]); }
                if (l2_flush) cudaFreeAsync(l2_flush, stream);
                if (free_tmp) cudaFreeAsync(bench_D, stream);
                plan.valid = found;
            }
            it = g_lt_cache.emplace(key, plan).first;
        }
    }

    LtPlan& p = it->second;
    if (!p.valid) return -1;
    if (mode == 1 || mode == 2 || mode == 3 || mode == 7 || (mode == 6 && has_bias)) {
        cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_or_dbias, sizeof(bias_or_dbias));
    }
    if (mode == 3 || mode == 5 || mode == 7) {
        cublasLtMatmulDescSetAttribute(p.opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_buf, sizeof(aux_buf));
    }
    cublasStatus_t st = cublasLtMatmul(
        lt, p.opDesc, &alpha, B_row, p.aDesc, A_row, p.bDesc,
        &beta, has_beta ? C_row : D_row, p.cDesc, D_row, p.dDesc,
        &p.bestAlgo, workspace, workspace_bytes, stream
    );
    return (st == CUBLAS_STATUS_SUCCESS) ? 0 : (int)st;
}


void launch_adaln_modulate_quant_1536(
    const void* x, const void* mod_params, void* y_row, void* y_col,
    void* mean_out, void* rstd_out,
    const void* curr_scale, const void* curr_inv, void* next_amax,
    int slot_id, int B, int L, int L_mod, int mod_offset, float eps,
    int dual, int warmed_up, cudaStream_t stream
) {
    const int M = B * L;
    const bool save_stats = (mean_out != nullptr);
    if (!warmed_up) {
        adaln_modulate_amax_calib_1536_k<<<M, 192, 0, stream>>>(
            (const __nv_bfloat16*)x, (const __nv_bfloat16*)mod_params,
            (int32_t*)next_amax, slot_id, B, L, L_mod, mod_offset, eps);
        calib_scale_slot_k<<<1, 1, 0, stream>>>(
            (const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 448.0f);
        if (save_stats) {
            adaln_modulate_quant_1536_k<true, false><<<M, 192, 0, stream>>>(
                (const __nv_bfloat16*)x, (const __nv_bfloat16*)mod_params, (uint8_t*)y_row,
                (float*)mean_out, (float*)rstd_out, (const float*)curr_inv, (int32_t*)next_amax,
                slot_id, B, L, L_mod, mod_offset, eps);
        } else {
            adaln_modulate_quant_1536_k<false, false><<<M, 192, 0, stream>>>(
                (const __nv_bfloat16*)x, (const __nv_bfloat16*)mod_params, (uint8_t*)y_row,
                nullptr, nullptr, (const float*)curr_inv, (int32_t*)next_amax,
                slot_id, B, L, L_mod, mod_offset, eps);
        }
    } else {
        if (save_stats) {
            adaln_modulate_quant_1536_k<true, true><<<M, 192, 0, stream>>>(
                (const __nv_bfloat16*)x, (const __nv_bfloat16*)mod_params, (uint8_t*)y_row,
                (float*)mean_out, (float*)rstd_out, (const float*)curr_inv, (int32_t*)next_amax,
                slot_id, B, L, L_mod, mod_offset, eps);
        } else {
            adaln_modulate_quant_1536_k<false, true><<<M, 192, 0, stream>>>(
                (const __nv_bfloat16*)x, (const __nv_bfloat16*)mod_params, (uint8_t*)y_row,
                nullptr, nullptr, (const float*)curr_inv, (int32_t*)next_amax,
                slot_id, B, L, L_mod, mod_offset, eps);
        }
    }
    if (dual && y_col != nullptr) {
        launch_transpose_fp8(y_row, y_col, M, 1536, stream);
    }
}

void launch_gated_add_fwd_1536(
    const void* x, const void* sub_out, const void* mod_params, void* out,
    int B, int L, int L_mod, int mod_offset, cudaStream_t stream
) {
    gated_add_fwd_1536_k<<<B * L, 192, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)sub_out,
        (const __nv_bfloat16*)mod_params, (__nv_bfloat16*)out,
        B, L, L_mod, mod_offset);
}

void launch_gated_bwd_1536(
    const void* dout, const void* sub_out, const void* mod_params,
    void* d_sub_bf16, void* d_sub_e5_row, void* d_sub_e5_t,
    void* dbias_bf16, void* dbias_f32_ws, void* d_mod_f32,
    const void* curr_inv, void* next_amax, int slot_id,
    int B, int L, int L_mod, int mod_offset, cudaStream_t stream
) {
    const int M = B * L;
    const int gx = (L_mod == 1) ? min((L + 7) / 8, 8) : ((L + 7) / 8);
    const dim3 grid(gx, B);
    const bool quant_e5 = (d_sub_e5_row != nullptr);
    const bool has_db = (dbias_bf16 != nullptr);
    if (has_db) {
        cudaMemsetAsync(dbias_f32_ws, 0, 1536 * sizeof(float), stream);
    }
    if (quant_e5) {
        if (has_db) {
            gated_bwd_1536_w8_k<true, true><<<grid, 256, 0, stream>>>(
                (const __nv_bfloat16*)dout, (const __nv_bfloat16*)sub_out, (const __nv_bfloat16*)mod_params,
                nullptr, (uint8_t*)d_sub_e5_row, (float*)dbias_f32_ws, (float*)d_mod_f32,
                (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, L, L_mod, mod_offset);
        } else {
            gated_bwd_1536_w8_k<true, false><<<grid, 256, 0, stream>>>(
                (const __nv_bfloat16*)dout, (const __nv_bfloat16*)sub_out, (const __nv_bfloat16*)mod_params,
                nullptr, (uint8_t*)d_sub_e5_row, nullptr, (float*)d_mod_f32,
                (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, L, L_mod, mod_offset);
        }
        if (d_sub_e5_t != nullptr) {
            launch_transpose_fp8(d_sub_e5_row, d_sub_e5_t, M, 1536, stream);
        }
    } else {
        if (has_db) {
            gated_bwd_1536_w8_k<false, true><<<grid, 256, 0, stream>>>(
                (const __nv_bfloat16*)dout, (const __nv_bfloat16*)sub_out, (const __nv_bfloat16*)mod_params,
                (__nv_bfloat16*)d_sub_bf16, nullptr, (float*)dbias_f32_ws, (float*)d_mod_f32,
                nullptr, nullptr, 0, B, L, L_mod, mod_offset);
        } else {
            gated_bwd_1536_w8_k<false, false><<<grid, 256, 0, stream>>>(
                (const __nv_bfloat16*)dout, (const __nv_bfloat16*)sub_out, (const __nv_bfloat16*)mod_params,
                (__nv_bfloat16*)d_sub_bf16, nullptr, nullptr, (float*)d_mod_f32,
                nullptr, nullptr, 0, B, L, L_mod, mod_offset);
        }
    }
    if (has_db) {
        cast_f32_to_bf16_k<<<6, 256, 0, stream>>>((const float*)dbias_f32_ws, (__nv_bfloat16*)dbias_bf16, 1536);
    }
}

void launch_adaln_modulate_bwd_1536(
    const void* d_mod_x, const void* d_res_in, const void* x, const void* mod_params,
    const void* mean_p, const void* rstd_p,
    void* dx_out, void* d_mod_f32,
    int B, int L, int L_mod, int mod_offset, cudaStream_t stream
) {
    const int gx = (L_mod == 1) ? min(L, 12) : min(L, 16);
    const dim3 grid(gx, B);
    if (d_res_in != nullptr) {
        adaln_modulate_bwd_1536_r1_k<true><<<grid, 192, 0, stream>>>(
            (const __nv_bfloat16*)d_mod_x, (const __nv_bfloat16*)d_res_in,
            (const __nv_bfloat16*)x, (const __nv_bfloat16*)mod_params,
            (const float*)mean_p, (const float*)rstd_p,
            (__nv_bfloat16*)dx_out, (float*)d_mod_f32, B, L, L_mod, mod_offset);
    } else {
        adaln_modulate_bwd_1536_r1_k<false><<<grid, 192, 0, stream>>>(
            (const __nv_bfloat16*)d_mod_x, nullptr,
            (const __nv_bfloat16*)x, (const __nv_bfloat16*)mod_params,
            (const float*)mean_p, (const float*)rstd_p,
            (__nv_bfloat16*)dx_out, (float*)d_mod_f32, B, L, L_mod, mod_offset);
    }
}

void launch_mmdit_joint_qkv_48_fwd(
    const void* act_qkv, const void* txt_qkv,
    const void* nq_w, const void* nk_w, const void* naq_w, const void* nak_w,
    const void* act_cos, const void* act_sin, const void* txt_cos, const void* txt_sin,
    void* joint_q, void* joint_k, void* joint_v, void* rstd_out,
    int B, int S_act, int S_txt, int H, float eps, cudaStream_t stream
) {
    int t_groups = (B * (S_act + S_txt) + 3) / 4;
    int total_sw = t_groups * 3 * H;
    mmdit_joint_qkv_norm_rope_48_fwd_vec8_k<<<(total_sw + 31) / 32, 256, 0, stream>>>(
        (const __nv_bfloat16*)act_qkv, (const __nv_bfloat16*)txt_qkv,
        (const __nv_bfloat16*)nq_w, (const __nv_bfloat16*)nk_w, (const __nv_bfloat16*)naq_w, (const __nv_bfloat16*)nak_w,
        (const float*)act_cos, (const float*)act_sin, (const float*)txt_cos, (const float*)txt_sin,
        (__nv_bfloat16*)joint_q, (__nv_bfloat16*)joint_k, (__nv_bfloat16*)joint_v, (float*)rstd_out,
        B, S_act, S_txt, H, eps);
}

void launch_mmdit_joint_qkv_48_bwd(
    const void* dj_q, const void* dj_k, const void* dj_v,
    const void* act_qkv, const void* txt_qkv,
    const void* nq_w, const void* nk_w, const void* naq_w, const void* nak_w,
    const void* act_cos, const void* act_sin, const void* txt_cos, const void* txt_sin,
    const void* rstd_p,
    void* d_act_qkv, void* d_txt_qkv,
    void* d_nq_f32, void* d_nk_f32, void* d_naq_f32, void* d_nak_f32,
    int B, int S_act, int S_txt, int H, cudaStream_t stream
) {
    int total_tok = B * (S_act + S_txt);
    dim3 grid(min(total_tok, 176), (3 * H + 31) / 32);
    mmdit_joint_qkv_norm_rope_48_bwd_vec8_k<<<grid, 256, 0, stream>>>(
        (const __nv_bfloat16*)dj_q, (const __nv_bfloat16*)dj_k, (const __nv_bfloat16*)dj_v,
        (const __nv_bfloat16*)act_qkv, (const __nv_bfloat16*)txt_qkv,
        (const __nv_bfloat16*)nq_w, (const __nv_bfloat16*)nk_w, (const __nv_bfloat16*)naq_w, (const __nv_bfloat16*)nak_w,
        (const float*)act_cos, (const float*)act_sin, (const float*)txt_cos, (const float*)txt_sin,
        (const float*)rstd_p,
        (__nv_bfloat16*)d_act_qkv, (__nv_bfloat16*)d_txt_qkv,
        (float*)d_nq_f32, (float*)d_nk_f32, (float*)d_naq_f32, (float*)d_nak_f32,
        B, S_act, S_txt, H);
}

void launch_mmdit_split_48_fwd(const void* joint_out, void* act_out, void* txt_out, int B, int S_act, int S_txt, cudaStream_t stream) {
    mmdit_split_48_fwd_k<<<B * (S_act + S_txt), 192, 0, stream>>>(
        (const __nv_bfloat16*)joint_out, (__nv_bfloat16*)act_out, (__nv_bfloat16*)txt_out, B, S_act, S_txt);
}

void launch_mmdit_split_48_bwd(const void* d_act_out, const void* d_txt_out, void* d_joint_out, int B, int S_act, int S_txt, cudaStream_t stream) {
    mmdit_split_48_bwd_k<<<B * (S_act + S_txt), 192, 0, stream>>>(
        (const __nv_bfloat16*)d_act_out, (const __nv_bfloat16*)d_txt_out, (__nv_bfloat16*)d_joint_out, B, S_act, S_txt);
}

void launch_mmdit_split_48_bwd_bhsd(const void* d_act_out, const void* d_txt_out, void* dj_bh_out, int B, int S_act, int S_txt, cudaStream_t stream) {
    mmdit_split_48_bwd_bhsd_k<<<B * (S_act + S_txt), 192, 0, stream>>>(
        (const __nv_bfloat16*)d_act_out, (const __nv_bfloat16*)d_txt_out, (__nv_bfloat16*)dj_bh_out, B, S_act, S_txt);
}

void launch_mmdit_split_quant_sub(
    const void* j_out, void* y_row, void* y_col,
    const void* curr_scale, const void* curr_inv, void* next_amax,
    int slot_id, int B, int S_sub, int S_tot, int s_offset,
    int layout_mode, int dual, int warmed_up, cudaStream_t stream
) {
    const int M = B * S_sub;
    dim3 grid(24, (M + 63) / 64);
    const bool do_dual = dual && ((M & 3) == 0);
    if (!warmed_up) {
        int blocks = min(1024, (M * 384 + 255) / 256);
        if (layout_mode == 0)      mmdit_split_quant_amax_k<0><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
        else if (layout_mode == 1) mmdit_split_quant_amax_k<1><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
        else                       mmdit_split_quant_amax_k<2><<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
        calib_scale_slot_k<<<1, 1, 0, stream>>>((const int32_t*)next_amax, (float*)curr_scale, (float*)curr_inv, slot_id, 448.0f);
        if (do_dual) {
            if (layout_mode == 0)      mmdit_split_quant_64x64_k<0, true, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, (uint8_t*)y_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            else if (layout_mode == 1) mmdit_split_quant_64x64_k<1, true, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, (uint8_t*)y_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            else                       mmdit_split_quant_64x64_k<2, true, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, (uint8_t*)y_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
        } else {
            if (layout_mode == 0)      mmdit_split_quant_64x64_k<0, false, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            else if (layout_mode == 1) mmdit_split_quant_64x64_k<1, false, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            else                       mmdit_split_quant_64x64_k<2, false, false><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            if (dual && y_col != nullptr) launch_transpose_fp8(y_row, y_col, M, 1536, stream);
        }
    } else {
        if (do_dual) {
            if (layout_mode == 0)      mmdit_split_quant_64x64_k<0, true, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, (uint8_t*)y_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            else if (layout_mode == 1) mmdit_split_quant_64x64_k<1, true, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, (uint8_t*)y_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            else                       mmdit_split_quant_64x64_k<2, true, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, (uint8_t*)y_col, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
        } else {
            if (layout_mode == 0)      mmdit_split_quant_64x64_k<0, false, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            else if (layout_mode == 1) mmdit_split_quant_64x64_k<1, false, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            else                       mmdit_split_quant_64x64_k<2, false, true><<<grid, 256, 0, stream>>>((const __nv_bfloat16*)j_out, (uint8_t*)y_row, nullptr, (const float*)curr_inv, (int32_t*)next_amax, slot_id, B, S_sub, S_tot, s_offset);
            if (dual && y_col != nullptr) launch_transpose_fp8(y_row, y_col, M, 1536, stream);
        }
    }
}

void lt_set_autotune_enabled(int enabled) {
    g_lt_autotune_enabled = enabled;
}

int lt_serialized_entry_bytes() {
    return (int)sizeof(LtSerializedAlgo);
}

int lt_export_algos(void* out_buf, int max_entries) {
    LtSerializedAlgo* dst = reinterpret_cast<LtSerializedAlgo*>(out_buf);
    int n = 0;
    for (const auto& kv : g_lt_cache) {
        if (!kv.second.valid) continue;
        if (n >= max_entries) break;
        LtSerializedAlgo e;
        e.key = kv.first;
        e.key.dev = 0;
        e.algo = kv.second.bestAlgo;
        dst[n++] = e;
    }
    return n;
}

void lt_import_algos(const void* in_buf, int count, size_t workspace_bytes) {
    int dev = 0;
    cudaGetDevice(&dev);
    cublasLtHandle_t lt = get_lt_handle(dev);
    const LtSerializedAlgo* src = reinterpret_cast<const LtSerializedAlgo*>(in_buf);
    for (int i = 0; i < count; ++i) {
        LtKey k0 = src[i].key;
        k0.dev = 0;
        g_lt_imported_algos[k0] = src[i].algo;

        LtKey kd = src[i].key;
        kd.dev = dev;
        auto it = g_lt_cache.find(kd);
        if (it != g_lt_cache.end() && it->second.valid) {
            cublasLtMatmulHeuristicResult_t chk{};
            if (cublasLtMatmulAlgoCheck(
                    lt, it->second.opDesc, it->second.aDesc, it->second.bDesc,
                    it->second.cDesc, it->second.dDesc, &src[i].algo, &chk) == CUBLAS_STATUS_SUCCESS
                && chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= workspace_bytes) {
                it->second.bestAlgo = src[i].algo;
            }
        }
    }
}

void launch_bf16_sum_sq_f64(const void* g, void* out_sum_sq_f64, int64_t N, int zero_first, cudaStream_t stream) {
    if (zero_first) cudaMemsetAsync(out_sum_sq_f64, 0, sizeof(double), stream);
    if (N <= 0) return;
    int blocks = (int)min((int64_t)528, (N + 2047) / 2048);
    bf16_sum_sq_f64_k<<<blocks, 256, 0, stream>>>((const __nv_bfloat16*)g, (double*)out_sum_sq_f64, N);
}

void launch_compute_adamw_inv_scale(
    const void* norm_sq_f64, void* global_norm_f32, void* inv_scale_f32,
    float loss_scale, float clip_grad, float dp_ws, int use_deferred_div, cudaStream_t stream
) {
    compute_adamw_inv_scale_k<<<1, 1, 0, stream>>>(
        (const double*)norm_sq_f64, (float*)global_norm_f32, (float*)inv_scale_f32,
        loss_scale, clip_grad, dp_ws, use_deferred_div);
}

void launch_fused_adamw_bf16_io(
    void* fp32_p, void* fp32_m, void* fp32_v,
    const void* bf16_g, void* bf16_p_out, const void* inv_scale_f32,
    float lr, float beta1, float beta2, float step_size, float inv_bc2_sqrt,
    float eps, float decay_factor, int64_t N, cudaStream_t stream
) {
    if (N <= 0) return;
    int blocks = (int)min((int64_t)1056, (N + 1023) / 1024);
    fused_adamw_fp32_master_bf16_io_k<<<blocks, 256, 0, stream>>>(
        (float*)fp32_p, (float*)fp32_m, (float*)fp32_v,
        (const __nv_bfloat16*)bf16_g, (__nv_bfloat16*)bf16_p_out, (const float*)inv_scale_f32,
        lr, beta1, beta2, step_size, inv_bc2_sqrt, eps, decay_factor, N);
}

void launch_cast_f32_to_bf16(const void* src, void* dst, int n, cudaStream_t stream) {
    cast_f32_to_bf16_k<<<(n + 255) / 256, 256, 0, stream>>>((const float*)src, (__nv_bfloat16*)dst, n);
}

void launch_mmdit_joint_qkv_48_bwd_e5_dual(
    const void* dj_q, const void* dj_k, const void* dj_v,
    const void* act_qkv, const void* txt_qkv,
    const void* nq_w, const void* nk_w, const void* naq_w, const void* nak_w,
    const void* act_cos, const void* act_sin, const void* txt_cos, const void* txt_sin,
    const void* rstd_p,
    void* da_row, void* da_t, void* da_db_bf16, void* da_db_f32,
    void* dt_row, void* dt_t, void* dt_db_bf16, void* dt_db_f32,
    const void* curr_inv, void* next_amax, int slot_act, int slot_txt,
    void* d_nq_f32, void* d_nk_f32, void* d_naq_f32, void* d_nak_f32,
    int B, int S_act, int S_txt, int H, int bhsd, cudaStream_t stream
) {
    cudaMemsetAsync(da_db_f32, 0, 4608 * sizeof(float), stream);
    cudaMemsetAsync(dt_db_f32, 0, 4608 * sizeof(float), stream);
    int gy = min(max(1, (B * max(S_act, S_txt) + 63) / 64), 12);
    dim3 grid(3 * H, gy);
    if (bhsd) {
        mmdit_joint_qkv_48_bwd_pers64_e5_k<true><<<grid, 256, 0, stream>>>(
            (const __nv_bfloat16*)dj_q, (const __nv_bfloat16*)dj_k, (const __nv_bfloat16*)dj_v,
            (const __nv_bfloat16*)act_qkv, (const __nv_bfloat16*)txt_qkv,
            (const __nv_bfloat16*)nq_w, (const __nv_bfloat16*)nk_w, (const __nv_bfloat16*)naq_w, (const __nv_bfloat16*)nak_w,
            (const float*)act_cos, (const float*)act_sin, (const float*)txt_cos, (const float*)txt_sin,
            (const float*)rstd_p,
            (uint8_t*)da_row, (uint8_t*)da_t, (float*)da_db_f32,
            (uint8_t*)dt_row, (uint8_t*)dt_t, (float*)dt_db_f32,
            (const float*)curr_inv, (int32_t*)next_amax, slot_act, slot_txt,
            (float*)d_nq_f32, (float*)d_nk_f32, (float*)d_naq_f32, (float*)d_nak_f32,
            B, S_act, S_txt, H);
    } else {
        mmdit_joint_qkv_48_bwd_pers64_e5_k<false><<<grid, 256, 0, stream>>>(
            (const __nv_bfloat16*)dj_q, (const __nv_bfloat16*)dj_k, (const __nv_bfloat16*)dj_v,
            (const __nv_bfloat16*)act_qkv, (const __nv_bfloat16*)txt_qkv,
            (const __nv_bfloat16*)nq_w, (const __nv_bfloat16*)nk_w, (const __nv_bfloat16*)naq_w, (const __nv_bfloat16*)nak_w,
            (const float*)act_cos, (const float*)act_sin, (const float*)txt_cos, (const float*)txt_sin,
            (const float*)rstd_p,
            (uint8_t*)da_row, (uint8_t*)da_t, (float*)da_db_f32,
            (uint8_t*)dt_row, (uint8_t*)dt_t, (float*)dt_db_f32,
            (const float*)curr_inv, (int32_t*)next_amax, slot_act, slot_txt,
            (float*)d_nq_f32, (float*)d_nk_f32, (float*)d_naq_f32, (float*)d_nak_f32,
            B, S_act, S_txt, H);
    }
    cast_f32_to_bf16_k<<<18, 256, 0, stream>>>((const float*)da_db_f32, (__nv_bfloat16*)da_db_bf16, 4608);
    cast_f32_to_bf16_k<<<18, 256, 0, stream>>>((const float*)dt_db_f32, (__nv_bfloat16*)dt_db_bf16, 4608);
}

void launch_scatter_rows_2560_bf16(void* dst_2d, const void* src_2d, const void* idx_i32, int N_rows, cudaStream_t stream) {
    if (N_rows <= 0) return;
    scatter_rows_2560_bf16_k<<<N_rows, 320, 0, stream>>>((__nv_bfloat16*)dst_2d, (const __nv_bfloat16*)src_2d, (const int32_t*)idx_i32, N_rows);
}

void launch_gather_and_zero_rows_2560_bf16(void* d_main_2d, void* d_src_out, const void* idx_i32, int N_rows, int zero_main, cudaStream_t stream) {
    if (N_rows <= 0) return;
    gather_and_zero_rows_2560_bf16_k<<<N_rows, 320, 0, stream>>>((__nv_bfloat16*)d_main_2d, (__nv_bfloat16*)d_src_out, (const int32_t*)idx_i32, N_rows, zero_main);
}

void launch_deepstack_add_2560_bf16(void* dst_2d, const void* vis_2d, const void* idx_i32, int N_rows, cudaStream_t stream) {
    if (N_rows <= 0) return;
    deepstack_add_2560_bf16_k<<<N_rows, 320, 0, stream>>>((__nv_bfloat16*)dst_2d, (const __nv_bfloat16*)vis_2d, (const int32_t*)idx_i32, N_rows);
}

void lt_clear_all_caches() {
    for (auto& kv : g_lt_cache) {
        cublasLtMatmulDescDestroy(kv.second.opDesc);
        cublasLtMatrixLayoutDestroy(kv.second.aDesc);
        cublasLtMatrixLayoutDestroy(kv.second.bDesc);
        cublasLtMatrixLayoutDestroy(kv.second.cDesc);
        cublasLtMatrixLayoutDestroy(kv.second.dDesc);
    }
    g_lt_cache.clear();
    g_lt_imported_algos.clear();
}

}  // extern "C"
