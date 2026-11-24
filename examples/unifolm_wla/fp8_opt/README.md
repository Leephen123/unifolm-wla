# UnifoLM-WLA SM90 Native FP8 Training Package (`fp8_opt`)

Production-refactored SM90 (`NVIDIA H100 / H800`) FP8 + BF16 hybrid training acceleration package for `UnifoLM-WLA` (`Qwen3-VL-4B` [4.44B] + `DiT-L` MMDiT Action Expert [1.37B], `5.838B` trainable BF16 parameters = `11.68 GB` weights).

---

## 1. Core Architectural Guarantees

1. **100% Full Gradient Checkpointing (`ZERO SAC`)**:
   - Retains full activation checkpointing across all `24` Vision blocks, `36` Qwen3-VL Decoder layers, and `16` MMDiT Joint-Attention blocks, preserving >22 GiB HBM3 headroom at `BS=64` (`52.48 / 57.74 GiB`) and scaling cleanly up to **`max_bs = 128` per GPU (`1,024` global batch size on `8 × H100 80GB`, `73.93 / 75.82 GiB`)**.
2. **Strict Mixed-Precision Boundaries**:
   - **BF16**: Vision Tower (`24` blocks), `robot_state_projector`, Action Encoder/Decoder, Timestep/AdaLN Modulations, QK/RMS/LayerNorms, RoPE, and FlashAttention-4 / cuDNN 9.12 SDPA.
   - **FP8 (`E4M3` forward / `E5M2` backward, 1-step delayed scaling with Step-1 exact calibration)**: Language (`36` layers) and MMDiT (`16` blocks) QKV/Out/MLP Linears (`444 -> 269` horizontally fused `StepCachedFP8Linear` modules). Master weights and AdamW moments remain FP32 (`clip_grad = 1.0`).
3. **Lossless `1,256`-Key Canonical Checkpoint Compatibility**:
   - `model.state_dict()` splits fused projections into the original 1,256 keys. The patched model accepts these keys via `load_state_dict(..., strict=True)`. With `strict=False`, missing slices keep their current weights and are reported by canonical name.
   - Before rank-0-only export, all ranks call `flush_pending_weight_gather(model)`. Calling `state_dict()` with a pending gather raises immediately. All-rank calls to `accelerator.get_state_dict(model)`, `engine.save_checkpoint(...)`, and `engine.save_16bit_model(...)` flush automatically.

---

## 2. Verified `8 × NVIDIA H100 80GB` Performance & `max_bs` Scaling (`nv-h100-009`, `ZeRO-2`, `100% Full GC`, `ZERO SAC`)

Dual utilization metrics are reported side-by-side (`seq_txt_valid = seq_txt_exec = 587`, `seq_act = 50`):
- **Standard PaLM MFU (`3.0x`, Chowdhery et al., 2022)**: Counts only necessary non-recomputed model FLOPs (`1,201.51 TFLOPs/GPU` at `BS=64`).
- **Strictly Executed HFU**: Counts strictly executed Tensor Core FLOPs after removing all 60 block-final dead recompute GEMMs (`1,495.96 TFLOPs/GPU` at `BS=64`).

| Configuration | BS / GPU (8-GPU Global) | Steady Step P50 (Min) | Samples/s/GPU (8-GPU) | Peak Alloc / Res | PaLM FP8 MFU (`3.0x`) | PaLM Mixed MFU | Executed FP8 HFU (Mixed) | Grad Rel L2 vs BF16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_fp8_fa4_bs32` | 32 (256) | `1.6828 s` | `19.02` (`152.1`) | `32.48 / 41.03 GiB` | `18.06%` | `22.10%` | `23.91%` (`29.22%`) | `0.29738` |
| `baseline_fp8_fa4_bs64` | 64 (512) | `3.1805 s` | `20.12` (`161.0`) | `44.90 / 48.10 GiB` | `19.10%` | `23.41%` | `25.31%` (`30.92%`) | `0.29738` |
| **`opt_v15_sub2s_bs64` (`BS=64`, Core Target)** | **64 (512)** | **`1.9981 s` (`1.9889 s`)** | **`32.03` (`256.24`)** | **`52.48 / 57.74 GiB`** | **`30.38%`** | **`37.26%`** | **`37.83%` (`46.18%`)** | **`0.29735`** |
| `opt_final_refactored_bs64` (`fp8_opt`, hot-GPU) | 64 (512) | `2.0021 s` (`2.0001 s`) | `31.97` (`255.73`) | `52.48 / 57.73 GiB` | `30.32%` | `37.18%` | `37.76%` (`46.09%`) | **`0.29577`** |
| `fp8_opt` (`BS=96`) | 96 (768) | `2.9983 s` | `32.02` (`256.14`) | `63.20 / 70.00 GiB` | `30.37%` | `37.24%` | `37.82%` (`46.16%`) | — |
| `fp8_opt` (`BS=112`) | 112 (896) | `3.4765 s` | `32.22` (`257.73`) | `68.57 / 74.73 GiB` | `30.56%` | `37.47%` | `38.05%` (`46.45%`) | — |
| **`fp8_opt` (`max_bs = 128`)** | **128 (1,024)** | **`3.8986 s`** | **`32.83` (`262.66`)** | **`73.93 / 75.82 GiB`** | **`31.15%`** | **`38.19%`** | **`38.78%` (`47.33%`)** | — |

> `BS = 132+` exceeds `80 GiB` HBM3 capacity; `per_device_batch_size = 128` (`global_batch = 1,024`) is the exact maximum stable batch size under the identical Full-GC configuration.

---

## 3. Modular Package Architecture

```text
fp8_opt/
├── __init__.py            # Public API exports
├── csrc/
│   └── wla_sm90_kernels.cu # Zero-spill SM90 CUDA C++ & CUDA 12.8 cublasLt kernel suite
├── cuda_ops.py            # Shared library builder/loader, 128 MiB DelayedScalingPool, scatter/gather & typed ops
├── delayed_scaling.py     # StepCachedFP8Linear, horizontal QKV/GateUp fusion, async weight cache & lockstep algo sync
├── vision_tower.py        # BF16 Vision Tower: _VisionBlockCkptFn, persistent QKV RoPE+dbias bwd, 4-warp LN(1024)
├── language_backbone.py   # Qwen3-VL 36L: _QwenLayerCkptFn, pers64 dual RMSNorm/QKV bwd, zero-DtoH DeepStack
├── mmdit_expert.py        # MMDiT 16L: _MMDiTBlockCkptFn, zero-copy BHSD pers64 dual E5M2 QKV+dbias bwd, dead-branch prune
├── zero2_hooks.py         # ZeRO-2 NVLS SUM, zero-copy Direct IPG dW, flat_part_grad, two-stage async weight AG, fused AdamW
├── data_collator.py       # Zero-DtoH FastWLAInputsCollator, GPU uint8 patch unpack, Vision-first forward scheduling
├── installer.py           # WLAFP8Config, install_wla_fp8_training, verify_state_dict_roundtrip (1,256 canonical keys)
├── verify_package.py      # Unit ops, bidirectional 1,256-key round-trip, and gradient equivalence
└── verify_distributed_zero2.py  # 8-GPU accumulation, collective export, FP32 communication, and resume
```

---

## 4. Integration Quickstart

```python
import torch

from examples.unifolm_wla.fp8_opt import (
    FastWLAInputsCollator,
    cache_all_fp8_weights,
    clear_all_fp8_weights,
    flush_pending_weight_gather,
    install_fast_zero2_hooks,
    install_fp8_training_optimizations,
    install_zero_sync_forward,
)

# 1. Install SM90 FP8 Linear fusion + Full-GC block runners + 1,256-key canonical state_dict hook
install_fp8_training_optimizations(
    model,
    bwd_fast_accum=True,
    use_cudnn_qwen_sdpa=True,
    skip_last_txt_mlp=True,
    compile_blocks=True,
)
install_zero_sync_forward(model)

# 2. Build DataLoader with Zero-DtoH tight-sequence collator & prepare with Accelerate/DeepSpeed ZeRO-2
collator = FastWLAInputsCollator(cfg, tight_seq=True, gpu_patch_unpack=True)
model, optimizer, loader = accelerator.prepare(model, optimizer, build_dataloader(cfg, collate_fn=collator))
bare_model = accelerator.unwrap_model(model)
install_fast_zero2_hooks(model)

# 3. Training step
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    cache_all_fp8_weights(bare_model)
    with accelerator.autocast():
        loss = model(batch)["action_loss"]
    accelerator.backward(loss)
    optimizer.step()
    clear_all_fp8_weights(bare_model)
```

For a rank-0-only canonical export, enter the gather on every rank.

```python
flush_pending_weight_gather(model)
if accelerator.is_main_process:
    torch.save(bare_model.state_dict(), "canonical.pt")
```

Alternatively, call `accelerator.get_state_dict(model)` on every rank before writing on rank 0. For training checkpoints, call `engine.save_checkpoint(...)` on every rank. Install the FP8 and ZeRO hooks on the fresh engine in the order shown above before resuming with `engine.load_checkpoint(..., load_module_strict=True)` or `accelerator.load_state(...)`.

Run the distributed checks on one eight-GPU node from the repository root.

```bash
torchrun --nproc_per_node=8 -m examples.unifolm_wla.fp8_opt.verify_distributed_zero2
```
