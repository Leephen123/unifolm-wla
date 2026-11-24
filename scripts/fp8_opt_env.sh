#!/bin/bash
# Source this script before launching train_full_pipeline.py with --fp8-opt on SM90 (H100/H800) nodes.
# Usage: source scripts/fp8_opt_env.sh

FP8_TE_DEPS="${FP8_TE_DEPS:-/storage/wangdonglinLab/weizhenyu/kuimouyu/research/unifoLM-WLA-fp8/.te-deps}"
VENV_SITE="${VENV_SITE:-/storage/wangdonglinLab/weizhenyu/kuimouyu/research/unifoLM-WLA/.venv/lib/python3.12/site-packages}"

if [ -d "$FP8_TE_DEPS/nvidia/cudnn/lib" ]; then
  export LD_LIBRARY_PATH="$FP8_TE_DEPS/nvidia/cudnn/lib:$FP8_TE_DEPS/transformer_engine/wheel_lib:${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$FP8_TE_DEPS/nvidia/cudnn/lib/libcudnn.so.9:$FP8_TE_DEPS/nvidia/cudnn/lib/libcudnn_adv.so.9:$FP8_TE_DEPS/nvidia/cudnn/lib/libcudnn_cnn.so.9:$FP8_TE_DEPS/nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9:$FP8_TE_DEPS/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.9:$FP8_TE_DEPS/nvidia/cudnn/lib/libcudnn_graph.so.9:$FP8_TE_DEPS/nvidia/cudnn/lib/libcudnn_heuristic.so.9:$FP8_TE_DEPS/nvidia/cudnn/lib/libcudnn_ops.so.9"
  export CUDNN_PATH="$FP8_TE_DEPS/nvidia/cudnn"
fi
export NVRTC_PATH="${NVRTC_PATH:-$VENV_SITE/nvidia/cuda_nvrtc}"
export CURAND_PATH="${CURAND_PATH:-$VENV_SITE/nvidia/curand}"

# SM90 fp8_opt optimal flags (verified ~2.00s/step @ BS=64 on 8xH100)
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export CUBLASLT_WORKSPACE_SIZE=131072
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
unset NCCL_PROTO || true
export DS_REDUCE_SCATTER=0
export DS_OVERLAP_COMM=1
export DS_BUCKET_SIZE=500000000
export WLA_FAST_ZERO2=1
export WLA_FUSED_ZERO2_ADAMW=1
export WLA_DIRECT_FLAT_PART=1
export WLA_DIRECT_IPG=1
export WLA_ASYNC_WEIGHT_AG=1
export WLA_AG_CTAS=0
export WLA_WARM_RETUNE=1
export WLA_ASYNC_DMA_IPG=0
export WLA_VISION_TAIL_FLUSH=1
export WLA_VISION_BUCKET_ELEMS=155000000
export WLA_ASYNC_WEIGHT_QUANT=1
export WLA_LT_BWD_SM_TARGET=0
export WLA_QWEN_CUDNN_SDPA=1
