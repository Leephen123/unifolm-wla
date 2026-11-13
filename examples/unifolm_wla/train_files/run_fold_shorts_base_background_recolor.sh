#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "$repo"
if [[ -z "${SLURM_JOB_NODELIST:-}" || -z "${SLURM_NODEID:-}" || -z "${SLURM_NNODES:-}" ]]; then
    printf '%s\n' 'Run this launcher with one Slurm task per node.' >&2
    exit 2
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR="$(/soft/slurm/bin/scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')"
fi
exec "${PYTHON_BIN:-python}" -m torch.distributed.run \
    --nnodes "$SLURM_NNODES" --nproc-per-node "${GPUS_PER_NODE:-8}" --node-rank "$SLURM_NODEID" \
    --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29791}" \
    -m examples.unifolm_wla.train_full_pipeline \
    --config "$repo/unifolm_wla/config/training/fold_shorts_base_background_recolor.yaml" "$@"
