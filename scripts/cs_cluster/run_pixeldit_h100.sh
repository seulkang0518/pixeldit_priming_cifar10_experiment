#!/usr/bin/env bash
# Submit from the frozen study source, with MANIFEST set to its absolute manifest.json.
#$ -S /bin/bash
#$ -N cifar_pixeldit
#$ -P aihub_ucl
#$ -cwd
#$ -pe gpu 4
#$ -l tmem=16G,h_rt=48:00:00,gpu=true,gpu_type=h100
#$ -j y
#$ -o /home/zongchen/
set -euo pipefail
: "${MANIFEST:?Pass -v MANIFEST=/absolute/study/manifest.json to qsub}"
export MANIFEST
PYTHON="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$PYTHON" - <<'PY'
import json, os, torch
from pathlib import Path
manifest = json.loads(Path(os.environ['MANIFEST']).read_text())
assert all('--data-parallel' in row['command'] for row in manifest['stages']), 'Prepare with --data-parallel for this four-GPU launcher'
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 4, f'Expected four allocated GPUs; got {torch.cuda.device_count()}'
for index in range(4):
    name = torch.cuda.get_device_name(index)
    print(index, name, flush=True)
    assert 'H100' in name, name
PY
worker="$(dirname "$MANIFEST")/source/scripts/cs_cluster/pixeldit_study.py"
extra=()
if [[ "${SMOKE_ONLY:-0}" == 1 ]]; then extra+=(--smoke-only); fi
exec "$PYTHON" -u "$worker" run --manifest "$MANIFEST" "${extra[@]}"
