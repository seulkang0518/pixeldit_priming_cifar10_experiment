#!/usr/bin/env bash
# qsub scripts/cs_cluster/run_cifar_official_b_h100.sh
# CIFAR pixel-space adaptations of the released B-model hyperparameters.
#$ -S /bin/bash
#$ -N cifar_official_b
#$ -P aihub_ucl
#$ -cwd
#$ -t 1-2
#$ -pe gpu 1
#$ -l tmem=32G,h_rt=48:00:00,gpu=true,gpu_type=h100
#$ -j y
#$ -o /home/zongchen/
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
DATA_ROOT="${DATA_ROOT:-/SAN/intelsys/imagenet_mmd_flow/data}"
case "${SGE_TASK_ID:?Expected array task 1 or 2}" in
    1) OBJECTIVE=drifting ;;
    2) OBJECTIVE=sinkhorn ;;
    *) exit 2 ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zongchen/mmd_flow_hudson_new/results/cifar}"
OUTPUT="$OUTPUT_ROOT/officialB_cifar_${JOB_ID:-dryrun}_${SGE_TASK_ID}_${OBJECTIVE}"
SOURCE="$OUTPUT/source"
COMMAND=("$PYTHON" -u "$SOURCE/main.py"
    --dataset cifar10 --data-root "$DATA_ROOT" --target-size 0 --target-batch-size 64
    --particles 64 --steps 200000 --learning-rate 0.0004
    --optimizer AdamW --adam-beta2 0.95 --weight-decay 0 --warmup-steps 10000 --max-grad-norm 2
    --ema-decay 0.999 --seed 42
    --architecture unet --base-channels 128 --num-res-blocks 4 --latent-dim 3072
    --objective "$OBJECTIVE" --tau-list 0.2 0.05 0.02 --normalize-per-temperature
    --sinkhorn-epsilon 0.05 --sinkhorn-iters 10 --field-step 1
    --multiscale-loss --coarse-sizes 8 16 --patch-sizes 7 14 --crops-per-size 1
    --global-loss-weight 1 --coarse-loss-weight 1 --patch-loss-weight 1
    --fid-samples 10000 --fid-batch-size 64 --save-every 5000 --checkpoint-every 2000
    --threads 4 --device cuda --trajectory-particles 8 --output "$OUTPUT")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0
fi
[[ $# == 0 && -x "$PYTHON" && -f "$PROJECT_ROOT/main.py" && -d "$PROJECT_ROOT/assets" ]]
[[ ! -e "$OUTPUT" ]]
mkdir -p "$SOURCE"
cp "$PROJECT_ROOT/"*.py "$SOURCE/"
cp "$PROJECT_ROOT/"*LICENSE* "$SOURCE/"
cp -r "$PROJECT_ROOT/configs" "$SOURCE/"
ln -s "$PROJECT_ROOT/assets" "$SOURCE/assets"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 DRIFT_COMPILE=0
export EXPERIMENT_OUTPUT="$OUTPUT" EXPERIMENT_OBJECTIVE="$OBJECTIVE"
write_status() {
    "$PYTHON" - "$1" "$2" <<'PY'
import datetime, json, os, pathlib, sys
p = pathlib.Path(os.environ['EXPERIMENT_OUTPUT']) / 'job_status.json'
value = dict(status=sys.argv[1], exit_code=int(sys.argv[2]),
             job_id=os.environ['JOB_ID'], task_id=int(os.environ['SGE_TASK_ID']),
             objective=os.environ['EXPERIMENT_OBJECTIVE'], hostname=os.uname().nodename,
             updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
temporary = p.with_suffix('.tmp')
temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(p)
PY
}
finish() {
    code=$?
    trap - EXIT
    set +e
    if [[ "$code" == 0 ]]; then
        "$PYTHON" - <<'PY'
import csv, json, os, pathlib
root = pathlib.Path(os.environ['EXPERIMENT_OUTPUT'])
config = json.loads((root / 'config.json').read_text())
rows = list(csv.DictReader((root / 'metrics.csv').open()))
fid = json.loads((root / 'fid_metrics.json').read_text())
assert config['total_updates'] == 200000 and config['actual_target_size'] == 50000
assert int(rows[-1]['update']) == 200000 and len(rows) == 41
assert len(fid) == 41 and fid[-1]['iteration'] == 200000
assert (root / 'checkpoint_latest.pt').is_file()
PY
        code=$?
    fi
    if [[ "$code" == 0 ]]; then write_status complete 0; else write_status failed "$code"; fi
    exit "$code"
}
trap finish EXIT
write_status running 0
"$PYTHON" - "$SOURCE" <<'PY'
import hashlib, json, pathlib, sys, torch
assert torch.cuda.is_available(), 'CUDA unavailable'
name = torch.cuda.get_device_name(0)
print('Allocated GPU:', name, flush=True)
assert 'H100' in name, name
root = pathlib.Path(sys.argv[1])
(root / 'sha256.json').write_text(json.dumps(
    {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}, indent=2) + '\n')
PY
printf '%q ' "${COMMAND[@]}"; printf '\n'
"${COMMAND[@]}" 2>&1 | tee "$OUTPUT/run.log"
