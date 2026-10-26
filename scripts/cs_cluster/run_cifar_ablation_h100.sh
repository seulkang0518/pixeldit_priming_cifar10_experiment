#!/usr/bin/env bash
# From the project root: qsub scripts/cs_cluster/run_cifar_ablation_h100.sh
# All eight tasks are eligible to run concurrently, one H100 per task.
#$ -S /bin/bash
#$ -N cifar_ablation
#$ -P aihub_ucl
#$ -cwd
#$ -t 1-8
#$ -pe gpu 1
#$ -l tmem=32G,h_rt=24:00:00,gpu=true,gpu_type=h100
#$ -j y
#$ -o /home/zongchen/
set -euo pipefail
# SGE executes a spooled copy of this script; -cwd preserves the submission root.
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
DATA_ROOT="${DATA_ROOT:-/SAN/intelsys/imagenet_mmd_flow/data}"
TASK_ID="${SGE_TASK_ID:?Submit with qsub (or set SGE_TASK_ID=1..8 for --dry-run)}"
# Keep this nonempty for the cluster's older Bash with `set -u`.
EXTRA_ARGS=(--field-step 1)
case "$TASK_ID" in
    1) VARIANT=sinkhorn; EXTRA_ARGS=(--objective sinkhorn) ;;
    2) VARIANT=lr_1e4; EXTRA_ARGS=(--learning-rate 0.0001) ;;
    3) VARIANT=lr_5e5; EXTRA_ARGS=(--learning-rate 0.00005) ;;
    4) VARIANT=particles128; EXTRA_ARGS=(--particles 128) ;;
    5) VARIANT=global4; EXTRA_ARGS=(--global-loss-weight 4) ;;
    6) VARIANT=no_patch; EXTRA_ARGS=(--patch-loss-weight 0) ;;
    7) VARIANT=noise; EXTRA_ARGS=(--noise-levels 0 0.05 0.15) ;;
    8) VARIANT=explicit_h01; EXTRA_ARGS=(--field-step 0.1) ;;
    *) echo "Unknown task ID: $TASK_ID" >&2; exit 2 ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/cifar}"
OUTPUT="$OUTPUT_ROOT/ablation500_${JOB_ID:-dryrun}_${TASK_ID}_${VARIANT}"
SOURCE="$OUTPUT/source"
COMMAND=("$PYTHON" -u "$SOURCE/main.py"
    --dataset cifar10 --data-root "$DATA_ROOT" --target-size 0 --target-batch-size 256
    --particles 64 --epochs 500 --learning-rate 0.0002 --ema-decay 0.999
    --architecture unet --base-channels 128 --num-res-blocks 4 --latent-dim 3072
    --objective softmax-riesz --tau-list 0.2 0.05 0.02 --normalize-per-temperature
    --multiscale-loss --coarse-sizes 8 16 --patch-sizes 7 14 --crops-per-size 1
    --global-loss-weight 1 --coarse-loss-weight 1 --patch-loss-weight 1
    --field-step 1 --noise-levels 0
    --fid-samples 10000 --fid-batch-size 64 --save-every 20 --checkpoint-every 20
    --seed 0 --threads 4 --device cuda --trajectory-particles 8
    "${EXTRA_ARGS[@]}" --output "$OUTPUT")
if [[ "${1:-}" == --dry-run ]]; then
    printf 'task=%s variant=%s\n' "$TASK_ID" "$VARIANT"
    printf '%q ' "${COMMAND[@]}"; printf '\n'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Only --dry-run is supported' >&2; exit 2; }
[[ -f "$PROJECT_ROOT/main.py" && -f "$PROJECT_ROOT/target_updates.py" && -f "$PROJECT_ROOT/loss.py" ]] || {
    echo 'Submit from the project root containing main.py, target_updates.py and loss.py' >&2; exit 2;
}
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT" >&2; exit 2; }
[[ -x "$PYTHON" && -d "$PROJECT_ROOT/assets" ]] || { echo 'Missing Python environment or assets' >&2; exit 2; }
mkdir -p "$SOURCE"
# Run an immutable per-task source copy; assets remain shared and read-only.
cp "$PROJECT_ROOT/"*.py "$SOURCE/"
cp "$PROJECT_ROOT/WFLOW_LICENSE.txt" "$PROJECT_ROOT/DRIFTING_LICENSE.txt" "$SOURCE/"
ln -s "$PROJECT_ROOT/assets" "$SOURCE/assets"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export EXPERIMENT_OUTPUT="$OUTPUT" EXPERIMENT_VARIANT="$VARIANT"
write_status() {
    "$PYTHON" - "$1" "$2" <<'PY'
import datetime, json, os, pathlib, sys
p = pathlib.Path(os.environ['EXPERIMENT_OUTPUT']) / 'job_status.json'
value = dict(status=sys.argv[1], exit_code=int(sys.argv[2]),
             job_id=os.environ['JOB_ID'], task_id=int(os.environ['SGE_TASK_ID']),
             variant=os.environ['EXPERIMENT_VARIANT'], hostname=os.uname().nodename,
             updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
temp = p.with_suffix('.tmp')
temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(p)
PY
}
finish() {
    code=$?
    trap - EXIT
    set +e
    if [[ "$code" == 0 ]]; then
        "$PYTHON" - <<'PY'
import csv, hashlib, json, os, pathlib
root = pathlib.Path(os.environ['EXPERIMENT_OUTPUT'])
config = json.loads((root / 'config.json').read_text())
rows = list(csv.DictReader((root / 'metrics.csv').open()))
fid = json.loads((root / 'fid_metrics.json').read_text())
assert config['epochs'] == 500 and config['actual_target_size'] == 50000
assert int(rows[-1]['epoch']) == 500 and int(rows[-1]['update']) == 98000
assert len(rows) == 501 and len(fid) == 26 and fid[-1]['iteration'] == 500
assert (root / 'checkpoint_latest.pt').is_file()
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
manifest = {p.relative_to(root).as_posix(): digest(p)
            for p in root.rglob('*') if p.is_file()
            and 'assets' not in p.relative_to(root).parts
            and p.name not in ('job_status.json', 'artifact_manifest.json')}
(root / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
        code=$?
    fi
    if [[ "$code" == 0 ]]; then write_status complete 0; else write_status failed "$code"; fi
    exit "$code"
}
trap finish EXIT
write_status running 0
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
name = torch.cuda.get_device_name(0)
print('Allocated GPU:', name, flush=True)
assert 'H100' in name, name
PY
"$PYTHON" - "$SOURCE" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
(root / 'sha256.json').write_text(json.dumps(
    {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}, indent=2) + '\n')
PY
printf 'task=%s variant=%s output=%s\n' "$TASK_ID" "$VARIANT" "$OUTPUT"
printf '%q ' "${COMMAND[@]}"; printf '\n'
"${COMMAND[@]}" 2>&1 | tee "$OUTPUT/run.log"
