#!/usr/bin/env bash
# Run from an immutable source snapshot; submit via submit_cifar_priming.py.
set -euo pipefail
stage="${1:?Expected reconstruction, posttraining, or baseline}"
study="${2:?Expected absolute study directory}"
source_dir="${SOURCE_DIR:-$study/source}"
python_bin="${PYTHON_BIN:-$study/../../../.venv/bin/python}"
data_root="${DATA_ROOT:-$study/../../../data}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 DRIFT_COMPILE=0
output="$study/$stage"
mkdir -p "$output"
common=(--dataset cifar10 --data-root "$data_root" --target-size 0
    --particles 64 --learning-rate 0.0002 --optimizer Adam --adam-beta2 0.999
    --seed 42 --architecture unet --base-channels 128 --num-res-blocks 4
    --latent-dim 3072 --objective softmax-riesz --threads 4 --device cuda
    --fid-samples 10000 --fid-batch-size 64 --trajectory-particles 8 --output "$output")
case "$stage" in
    reconstruction)
        stage_args=(--training-stage reconstruction --corruption-level 0.8
            --epochs 80 --target-batch-size 64 --ema-decay 0.9999
            --save-every 20 --checkpoint-every 1) ;;
    posttraining|baseline)
        stage_args=(--steps 250000 --target-batch-size 256 --ema-decay 0.999
            --multiscale-loss --coarse-sizes 8 16 --patch-sizes 7 14
            --save-every 2500 --checkpoint-every 1000)
        if [[ "$stage" == posttraining ]]; then
            stage_args+=(--init-checkpoint "$study/reconstruction/checkpoint_latest.pt")
        fi ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
if [[ -f "$output/checkpoint_latest.pt" ]]; then
    stage_args+=(--resume "$output/checkpoint_latest.pt")
fi
command=("$python_bin" -u "$source_dir/main.py" "${common[@]}" "${stage_args[@]}")
if [[ "${3:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
write_status() {
    "$python_bin" - "$output" "$stage" "$1" "$2" <<'PY'
import datetime, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
value = dict(stage=sys.argv[2], status=sys.argv[3], exit_code=int(sys.argv[4]),
             job_id=os.environ.get('LSB_JOBID'), host=os.uname().nodename,
             updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
p = root / 'job_status.json.tmp'
p.write_text(json.dumps(value, indent=2) + '\n')
p.replace(root / 'job_status.json')
PY
}
finish() {
    code=$?
    trap - EXIT
    if [[ "$code" == 0 ]]; then write_status complete 0; else write_status failed "$code"; fi
    exit "$code"
}
trap finish EXIT
write_status running 0
"$python_bin" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), flush=True)'
printf '%q ' "${command[@]}"; printf '\n'
"${command[@]}" 2>&1 | tee -a "$output/run.log"
"$python_bin" - "$output" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
config = json.loads((root / 'config.json').read_text())
rows = list(csv.DictReader((root / 'metrics.csv').open()))
assert int(rows[-1]['update']) == config['total_updates']
assert (root / 'checkpoint_latest.pt').is_file()
assert json.loads((root / 'fid_metrics.json').read_text())[-1]['iteration'] == config['snapshot_iterations'][-1]
PY
