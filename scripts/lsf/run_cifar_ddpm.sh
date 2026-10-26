#!/usr/bin/env bash
set -euo pipefail
study="${1:?Expected absolute study directory}"
steps="${2:-250000}"
source_dir="${SOURCE_DIR:-$study/source}"
output="$study/training"
python_bin="${PYTHON_BIN:?Expected PYTHON_BIN}"
data_root="${DATA_ROOT:?Expected DATA_ROOT}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$output"
command=("$python_bin" -u "$source_dir/ddpm.py" --data-root "$data_root" --output "$output"
    --steps "$steps" --batch-size 128 --base-channels 128 --num-res-blocks 4
    --learning-rate .0002 --warmup 5000 --ema-decay .9999 --seed 42
    --checkpoint-every 1000 --preview-every 5000 --fid-samples 10000
    --sample-batch-size 64 --threads 4 --device cuda)
if [[ -f "$output/checkpoint_latest.pt" ]]; then
    command+=(--resume "$output/checkpoint_latest.pt")
fi
if [[ "${3:-}" == --dry-run ]]; then
    printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
write_status() {
    "$python_bin" - "$output" "$1" "$2" <<'PY'
import datetime, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
value = dict(stage='ddpm', status=sys.argv[2], exit_code=int(sys.argv[3]),
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
"$python_bin" - "$output" "$steps" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
assert (root / 'checkpoint_latest.pt').is_file()
fid = json.loads((root / 'fid_metrics.json').read_text())[-1]
assert fid['iteration'] == int(sys.argv[2]) and fid['generated_count'] == 10000
PY
