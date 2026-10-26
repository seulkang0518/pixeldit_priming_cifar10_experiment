#!/usr/bin/env bash
# Submit from the project root: qsub -P aihub_ucl scripts/cs_cluster/run_cifar_h100.sh
#$ -S /bin/bash
#$ -N cifar_ddpmpp_ema
#$ -P aihub_ucl
#$ -cwd
#$ -pe gpu 1
#$ -l tmem=32G,h_rt=48:00:00,gpu=true,gpu_type=h100
#$ -j y
#$ -o /home/zongchen/
set -euo pipefail
[[ -f "$PWD/main.py" && -f "$PWD/run.sh" ]] || { echo 'Submit from the project root containing main.py and run.sh.' >&2; exit 1; }
export PYTHON="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow/bin/python}"
export DATA_ROOT="${DATA_ROOT:-/SAN/intelsys/imagenet_mmd_flow/data}"
MULTISCALE_LOSS="${MULTISCALE_LOSS:-0}"
export MULTISCALE_LOSS
LOSS_NAME=""
if [[ "$MULTISCALE_LOSS" == 1 ]]; then LOSS_NAME="_multiscale"; fi
export OUTPUT="$PWD/results/cifar/h100_ddpmpp_ema0999${LOSS_NAME}_${JOB_ID}"
export EMA_DECAY=0.999 EPOCHS=2000 SAVE_EVERY=20 PARTICLES=64 BATCH_SIZE=256
export DATASET=cifar10 ARCHITECTURE=unet OBJECTIVE=softmax-riesz
export BASE_CHANNELS=128 NUM_RES_BLOCKS=4 LATENT_DIM=3072 LEARNING_RATE=0.0002 SEED=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT"
write_status() {
    "$PYTHON" - "$1" "$2" <<'PY'
import json,os,pathlib,sys,datetime
p=pathlib.Path(os.environ['OUTPUT'])/'job_status.json'
value=dict(status=sys.argv[1],exit_code=int(sys.argv[2]),job_id=os.environ['JOB_ID'],
           hostname=os.uname().nodename,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)
PY
}
finish() {
    code=$?
    set +e
    trap - EXIT
    if [[ "$code" == 0 ]]; then
        "$PYTHON" - <<'PY'
import csv,hashlib,json,os,pathlib
r=pathlib.Path(os.environ['OUTPUT'])
config=json.loads((r/'config.json').read_text());rows=list(csv.DictReader((r/'metrics.csv').open()))
assert int(rows[-1]['epoch'])==2000 and int(rows[-1]['update'])==392000
fid=json.loads((r/'fid_metrics.json').read_text());assert len(fid)==101 and fid[-1]['iteration']==2000
assert config['ema_decay']==0.999 and config['evaluation_weights']=='ema'
manifest={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
          for p in r.rglob('*') if p.is_file() and p.name not in ('job_status.json','artifact_manifest.json')}
(r/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
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
assert torch.cuda.is_available()
name=torch.cuda.get_device_name(0);print('Allocated GPU:',name,flush=True)
assert 'H100' in name, name
PY
bash run.sh
