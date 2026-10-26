#!/usr/bin/env bash
# Shared GPU launcher for all datasets, backbones, and Riesz objectives.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/zongchen/miniconda3/envs/mmd_flow_hudson/bin/python}"
DATASET="${DATASET:-cifar10}"
ARCHITECTURE="${ARCHITECTURE:-unet}"
OBJECTIVE="${OBJECTIVE:-softmax-riesz}"
case "$DATASET" in
    mnist)
        RESULT_GROUP="mnist"
        DEFAULT_EPOCHS=100
        BASE_CHANNELS="${BASE_CHANNELS:-32}"
        LEARNING_RATE="${LEARNING_RATE:-0.001}"
        IMAGE_DIM=784
        EMA_DECAY="${EMA_DECAY:-0}"
        ;;
    cifar10)
        RESULT_GROUP="cifar"
        DEFAULT_EPOCHS=2000
        BASE_CHANNELS="${BASE_CHANNELS:-128}"
        LEARNING_RATE="${LEARNING_RATE:-0.0002}"
        IMAGE_DIM=3072
        EMA_DECAY="${EMA_DECAY:-0.999}"
        ;;
    *) printf 'Unknown DATASET: %s\n' "$DATASET" >&2; exit 1 ;;
esac
PARTICLES="${PARTICLES:-64}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
SAVE_EVERY="${SAVE_EVERY:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-4}"
SEED="${SEED:-0}"
DATA_ROOT="${DATA_ROOT:-/home/zongchen/data}"
MULTISCALE_LOSS="${MULTISCALE_LOSS:-0}"
LOSS_ARGS=()
LOSS_NAME=""
if [[ "$MULTISCALE_LOSS" == 1 ]]; then
    LOSS_ARGS+=(--multiscale-loss)
    LOSS_NAME="_multiscale"
elif [[ "$MULTISCALE_LOSS" != 0 ]]; then
    printf 'MULTISCALE_LOSS must be 0 or 1\n' >&2; exit 1
fi
if [[ "$ARCHITECTURE" == "unet" ]]; then
    LATENT_DIM="${LATENT_DIM:-$IMAGE_DIM}"
    ARCH_NAME="unet${BASE_CHANNELS}_noise${LATENT_DIM}"
    if [[ "$DATASET" == "cifar10" ]]; then
        ARCH_NAME="ddpmpp_${ARCH_NAME}_resblocks${NUM_RES_BLOCKS}"
    fi
elif [[ "$ARCHITECTURE" == "mlp" ]]; then
    LATENT_DIM="${LATENT_DIM:-32}"
    ARCH_NAME="mlp${LATENT_DIM}x128x256"
else
    printf 'Unknown ARCHITECTURE: %s\n' "$ARCHITECTURE" >&2; exit 1
fi
RUN_NAME="${DATASET}_${OBJECTIVE}${LOSS_NAME}_${ARCH_NAME}_resampled_fulltrain_particles${PARTICLES}_batch${BATCH_SIZE}_epochs${EPOCHS}_lr${LEARNING_RATE}_ema${EMA_DECAY}_seed${SEED}_gpu"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results/${RESULT_GROUP}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Python environment not found: %s\n' "$PYTHON" >&2; exit 1
fi
OUTPUT="$("$PYTHON" "$SCRIPT_DIR/run_storage.py" "$OUTPUT")"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
if [[ -e "$OUTPUT/config.json" ]]; then
    printf 'Existing run: %s; choose a fresh OUTPUT.\n' "$OUTPUT" >&2; exit 1
fi
mkdir -p -- "$OUTPUT"
"$PYTHON" -u "$SCRIPT_DIR/main.py" \
    --dataset "$DATASET" --data-root "$DATA_ROOT" \
    --target-size 0 --target-batch-size "$BATCH_SIZE" --particles "$PARTICLES" \
    --epochs "$EPOCHS" --learning-rate "$LEARNING_RATE" --objective "$OBJECTIVE" \
    --ema-decay "$EMA_DECAY" \
    --architecture "$ARCHITECTURE" --base-channels "$BASE_CHANNELS" \
    --num-res-blocks "$NUM_RES_BLOCKS" --latent-dim "$LATENT_DIM" \
    --seed "$SEED" --device cuda --save-every "$SAVE_EVERY" --trajectory-particles 8 \
    "${LOSS_ARGS[@]}" \
    --output "$OUTPUT" 2>&1 | tee "$OUTPUT/run.log"
