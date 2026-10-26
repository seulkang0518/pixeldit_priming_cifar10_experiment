#!/usr/bin/env bash
# Submit from the project root:
# qsub -P aihub_ucl scripts/cs_cluster/run_cifar_multiscale_h100.sh
#$ -S /bin/bash
#$ -N cifar_multiscale
#$ -P aihub_ucl
#$ -cwd
#$ -pe gpu 1
#$ -l tmem=32G,h_rt=48:00:00,gpu=true,gpu_type=h100
#$ -j y
#$ -o /home/zongchen/
set -euo pipefail
export MULTISCALE_LOSS=1
# Reuse the validated CIFAR/EMA recipe, status reporting and artifact validation.
exec bash "$PWD/scripts/cs_cluster/run_cifar_h100.sh"
