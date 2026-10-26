# CIFAR-10: PixelDiT with Hudson's priming recipe

This experiment changes the **generator backbone** in
[Hudson's repository](https://github.com/hudsonchen/mmd_flow_hudson_new), pinned at
`12213c3d9889b441e520b6fe375fe001f48b9ce6`. It uses the original dual-path
[NVIDIA PixelDiT](https://github.com/NVlabs/PixelDiT) implementation, pinned at
`f7d7c35f71893d81c6e71d2576ea80601a1e8529`.

## What happens

1. **Reconstruction priming:** corrupt a real CIFAR image at fixed strength 0.8,
   predict its clean RGB pixels, and minimize the existing mean squared error.
2. **Riesz post-training:** generate images from Gaussian noise and train against
   real images with Hudson's existing multiscale pixel-space softmax-Riesz loss.
3. **Generation:** one generator call maps Gaussian noise to an RGB image.

The default recipe has no MAE or additional feature encoder, class conditioning,
time conditioning, MeanFlow objective, or velocity-denoising objective. The
upstream PixelDiT time and label inputs are fixed constants. Its output is passed
through sigmoid and flattened to retain Hudson's `[0,1]` image interface.
Upstream blocks and initialization are unchanged. The initial output is 0.5;
gradients reach the pathways as the zero output and modulation gates open.

The default is a **CIFAR adaptation of PixelDiT-B**, with patch width 768,
12 patch blocks, two pixel blocks, 12 attention heads, pixel width 16, and 4x4
patches. A 32x32 image has 64 patch tokens and 1,024 pixel tokens. Patch size 4
is a CIFAR adaptation; this is not the published ImageNet training configuration.

| Generator | Parameters |
| --- | ---: |
| Hudson compact DiT (384 wide, 12 blocks, 6 heads, patch 2) | 21,313,164 |
| CIFAR PixelDiT-B (configuration above) | 136,189,283 |

**This first comparison changes architecture and capacity together.** Report
model sizes and runtime alongside FID. It cannot establish a gain from the pixel
pathway alone; a subsequent study can control capacity or ablate that pathway.

## Recipe and comparison limits

The repository does not include the saved manifest for Hudson's reported FID
around 65. Without `--reference`, the launcher uses the documented recipe:

| Setting | Reconstruction | Post-training |
| --- | --- | --- |
| Duration | 80 epochs | 250,000 optimizer updates |
| Real images per update | 64 | 256 |
| Generated images per update | one corruption per real image | 128 interacting samples |
| Optimizer | Adam, betas 0.9/0.999 | same |
| Learning rate | constant 0.0002 | 1,000-update warmup, cosine 0.0002 to 0.0001 |
| EMA | 0.9999 | 0.9999 |
| Seed | 42 | 42 |
| Loss | MSE, no LPIPS by default | corrected softmax Riesz, no field RMS normalization |
| Pixel views | paired whole images | global/coarse/patch weights 2:1:1, four crops per size |
| Temperatures | — | 0.2, 0.05, 0.02 |
| FID | 10K samples every 20 epochs | 10K samples every 2,500 updates |

Post-training uses a fresh optimizer and transfers both online and EMA weights.
A recovery checkpoint is saved every 1,000 post-training updates; numbered
checkpoints are retained every 20,000. Changing checkpoint cadence affects I/O,
not the loss. FID retains Hudson's float-image pytorch-fid protocol and all 50,000
training images as its reference. It should not be compared directly to the
separate DMMD experiment's 5K-sample uint8 torch-fidelity scores.

If Hudson shares his `submit_compact_dit.py` **manifest.json**, pass it with
`--reference /absolute/path/manifest.json` when preparing each study. Both stage
commands are retained, with the backbone and execution paths replaced.
LPIPS is preserved if that manifest enables it; install `lpips==0.1.4` then.
A `run.training_command` JSON is also supported, but supplies only post-training
arguments; reconstruction still uses the documented MSE defaults. The manifest's
`recipe_status` records which case applies. `config.json` alone is not accepted
by this launcher.

For a controlled local comparison, prepare **both** `--backbone dit` and
`--backbone pixeldit` using the same reference, or the same documented defaults.
Do not claim reproduction of FID 65 without matching that run's configuration.

## Set up on UCL

Extract this package into `/SAN/intelsys/imagenet_mmd_flow/pixeldit_priming_cifar10`,
a separate folder beside Hudson's checkout. All new outputs stay inside this
new folder. Do not extract over `mmd_flow_hudson_new`. It is a complete
source bundle; no changes to an active Hudson run or his source snapshot are
needed. Use an existing CUDA-enabled PyTorch environment:

```bash
cd /SAN/intelsys/imagenet_mmd_flow/pixeldit_priming_cifar10
conda activate mmd_flow
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -r requirements_pixeldit.txt
python scripts/cs_cluster/prepare_pixeldit_assets.py
```

The FID download is verified against the baseline's recorded SHA-256. Download
on the login node before preparing a frozen study. Training nodes need no network
for the model or FID weights. CIFAR data must also be available before submission:

```bash
python main.py --dataset cifar10 \
  --data-root /SAN/intelsys/imagenet_mmd_flow/data --download-only
```

Use a new absolute study directory. The shared `/SAN/...` path below is a
suggested destination based on your checkout location; verify it is writable and
has space for model, EMA, optimizer, and retained checkpoints. A PixelDiT Adam
checkpoint can exceed 2 GB; allow roughly 40–60 GB per full study. The launcher
uses `run_storage.new_run_path` and never silently falls back to home storage.

```bash
python scripts/cs_cluster/pixeldit_study.py prepare \
  --study /SAN/intelsys/imagenet_mmd_flow/pixeldit_priming_cifar10/results/cifar/pixeldit_b_seed42 \
  --data-root /SAN/intelsys/imagenet_mmd_flow/data \
  --backbone pixeldit --data-parallel
```

This freezes source files, weights, both commands, and their hashes under the
study directory. The Python executable is recorded from the activated environment.
For the compact-DiT control, repeat with `--backbone dit` and a different study
directory, such as `compact_dit_seed42`. Add the same `--reference` to both if
available. Do not prepare over an existing study.

## Check the commands and allocated GPUs first

```bash
export STUDY=/SAN/intelsys/imagenet_mmd_flow/pixeldit_priming_cifar10/results/cifar/pixeldit_b_seed42
python "$STUDY/source/scripts/cs_cluster/pixeldit_study.py" run \
  --manifest "$STUDY/manifest.json" --dry-run

qsub -v MANIFEST="$STUDY/manifest.json",PYTHON="$(command -v python)",SMOKE_ONLY=1 \
  "$STUDY/source/scripts/cs_cluster/run_pixeldit_h100.sh"
```

The SGE script follows the existing repo's `aihub_ucl` syntax, requests four H100s
on one host, and checks that exactly four H100s are visible. It does not override
the scheduler's CUDA device assignment. Override the project or scheduler-log
destination with `qsub -P ... -o ...` if needed.

The smoke check uses the full model and production batch counts, allocates
dataset-sized GPU storage, performs reconstruction and multiscale Riesz updates,
updates EMA, and runs Inception on generated images. Inspect:

```bash
cat "$STUDY/smoke.json"
tail -40 "$STUDY/smoke.log"
```

A successful smoke is a capacity/numerics check, not evidence of good FID. It does
not train the production checkpoint. Multi-GPU operation and H100 memory usage
have not been tested on the local CPU-only development machine.

## Full run and recovery

After the smoke job reports `status: passed`, submit the full run:

```bash
qsub -v MANIFEST="$STUDY/manifest.json",PYTHON="$(command -v python)" \
  "$STUDY/source/scripts/cs_cluster/run_pixeldit_h100.sh"
```

The worker repeats the smoke, then runs reconstruction followed by post-training.
One Python training process splits generator forwards over the four GPUs and
gathers **all 128 generated images before computing the loss** on GPU 0.
The real batch remains 256. There is one Adam and one EMA update per global
training update. This preserves the particle interaction populations; it is not
four independent losses or gradient accumulation. Float32 is retained.

Progress and results are saved here:

```bash
tail -f "$STUDY/reconstruction/run.log"
tail -f "$STUDY/posttraining/run.log"
cat "$STUDY/posttraining/fid_metrics.json"
qstat -u "$USER"
```

If the job is interrupted or reaches its 48-hour wall limit, resubmit the same
full-run command. Completed stages are skipped, and the active stage resumes its
last atomic `checkpoint_latest.pt` with optimizer, EMA, and RNG state. Work since
the last checkpoint may be repeated. A file lock prevents two workers from
training the same study concurrently. Frozen-file hashes are checked on recovery.
Do not edit the source inside a prepared study.

## Validation and modified files

```bash
python -m unittest discover -s tests -p 'test_pixeldit*.py' -v
bash -n scripts/cs_cluster/run_pixeldit_h100.sh
```

The new tests cover upstream source hashes, native PixelDiT output equivalence,
sample independence, gradients through both pathways with the actual Riesz loss,
checkpoint transfer, fresh post-training optimizer state, deterministic recovery,
architecture guards, reference command preservation, and frozen-source checks.
A two-GPU equivalence test runs only when at least two CUDA GPUs are available.

The integration changes the results default to this standalone copy, modifies
`backbone.py`, `main.py`, and only architecture
compatibility checks in `priming.py`. `loss.py`, `multiscale.py`, `target_updates.py`,
`fid.py`, and the reconstruction update math are unchanged. There are no new
loss functions. The optional DataParallel wrapper preserves checkpoint key names.

NVIDIA's unchanged code, license, notice, commit, and file hashes are retained
under `third_party/pixeldit/`. Use is subject to that upstream license.
