# Neural Riesz image generation

One shared trainer generates MNIST or CIFAR-10 images from fresh Gaussian
noise using ordinary Riesz MMD, softmax Riesz, Drifting, or Sinkhorn. Training updates only
network parameters. Fixed, independent evaluation inputs track the same sample
identities across epochs.

## Layout

```text
main.py          CLI, training loop, optimizer, experiment configuration
fid.py           CIFAR-10 FID evaluation of generators or saved NumPy samples
assets/          pretrained FID Inception weights, checksum, and license
backbone.py      MLP, compact MNIST U-Net, larger CIFAR RGB attention U-Net
loss.py          Riesz fields/MMD, softmax Riesz, Drifting, and Sinkhorn losses
sinkhorn.py      compatibility imports from loss.py
multiscale.py    coarse-image and aligned-patch training objectives
data.py         verified dataset downloads and both dataset loaders
visualization.py shared image grids, trajectories, and MMD plots
run.sh           shared GPU launcher
scripts/cs_cluster/ CIFAR-10 H100 launcher and result transfer watcher
tests/          mathematical, gradient, backbone, and data/rendering checks
notebooks/       reproducible saved-result comparisons
results/mnist/   MNIST experiments
results/cifar/   CIFAR-10 experiments
figures/         comparison figures and summaries
```

Historical run configurations retain their original launch paths.
`results/reorganization.json` records the old/new paths and migration-time
file checksums. The already-running CIFAR job uses a temporary link at its old
path; a cleanup watcher removes that link once the process exits. Its physical
outputs are stored under `results/cifar/`.

## Run experiments

New runs save under `/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/`.
The trainer, shared launcher, and LSF submission scripts route new relative
destinations such as `results/cifar/my_run` to that directory, including logs,
checkpoints, metrics, and source snapshots. New absolute paths under the old home
`results/` directory are routed there too. Existing runs retain their recorded
paths; the already-running primed DiT study remains in home storage.
`MMD_RESULTS_ROOT` can override the root; an explicit absolute destination outside
the old home results directory is also honored. Missing LongGroup access fails
normally rather than falling back to home storage.

### Reconstruction priming (two-stage CIFAR-10)

The optional reconstruction stage adapts Eq. (1) of
[Are We Overthinking One-Step Generation?](https://arxiv.org/abs/2610.20331).
It trains the existing generator on paired noisy/clean images at fixed `s=0.8`.
For stored pixels `x` in `[0,1]`, inputs are `(1-s)*(2*x-1) + s*noise`,
with standard Gaussian noise; the sigmoid output predicts `x` using mean MSE.
There is no time input. The centered input convention matches the paper's
noise scale; the output and MSE retain this repo's `[0,1]` convention.

```bash
python main.py --dataset cifar10 --data-root data --target-size 0 \
  --training-stage reconstruction --corruption-level 0.8 \
  --epochs 80 --target-batch-size 64 --particles 64 --seed 42 \
  --ema-decay 0.9999 --save-every 20 --checkpoint-every 1 \
  --device cuda --output results/cifar/priming/reconstruction

python main.py --dataset cifar10 --data-root data --target-size 0 \
  --init-checkpoint results/cifar/priming/reconstruction/checkpoint_latest.pt \
  --objective softmax-riesz --multiscale-loss --steps 250000 \
  --target-batch-size 256 --particles 64 --seed 42 --ema-decay 0.999 \
  --save-every 2500 --checkpoint-every 1000 \
  --device cuda --output results/cifar/priming/posttraining
```

Initialization requires a completed, compatible reconstruction checkpoint.
It transfers **online weights and the pretrained EMA shadow**, starts a fresh
optimizer and post-training random streams, and records the source checkpoint
SHA-256 in `config.json`. `--resume` instead restores the entire interrupted run,
including optimizer, random streams, metrics, and EMA. For a resumed post-training
run, keep its original `--init-checkpoint` argument alongside `--resume`.

The reconstruction CSV adds training MSE and fixed-noise held-out MSE at saved
snapshots. FID always measures generation from **pure Gaussian noise**, including
during reconstruction; it does not evaluate reconstructions. Checkpoints are
written atomically when `--checkpoint-every` is enabled.

This first experiment deliberately retains the existing unconditional DDPM++
network, Adam optimizer, and multiscale pixel-space softmax-Riesz loss. It uses
plain reconstruction MSE, without perceptual losses. It tests the two-stage idea;
it does not reproduce the paper's conditional ImageNet feature-space OT recipe.
See [the LSF launcher](scripts/lsf/README.md) for the current cluster's two-stage
submission workflow.

```bash
bash run.sh                                  # CIFAR-10, RGB U-Net, softmax Riesz
DATASET=mnist bash run.sh                     # MNIST, compact U-Net, softmax Riesz
DATASET=mnist OBJECTIVE=riesz bash run.sh      # standard Riesz control
DATASET=mnist ARCHITECTURE=mlp bash run.sh     # MLP control
```

The launcher uses the `mmd_flow_hudson` environment and requires CUDA.
`DATASET`, `ARCHITECTURE`, `OBJECTIVE`, `PARTICLES`, `EPOCHS`, `BATCH_SIZE`,
`SAVE_EVERY`, `BASE_CHANNELS`, `NUM_RES_BLOCKS`, `LATENT_DIM`, `LEARNING_RATE`, `SEED`,
`DATA_ROOT`, `PYTHON`, and `OUTPUT` are configurable shell variables.
The local data root defaults to `/home/zongchen/data`. CIFAR-10 is loaded from
`/home/zongchen/data/CIFAR10/raw/cifar-10-python.tar.gz`; the loader also accepts
the binary archive in the same directory. Set `DATA_ROOT` to use another root.
`EMA_DECAY` defaults to 0.999 for CIFAR-10 and 0 for MNIST; 0 disables EMA.
After each optimizer update, EMA weights are updated from the live training
weights. FID, MMD diagnostics, sample grids, and trajectories use EMA weights
when enabled. The training objective always uses the live model. The EMA copy
starts from the initial model, without a decay ramp; `config.json` records the
decay, update unit, and evaluation weights. CLI runs accept `--ema-decay`.
The default is 2,000 epochs for CIFAR-10 and 100 for MNIST, with image output
every 20 epochs, 64 generated samples per update, and target batches
of 256. Each epoch visits all eligible training images exactly once, including
its final partial batch. This gives 235 updates per MNIST epoch (60,000 images;
last batch 96) or 196 per CIFAR epoch (50,000 images; last batch 80).
Fresh training noise uses its own seeded RNG, independent of target shuffling.
Existing output configurations are protected against overwriting.

For full CLI control, use the main entry point:

```bash
/home/zongchen/miniconda3/envs/mmd_flow_hudson/bin/python main.py \
  --dataset cifar10 --architecture unet --objective softmax-riesz \
  --target-size 0 --target-batch-size 256 --particles 64 --epochs 2000 \
  --device cuda --save-every 20 --output results/cifar/my_run
```

`--steps` selects individual updates instead of `--epochs`. `--target-size 0`
uses all eligible training images; positive sizes select a fixed subset.
`--class-id` restricts either dataset to one class. MNIST also accepts `--digit`.
Without either duration option, the CLI defaults to 2,000 updates.

## Backbones

The compact CIFAR transformer is selected with `--architecture dit`. Its defaults
are 2x2 patches (256 tokens), width 384, 12 blocks, six attention heads, and a
4x GELU MLP. `--dit-width`, `--dit-depth`, `--dit-heads`, and `--dit-patch-size`
configure it. This unconditional DiT-style generator uses fixed 2D sine/cosine
positions, learned constant adaLN-Zero modulation, no dropout, and a sigmoid
pixel head. It has no time/class input and uses the same distributional objectives
as the U-Net. The small nonzero output head lets residual gates learn immediately.

The primed DiT comparison uses its own 80-epoch reconstruction stage, followed by
250,000 updates of corrected softmax Riesz without field RMS normalization.
Post-training transfers both DiT online and EMA weights with fresh Adam state;
U-Net checkpoints are incompatible. Seed 42, generated/real batches 128/256,
multiscale weights 2:1:1, four crops per size, EMA 0.9999, and the LR schedule
2e-4 to 1e-4 match the corrected U-Net reference. FID-10K runs every 2,500
post-training updates and checkpoints every 20,000. The architecture is smaller
than the 56M U-Net, so this comparison does not match parameter count.

```bash
# Without --submit, only create the frozen source and exact command manifest.
.venv/bin/python scripts/lsf/submit_compact_dit.py \
  --study results/cifar/priming_compact_dit_seed42_20261026 --submit
```

One GPU job runs the full-batch smoke check, reconstruction, and post-training
sequentially. It records stage status and supports restart from each stage's latest
checkpoint via `scripts/lsf/run_compact_dit.py --manifest STUDY/manifest.json`.
Use `--dry-run` on that worker to print the recorded commands.

For perceptual priming, install `lpips==0.1.4` and pass `--lpips-weight 0.1`
to `scripts/lsf/submit_compact_dit.py`, using a fresh study name. Reconstruction
then minimizes `[0,1]` pixel MSE plus `0.1 * LPIPS` with frozen VGG16 and calibrated
LPIPS v0.1 weights. This weight is an initial experimental choice, not a tuned
CIFAR optimum. LPIPS receives native 32x32 RGB images, converted internally to
`[-1,1]`, with no resizing. The 80-epoch priming budget and pixel-only Riesz
post-training recipe are retained. The encoder is used only during priming;
its weights are frozen but gradients pass through it to the generator.
The launcher stores its pretrained-weight cache inside the project study,
records weight hashes and package versions, and tests the perceptual objective
in the GPU smoke check. Reconstruction metrics report MSE, LPIPS, and their
weighted total separately. Direct `main.py` use requires both `--lpips-weight`
and an explicit `--lpips-cache`; LPIPS is rejected outside CIFAR reconstruction.

For CIFAR-10 H100 submission with `-P aihub_ucl` and result copying back from
the cluster, see [scripts/cs_cluster/README.md](scripts/cs_cluster/README.md).

All architectures are built through `backbone.build_backbone` and described by
`backbone.backbone_metadata`:

- MNIST U-Net: resolutions $28,14,7$, channels $32,64,128$, residual blocks,
  concatenated skips, GroupNorm, SiLU, and sigmoid pixels. It has 1,018,721
  parameters and takes one $1\times28\times28$ Gaussian noise image.
- CIFAR U-Net: the official Score-SDE CIFAR DDPM++ spatial topology,
  with 56,353,923 parameters in our adaptation,
  with resolutions $32,16,8,4$, base width 128, multipliers $(1,2,2,2)$,
  four encoder residual blocks and five decoder residual blocks per resolution.
  Every encoder block and downsample contributes a concatenated decoder skip.
  Single-head attention follows each encoder block at resolution 16, the decoder
  stage at 16, and the bottleneck. BigGAN residual blocks perform nearest-neighbour
  upsampling and average-pool downsampling, with residual scaling by $1/\sqrt{2}$.
  GroupNorm uses epsilon $10^{-6}$, SiLU activations, dropout 0.1, and DDPM
  fan-average initialization with near-zero residual/attention output branches.
  The spatial topology follows [Song et al.'s CIFAR DDPM++ configuration](https://github.com/yang-song/score_sde_pytorch/blob/main/configs/vp/cifar10_ddpmpp.py)
  and [implementation](https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py).
  Our one-pass adaptation omits timestep embedding and applies a sigmoid to
  generate image pixels from $3\times32\times32$ Gaussian noise. It is not a
  diffusion denoiser and cannot load an official diffusion checkpoint directly.
  Source attribution and the Apache-2.0 license are in `backbone.py` and
  `assets/SCORE_SDE_LICENSE`.
- MLP: default latent dimension 32 and hidden widths $128,256$, with SiLU and
  sigmoid output. `--hidden-dims` customizes these widths for either dataset.

U-Net latent dimensions must match the image size: 784 for MNIST, 3072 for
CIFAR. `--base-channels` controls U-Net width; `--num-res-blocks` controls the
CIFAR stage depth. Adam defaults to learning rate $10^{-3}$ for MNIST and
$2\times10^{-4}$ for CIFAR, with betas $(0.9,0.999)$, epsilon $10^{-8}$,
and no weight decay. These are our experiment budgets and training settings,
rather than a reproduction of the full Score-SDE training recipe.

## Objectives

### Optional multiscale pixel losses

Enable the full-image, coarse-image, and local-patch combination with:

```bash
MULTISCALE_LOSS=1 bash run.sh
```

The generator still takes Gaussian noise and emits one full image in one pass.
Training applies the selected objective independently to the full image,
average-pooled views at 8x8 and 16x16, and aligned crops at 7x7 and 14x14.
For each crop size, one uniformly sampled valid location is shared across the
generated and real populations; images are not paired. Locations are resampled
each update using a dedicated RNG with seed `seed + 2`.

The default loss is `(global + mean(coarse views) + mean(patch views)) / 3`.
CLI options `--coarse-sizes`, `--patch-sizes`, `--crops-per-size`,
`--global-loss-weight`, `--coarse-loss-weight`, and `--patch-loss-weight` configure
the views and group weights. A zero weight disables a group. We average within
groups and normalize the active group weights to sum one. Ordinary Riesz uses
Euclidean distance divided by the square root of the view dimension; softmax
Riesz retains its existing batch-distance and per-coordinate normalization.

The configuration records this geometry and RNG convention. Extra CSV columns
`train_global_surrogate`, `train_coarse_surrogate`, `train_patch_surrogate`, and
`train_combined_surrogate` average training surrogates over updates in each
iteration; these values are not MMD energies. Initialization and disabled groups
have blank entries. FID and the existing full-image MMD diagnostics retain their
evaluation definitions. Without the flag, training follows the original objective.

### Base objectives

The ordinary Riesz kernel is $k(x,y)=-\|x-y\|_2$. For generated images
$x_i=G_\theta(z_i)$, the energy is half the biased empirical MMD squared.
`loss.py` computes its exact image gradient via the field
$v_i=\operatorname{mean}_j (x_i-x_j)/\|x_i-x_j\|_2
-\operatorname{mean}_j (x_i-y_j)/\|x_i-y_j\|_2$.
Backpropagation uses $-v_i/N$, preserving the exact energy gradient through
the generator. Exact collisions contribute zero and pairwise operations are
chunked to bound memory.

Softmax Riesz reuses the loss from
[`hudsonchen/riesz_flow_hudson`](https://github.com/hudsonchen/riesz_flow_hudson),
revision `777d696f31340a1370f5b39a2e7d000464b10526`. The upstream source hash
and license reference are retained in `loss.py`; the license text is in
`SOFTMAX_RIESZ_LICENSE.txt`. It applies row-softmax weights to unit directions
at temperatures `[0.2, 0.05, 0.02]`, masking generated self-diagonals. All other
particles participate; there is no hard top-k truncation. Each temperature
field is RMS-normalized before summation, then the generator regresses toward
a detached goal in batch-scaled coordinates. `--tau-list` controls temperatures;
`--no-normalize-per-temperature` selects upstream's combined-field normalization.
`--no-field-rms-normalization` bypasses both field RMS normalizations while
retaining batch-distance coordinate scaling, softmax temperatures, unit
directions, and the `1/tau` factors. This softmax-only ablation records raw
combined/per-temperature field RMS, attraction/repulsion RMS, pre-clipping
parameter gradient norm, and clipping frequency in `training_diagnostics.csv`.
Rows average updates in intervals of at most 100 updates (also flushed at
checkpoints and completion); field magnitudes first average equally over active
views. `clip_fraction` is the fraction of updates exceeding the configured
gradient norm threshold, not a fraction of parameters. Learning rate is the
last rate in each interval. The existing modes retain their behavior.

`loss.backpropagate_objective` dispatches the Riesz and Drifting training rules. Softmax Riesz's
regression surrogate is distinct from the ordinary Riesz energy. Both are
used in raw pixel space with no feature encoder or CFG fixed negatives.
Finite Adam/minibatch steps need not decrease MMD monotonically.

### Official Drifting loss without a feature encoder

Select `--objective drifting`, or `OBJECTIVE=drifting bash run.sh`.
This ports the [official JAX loss](https://github.com/lambertae/drifting/blob/8861066430db11d2fb0858d042c7845b99df49f1/drift_loss.py)
to PyTorch in `loss.drifting_loss` (also available as `loss.drift_loss`).
The upstream revision/hash are recorded in the source and run configuration;
its MIT license is retained in `DRIFTING_LICENSE.txt`.

The loss uses the geometric mean of row and column softmax affinities over
joint generated/real support, balances attraction and repulsion using their
opposite population's affinity mass, and uses displacement vectors directly.
It preserves the official distance floor, finite self-distance mask, affinity
floor, batch coordinate scaling, per-temperature RMS normalization, and detached
regression target. Optional weighted positives and fixed negatives are supported
by the loss API. `--tau-list` selects temperatures; per-temperature normalization
is always enabled for Drifting.

The trainer applies this loss directly to pixels, without a feature encoder.
`--multiscale-loss` optionally applies it to the existing pooled images and aligned
patches too. This is the official loss within our generator/training setup, rather
than the full ImageNet training recipe: pretrained MAE features, class conditioning,
and sample queues are not added. The published ImageNet models use pretrained
feature encoders, including for pixel-output generation, so changing the loss
alone does not reproduce those results.

```bash
OBJECTIVE=drifting MULTISCALE_LOSS=1 bash run.sh
```

### Sinkhorn implementation

`loss.py` also owns `sinkhorn_plan`, `sinkhorn_barycenter`, and
`sinkhorn_target_loss`. The two-independent-generated-batch W-Flow rule and its
normalization are unchanged. `--objective sinkhorn` selects it; `sinkhorn.py`
retains compatibility imports for existing callers.

## Data and outputs

MNIST downloads from the CVDF mirror to the data root, with canonical checksums.
CIFAR accepts either official binary or Python archives after verifying the
[official checksums](https://cave.cs.toronto.edu/kriz/cifar.html). Verified existing
archives in the data root or Torch vision cache are reused. Archive members are
read directly without extraction. CIFAR-10 FID requires torchvision and SciPy. Pixels are
scaled from uint8 to $[0,1]$; no augmentation or conditioning is used by default.

Each run saves:

- `config.json`, including architecture, optimizer, objective, seed, data source,
  checksums, update budget, evaluation convention, and snapshot epochs.
- `data_indices.json`, training/test example grids, generated snapshot grids,
  and `trajectory.png` with fixed evaluation identities.
- `metrics.csv` and PNG curves. MMD is recorded each epoch. CIFAR-10 saves
  `fid_vs_iteration.png` with MMD and FID in two horizontal panels; MNIST saves
  `mmd_vs_iterations.png`. No PDFs are produced by the trainer.
- `run.log` when launched through `run.sh`.

Pixel-space diagnostic evaluation uses 64 generated images and fixed subsets of up to 512
training and 512 test images. The CSV records ordinary biased Riesz MMD squared;
run-level plots show its square root, without smoothing. This is a pixel-space
diagnostic. The field RMS column
uses the objective-specific convention recorded in the configuration.
Only fixed evaluation outputs are recomputed at each epoch, avoiding a second
large-model forward pass per training minibatch. Set `--checkpoint-every N` to
save resumable checkpoints; the two-stage launchers enable this automatically.
Passed smoke-test outputs are removed; full experiment artifacts are retained.

### CIFAR-10 FID

CIFAR-10 uses FID as the primary quality metric, evaluated at initialization
and every saved snapshot, including the final model. The default compares
10,000 generated images against all 50,000 training images, independently of
the training subset. `--class-id` also filters the FID reference. Use
`--fid-samples 50000` for FID-50K and `--fid-batch-size 64` to control memory.
Report the sample count and reference split/class when comparing scores.
The separate fixed evaluation seed does not consume training randomness.
Reference features are computed once per run; FID adds substantial computation.
The launcher evaluates FID and saves generated grids and the combined metric
plot at epoch 0 and every 20 epochs through epoch 2,000.

The evaluator uses 2048-dimensional features from the TensorFlow-compatible
[pytorch-fid v0.3.0 Inception](https://github.com/mseitzer/pytorch-fid/tree/v0.3.0).
Weights and their SHA-256 checksum are in `assets/`; `_fid_inception.py` vendors
the architecture, with its Apache license in `assets/PYTORCH_FID_LICENSE`.
Preprocessing uses RGB pixels in $[0,1]$, bilinear resizing to $299\times299$
with `align_corners=False`, and scaling to $[-1,1]$.
The `mmd_flow_hudson` environment includes the required torchvision and SciPy.

Scores are saved in `fid_metrics.json` and the `fid` column of `metrics.csv`
(blank between FID snapshots). The configuration records the reference size,
feature dimension, seed, sample count, preprocessing, and weight hash.

Standalone evaluation accepts a NumPy array of individual generated images:

```bash
/home/zongchen/miniconda3/envs/mmd_flow_hudson/bin/python fid.py \
  --samples generated.npy --data-root /home/zongchen/data --output fid.json
```

Samples must have shape `(N,3,32,32)` or `(N,3072)`, with uint8 pixels or
floating pixels in $[0,1]$. The default reference is the full training split;
`--reference-split test` selects the 10,000 test images instead.
The output JSON records FID, counts, split/class, and input hashes.
Rendered grids are unsuitable for FID; historical runs without saved samples
or checkpoints cannot be evaluated retrospectively. New runs evaluate the
live generator directly.

## Comparisons and checks

Softmax Riesz unit directions use direct pairwise differences, with queries
chunked in groups of 32 while retaining the full softmax support. This avoids
false zero distances and cancellation in weighted coordinate sums when
features nearly coincide. Batch distance scaling also uses direct distances.
The detached fixed-point target, temperature weights, and normalization options
are unchanged. Regression tests cover near-collisions, exact duplicates, and
weighted supports against explicit reference calculations.

The corrected CIFAR no-RMS run is
`results/cifar/priming_softmax_no_rms_directdiff_seed42_20261026/` (current LSF job
49819812; initial job 49819790). It starts from the same 80-epoch reconstruction
checkpoint with
temperatures 0.2/0.05/0.02, peak LR 2e-4, 1,000-update warmup, and EMA 0.9999.
It requests one exclusive full GPU and four CPU cores. All four active CIFAR
runs evaluate FID-10K every 2,500 updates and save/retain
checkpoints every 20,000 updates plus completion. Extra best-FID copies are
disabled. The workers refresh the comparison notebook after new FID scores.
Each run's frozen source and manifest preserve its deployed recipe. The earlier no-RMS
run and shared-GPU Sinkhorn pair continue independently.

`notebooks/compare_unet_riesz.ipynb` selects the two exact MNIST seed-0 U-Net
runs. It verifies matching settings, initial-image/data-index hashes, and
complete update coverage, then saves curves, final grids, and metric summaries
under `figures/unet_riesz_comparison_seed0/`. It compares complete recipes:
softmax weighting, normalization, and regression scaling all differ. A single
seed does not isolate these factors or establish general superiority.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 DRIFT_COMPILE=0 \
  /home/zongchen/miniconda3/envs/mmd_flow_hudson/bin/python \
  -m unittest discover -s tests -v
```

The additional upstream softmax function tests are in `tests/test_softmax.py`
and can be run with pytest. End-to-end checks use temporary output directories,
which are removed after successful validation.
