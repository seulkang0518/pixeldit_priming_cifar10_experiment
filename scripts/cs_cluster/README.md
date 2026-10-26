# CIFAR-10 on cs-cluster

## Drifting and Sinkhorn with released B-model hyperparameters

`run_cifar_official_b_h100.sh` launches a two-task H100 array: task 1 is
Drifting and task 2 is Sinkhorn. Both use 64 generated / 64 real particles,
AdamW with betas (0.9, 0.95), LR 0.0004, 10,000-update warmup from 1e-6,
gradient clipping at 2, EMA 0.999, seed 42, and 200,000 optimizer updates.
Sinkhorn adds an independent 64-particle generated support with epsilon 0.05
and 10 quadratic-cost Sinkhorn iterations; Drifting uses temperatures
[0.2, 0.05, 0.02]. Both use a unit normalized-field target step.

The releases have no CIFAR-10 configs. These runs retain the existing CIFAR
DDPM++ network and multiscale pixel views. See
[the saved reference configs and adaptation boundaries](../../configs/official_reference/README.md)
for the differences from the official conditional ImageNet training recipe.

Submit from a staged project root with `assets/` available:

```bash
qsub scripts/cs_cluster/run_cifar_official_b_h100.sh
```

Each task requests one H100 for 48 hours and writes a source snapshot,
`config.json`, `job_status.json`, and `run.log` to
`results/cifar/officialB_cifar_<JOB_ID>_<TASK_ID>_<OBJECTIVE>/` in the main
cluster checkout. FID-10K runs every 5,000 updates; the latest resumable
checkpoint is saved every 2,000 updates. Scheduler logs are
`/home/zongchen/cifar_official_b.o<JOB_ID>.<TASK_ID>`.

`run_cifar_h100.sh` launches the CIFAR-10 DDPM++ one-pass generator with
softmax Riesz, EMA 0.999, and a 2,000-epoch budget. It requests one H100,
32 GB host memory, 48 hours, and the `aihub_ucl` project.

On `cs-cluster`, change to `/home/zongchen/mmd_flow_hudson_new`, with the source
and FID weights in `assets/`. The default dataset root is
`/SAN/intelsys/imagenet_mmd_flow/data`; the CIFAR-10 archive lives at
`CIFAR10/raw/cifar-10-python.tar.gz` beneath that root (the binary archive is
also supported). Override it with
`qsub -v DATA_ROOT=/absolute/data/path` if needed:

```bash
cd /home/zongchen/mmd_flow_hudson_new
qsub -P aihub_ucl scripts/cs_cluster/run_cifar_h100.sh
```

The dedicated multiscale launcher enables the full-image, 8x8/16x16 coarse-image,
and aligned 7x7/14x14 patch losses with the same DDPM++/EMA recipe:

```bash
qsub -P aihub_ucl scripts/cs_cluster/run_cifar_multiscale_h100.sh
```

Its result directory includes `_multiscale` before the job ID. The source copy
must include `multiscale.py` and `tests/test_multiscale.py`.

The merged scheduler log is written to the home directory as
`/home/zongchen/cifar_ddpmpp_ema.o<JOB_ID>` for the base launcher, or
`/home/zongchen/cifar_multiscale.o<JOB_ID>` for the multiscale launcher,
using the number returned by `qsub`.

This default applies to new submissions. An explicit `qsub -o` overrides it.

The default Python environment is
`/home/zongchen/miniconda3/envs/mmd_flow/bin/python`. To select another existing
environment, pass `-v PYTHON=/absolute/path/to/python` to `qsub`.

The launcher checks the allocated GPU, then starts training directly.
It saves outputs under `results/cifar/h100_ddpmpp_ema0999_<JOB_ID>/`, including
`job_status.json`, `run.log`, metrics, and images. Successful completion produces
`artifact_manifest.json` for SHA-256 verification.

Run the transfer watcher on the destination machine:

```bash
python scripts/cs_cluster/sync_cifar_h100.py path/to/deployment.json
```

The deployment JSON contains `job_id`, `remote_source`, `remote_output`, and
`local_output`. For new submissions, also set `remote_log` to
the appropriate home-directory scheduler log above. Older deployment records without this
field retain their original `remote_source/scheduler.log` location.
The watcher uses the `cs-cluster` SSH alias, copies available
outputs every 300 seconds, and verifies final artifact hashes after completion.
The interval can be changed with `--interval SECONDS`. While training continues,
copied outputs are marked `partial_synced` in `transfer_status.json`.

Existing submitted jobs keep their original source snapshots and script paths.

## Parallel 500-epoch single-factor comparison

From a project root containing the updated Python sources and `assets/`, submit:

```bash
qsub scripts/cs_cluster/run_cifar_ablation_h100.sh
```

This is an SGE array (`-t 1-8`), with one H100 per task and no concurrency cap or
job dependencies. All tasks can run simultaneously, subject to available H100s
and scheduler/project limits. Each task requests 32 GB host memory and 24 hours.
The array has not been submitted by Codex.

| Task | Variant | Only experimental change from control |
| --- | --- | --- |
| 1 | `sinkhorn` | W-Flow two-batch Sinkhorn objective, epsilon 0.05, 10 iterations |
| 2 | `lr_1e4` | Adam learning rate 0.0001 |
| 3 | `lr_5e5` | Adam learning rate 0.00005 |
| 4 | `particles128` | 128 interacting generated samples per update |
| 5 | `global4` | Global/coarse/patch weights 4:1:1 |
| 6 | `no_patch` | Patch-group weight 0, retaining global and coarse |
| 7 | `noise` | Average losses at Gaussian noise std 0, 0.05, 0.15 |
| 8 | `explicit_h01` | Explicit detached-target field step 0.1 |

The completed multiscale run's epoch-500 result is the control; it is not
submitted again. All tasks use the same DDPM++ one-pass generator, EMA 0.999, real batch
256, seed 0, and 500 full passes through 50,000 real training examples
(98,000 optimizer updates). The control learning rate is 0.0002 and generated
population is 64. Tasks 2-8 retain softmax Riesz temperatures `[0.2, 0.05, 0.02]`
and per-temperature RMS normalization. No feature encoder or conditioning is added.

All tasks save FID-10K, sample grids, and a resumable `checkpoint_latest.pt` every
20 epochs, including the final epoch. FID always evaluates clean EMA samples
against the full CIFAR-10 training set, using the same latent seed. Evaluation
latents have their own RNG so changing diagnostic population size does not
change the training data permutation. No compatibility draws are made to
reproduce the older control's RNG progression.

Noise is added independently to real and generated pixels *after constructing
each view*, without clipping, using a separate RNG. These are noise-averaged
distribution losses, not a noise-conditioned network.

Tasks 2-8 retain the existing Drifting-style fixed-point regression: construct
the stop-gradient target as current scaled output plus the sum of the three
per-temperature normalized fields, then regress the generator toward it.
There is no implicit numerical solver. Task 8 scales that field by 0.1; this
rescales the gradient, which Adam largely compensates, so it need not behave
like lowering Adam's learning rate.

Task 1 replaces the field with the W-Flow update from
[Algorithms 1 and 2 of arXiv:2605.11755v2](https://arxiv.org/html/2605.11755v2#S3):
generated-to-real barycentric transport minus generated-to-independent-generated
barycentric transport, followed by detached-target regression. An independent
64-particle latent population is drawn each update using its own saved RNG;
its generator forward has no autograd graph. All three populations use aligned
pixel views and shared crop coordinates. The self-transport has no diagonal mask.

The solver uses uniform balanced marginals, quadratic cost, log-domain updates,
and 10 iterations. Following the
[official implementation](https://github.com/hanjq17/W-Flow/blob/main/drift_loss_ot.py),
each view uses an RMS batch-distance scale, regularization `0.05 * view_dimension`
in scaled coordinates, and RMS-normalized velocity. The reference source
SHA-256 is `c987285caf69cb1085e3e50450936368373087d6066bb6fe15020bae8796b58e`;
the adapted normalization is MIT licensed (see `WFLOW_LICENSE.txt`).
The CSV reports raw velocity RMS and the largest absolute transport-marginal
error across views, averaged over each epoch. Ten iterations give an approximate
balanced plan; the barycenters use actual row masses as in the release.

This is an encoder-free CIFAR adaptation of the Sinkhorn objective, not a
reproduction of the paper's ImageNet feature-space recipe. Architecture,
optimizer settings, pixel-view groups, and their weights remain those of the
completed CIFAR control. The second generator forward adds compute, but only
one Adam update is taken per real minibatch.

Outputs live under `results/cifar/ablation500_<JOB_ID>_<TASK_ID>_<VARIANT>/`.
Each task runs its own source copy in `source/`, records source hashes, and
writes `job_status.json` and `run.log`. Scheduler logs go to
`/home/zongchen/cifar_ablation.o<JOB_ID>.<TASK_ID>`. Completion checks require
501 metric rows, 26 FID evaluations, 98,000 updates, and a final checkpoint;
successful completion also writes SHA-256 hashes in `artifact_manifest.json`.
The existing single-job transfer watcher is not an array-task watcher.

Inspect a task's command without submitting or creating output:

```bash
SGE_TASK_ID=8 bash scripts/cs_cluster/run_cifar_ablation_h100.sh --dry-run
```

To resume an interrupted task, use its recorded command with the saved
`source/main.py`, the same output and experiment arguments, and append
`--resume /absolute/output/path/checkpoint_latest.pt`. Checkpoints retain raw
and EMA weights, Adam state, RNGs, and plotting/metric history. Only the latest
checkpoint is retained. Resume rejects changed experimental settings.

Shell syntax check:

```bash
bash -n scripts/cs_cluster/run_cifar_ablation_h100.sh
```
