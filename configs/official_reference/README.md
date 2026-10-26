Reference YAMLs fetched on 2026-10-15:

- W-Flow: https://github.com/hanjq17/W-Flow/blob/main/configs/gen/latent_sota_B_ot_8node.yaml
- Drifting: https://github.com/lambertae/drifting/blob/main/configs/gen/latent_sota_B.yaml

Neither release provides a CIFAR-10 configuration. The launcher
`scripts/cs_cluster/run_cifar_official_b_h100.sh` transfers the following B-model
hyperparameters to this repository's existing unconditional CIFAR pixel model:

| Setting | Both methods |
| --- | --- |
| Generated / positive population per interaction group | 64 / 64 |
| Independent negative generated population (Sinkhorn) | 64 |
| Optimizer | AdamW, betas (0.9, 0.95), epsilon 1e-8, weight decay 0 |
| Peak learning rate | 0.0004 |
| Warmup | 10,000 updates, linear from 1e-6; constant thereafter |
| Gradient norm cap | 2 |
| EMA | 0.999 per update |
| Training budget | 200,000 optimizer updates |
| Seed | 42 |
| Detached normalized-field target step | 1 |

Drifting uses temperatures [0.2, 0.05, 0.02]. Sinkhorn uses quadratic cost,
epsilon 0.05 times view dimension, 10 iterations, independent resampled
generated support, and no diagonal mask. `--field-step 1` records the unit step;
the Sinkhorn and Drifting losses directly construct their normalized targets.

Adaptation boundaries: the official conditional ImageNet model uses 128
interaction groups per train batch, positive memory banks, 32 unconditional
negative samples, CFG, a latent DiT, VAE, and MAE activations. These CIFAR jobs
use one unconditional interaction group, fresh random real batches from all
50,000 training images, the existing DDPM++ generator, and equally weighted
global/coarse/aligned-patch pixel views. They have no unconditional-negative
population or CFG. The official global batch sizes (4096 W-Flow / 2048 Drifting)
are not reproduced. The CIFAR diagnostics remain FID-10K every 5,000 updates;
only the latest resumable checkpoint is kept, written every 2,000 updates.
These are adaptations of released B hyperparameters, not full paper reproductions.

Sinkhorn audit: `tests/fixtures/sinkhorn_official.json` stores losses, gradients,
and plans evaluated directly with the unmodified released `drift_loss_ot.py`
(SHA256 c987285caf69cb1085e3e50450936368373087d6066bb6fe15020bae8796b58e).
Cases include unequal population sizes and 64/64/64 particles. The tests compare
the local unconditional, quadratic-cost path against these reference values.
The official Drifting fixtures separately check its loss and generator gradient.
The W-Flow and Drifting licenses are retained in the project root.
