# Vendored from https://github.com/hudsonchen/riesz_flow_hudson
# Revision: 777d696f31340a1370f5b39a2e7d000464b10526
# Upstream SHA256: a64f0a97e8ae4998cec7e72b8fff7bae27563467c1a4f75acb6d2307455dd578
# License: see SOFTMAX_RIESZ_LICENSE.txt
"""Riesz, Drifting, and two-batch Sinkhorn detached-target losses."""

from __future__ import annotations

import os
import math
from typing import Dict, Iterable, Tuple

import torch


_COMPILE = os.environ.get("DRIFT_COMPILE", "1") != "0"


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value * value))


@torch.compiler.disable
def _temperature_metrics(temperatures, per_temperature_rms):
    # Keep Python metric-name formatting outside Dynamo, which cannot trace
    # the :g format specifier for symbolic temperatures.
    return {
        f"riesz_topk_softmax_force_rms_tau_{temperature:g}": value.detach()
        for temperature, value in zip(temperatures, per_temperature_rms)
    }


def _softmax_toward(
    source: torch.Tensor,
    target: torch.Tensor,
    target_weight: torch.Tensor,
    tau: float,
    direction_epsilon: float,
    *,
    exclude_self: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return weighted unit directions using direct differences, bounded in memory.

    Avoid squared-norm distance identities and subtracting large weighted sums:
    both lose precision for coincident or nearly coincident pixel features.
    Chunk queries while keeping the complete support in each row's softmax.
    """
    if target.shape[1] == 0:
        empty_ess = source.new_zeros(source.shape[:2])
        return torch.zeros_like(source), empty_ess
    if exclude_self and source.shape[1] != target.shape[1]:
        raise ValueError("self-interaction exclusion requires equal particle counts")

    available = target.shape[1] - int(exclude_self)
    if available == 0:
        empty_ess = source.new_zeros(source.shape[:2])
        return torch.zeros_like(source), empty_ess

    toward_chunks, ess_chunks = [], []
    support_indices = torch.arange(target.shape[1], device=source.device)
    for start in range(0, source.shape[1], 32):
        query = source[:, start:start + 32]
        delta = target[:, None, :, :] - query[:, :, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        logits = -distance / tau
        if exclude_self:
            query_indices = torch.arange(start, start + query.shape[1], device=source.device)
            diagonal = (query_indices[:, None] == support_indices[None, :]).unsqueeze(0)
            logits = logits.masked_fill(diagonal, float('-inf'))
        softmax_weight = torch.softmax(logits, dim=-1)
        coefficient = (softmax_weight * target_weight[:, None, :]
                       / distance.clamp_min(float(direction_epsilon)))
        toward_chunks.append((delta * coefficient[..., None]).sum(dim=2))
        ess_chunks.append(1.0 / softmax_weight.square().sum(dim=-1).clamp_min(1e-30))
    return torch.cat(toward_chunks, dim=1), torch.cat(ess_chunks, dim=1)


def _riesz_topk_softmax_loss_impl(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    *,
    tau: float = 1.0,
    tau_list: Iterable[float] | None = None,
    epsilon: float = 1e-8,
    rms_epsilon: float = 1e-8,
    direction_epsilon: float = 1e-8,
    normalize_per_temperature: bool = False,
    no_field_rms_normalization: bool = False,
    return_field: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regress toward the detached, RMS-normalized softmax Riesz velocity.

    As in ``drift_loss``, softmax logits use distances divided by the global
    batch-derived distance scale. For every configured temperature, the field
    is ``1 / tau`` times attraction toward ``fixed_pos``, minus attraction
    toward the other generated particles, minus attraction toward
    ``fixed_neg``. Fields are summed across ``tau_list``, then the complete
    field is divided by its global RMS magnitude, matching hard top-k Riesz.
    With ``normalize_per_temperature``, each temperature field is instead
    divided by its own RMS before summation, as in Drifting. This mode uses
    Drifting's squared-RMS floor of ``1e-8`` and no final normalization.
    With no_field_rms_normalization, neither field RMS normalization is applied;
    batch-distance coordinate scaling and softmax temperatures are unchanged.
    The legacy total-field RMS floor is applied after the square root. Target
    weights multiply the normalized softmax averages, preserving the training
    code's CFG coefficients. The generated particle itself is excluded from
    its self-interaction softmax because its unit direction is undefined.
    """
    temperatures = (
        tuple(float(value) for value in tau_list)
        if tau_list is not None
        else (float(tau),)
    )
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("tau_list must contain at least one positive temperature")
    if epsilon <= 0 or rms_epsilon <= 0 or direction_epsilon <= 0:
        raise ValueError("epsilon, rms_epsilon, and direction_epsilon must be positive")
    if gen.ndim != 3 or fixed_pos.ndim != 3:
        raise ValueError("gen and fixed_pos must have shape [B, particles, D]")
    if gen.shape[0] != fixed_pos.shape[0] or gen.shape[-1] != fixed_pos.shape[-1]:
        raise ValueError("gen and fixed_pos must have matching batch and feature dimensions")
    if gen.shape[1] < 1 or fixed_pos.shape[1] < 1:
        raise ValueError("gen and fixed_pos must each contain at least one particle")

    if fixed_neg is None:
        fixed_neg = gen.new_empty(gen.shape[0], 0, gen.shape[-1])
    if fixed_neg.ndim != 3:
        raise ValueError("fixed_neg must have shape [B, particles, D]")
    if fixed_neg.shape[0] != gen.shape[0] or fixed_neg.shape[-1] != gen.shape[-1]:
        raise ValueError("fixed_neg and gen must have matching batch and feature dimensions")

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    expected_shapes = (
        ("weight_gen", weight_gen, gen.shape[:2]),
        ("weight_pos", weight_pos, fixed_pos.shape[:2]),
        ("weight_neg", weight_neg, fixed_neg.shape[:2]),
    )
    for name, value, shape in expected_shapes:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {tuple(shape)}")

    gen = gen.float()
    fixed_pos = fixed_pos.detach().float()
    fixed_neg = fixed_neg.detach().float()
    weight_gen = weight_gen.detach().float()
    weight_pos = weight_pos.detach().float()
    weight_neg = weight_neg.detach().float()

    with torch.no_grad():
        old_gen = gen.detach()
        scale_targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale_distance = torch.cdist(old_gen, scale_targets, compute_mode='donot_use_mm_for_euclid_dist')
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        scale_inputs = torch.clamp(scale / (gen.shape[-1] ** 0.5), min=1e-3)

        old_gen_scaled = old_gen / scale_inputs
        pos_scaled = fixed_pos / scale_inputs
        neg_scaled = fixed_neg / scale_inputs

        attraction_field = torch.zeros_like(old_gen_scaled)
        self_repulsion_field = torch.zeros_like(old_gen_scaled)
        fixed_negative_field = torch.zeros_like(old_gen_scaled)
        positive_ess_values = []
        generated_ess_values = []
        negative_ess_values = []
        per_temperature_rms = []

        # Since old_gen_scaled = old_gen / (scale / sqrt(D)), using
        # sqrt(D) * tau here gives logits -||old_gen-target|| / (scale * tau),
        # exactly matching Drifting's temperature convention.
        dimension_factor = gen.shape[-1] ** 0.5
        for temperature in temperatures:
            softmax_temperature = float(temperature) * dimension_factor
            toward_pos, positive_ess = _softmax_toward(
                old_gen_scaled,
                pos_scaled,
                weight_pos,
                softmax_temperature,
                direction_epsilon,
            )
            toward_gen, generated_ess = _softmax_toward(
                old_gen_scaled,
                old_gen_scaled,
                weight_gen,
                softmax_temperature,
                direction_epsilon,
                exclude_self=True,
            )
            toward_neg, negative_ess = _softmax_toward(
                old_gen_scaled,
                neg_scaled,
                weight_neg,
                softmax_temperature,
                direction_epsilon,
            )
            temperature_attraction = toward_pos / float(temperature)
            temperature_self_repulsion = -toward_gen / float(temperature)
            temperature_fixed_negative = -toward_neg / float(temperature)
            temperature_field = (
                temperature_attraction
                + temperature_self_repulsion
                + temperature_fixed_negative
            )
            temperature_rms = _rms(temperature_field)
            if normalize_per_temperature and not no_field_rms_normalization:
                temperature_scale = torch.sqrt(
                    torch.clamp(temperature_field.square().mean(), min=1e-8)
                )
                temperature_attraction = temperature_attraction / temperature_scale
                temperature_self_repulsion = temperature_self_repulsion / temperature_scale
                temperature_fixed_negative = temperature_fixed_negative / temperature_scale
            attraction_field.add_(temperature_attraction)
            self_repulsion_field.add_(temperature_self_repulsion)
            fixed_negative_field.add_(temperature_fixed_negative)
            positive_ess_values.append(positive_ess.mean())
            generated_ess_values.append(generated_ess.mean())
            negative_ess_values.append(negative_ess.mean())
            per_temperature_rms.append(temperature_rms)

        raw_field = attraction_field + self_repulsion_field + fixed_negative_field
        force_rms = _rms(raw_field)
        force_scale = torch.clamp(force_rms, min=float(rms_epsilon))
        frozen_velocity = (
            raw_field if normalize_per_temperature or no_field_rms_normalization
            else raw_field / force_scale
        ).detach()
        goal_scaled = (old_gen_scaled + frozen_velocity).detach()

    gen_scaled = gen / scale_inputs
    loss = torch.mean((gen_scaled - goal_scaled) ** 2, dim=(-1, -2))

    info: Dict[str, torch.Tensor] = {
        "scale": scale.detach(),
        "riesz_topk_softmax_force_rms": force_rms.detach(),
        "riesz_topk_softmax_tau_count": torch.tensor(
            float(len(temperatures)), device=gen.device
        ),
        "riesz_topk_softmax_velocity_rms": _rms(frozen_velocity).detach(),
        "riesz_topk_softmax_attraction_rms": _rms(attraction_field).detach(),
        "riesz_topk_softmax_self_repulsion_rms": _rms(
            self_repulsion_field
        ).detach(),
        "riesz_topk_softmax_fixed_negative_rms": _rms(
            fixed_negative_field
        ).detach(),
        "riesz_topk_softmax_positive_ess": torch.stack(
            positive_ess_values
        ).mean().detach(),
        "riesz_topk_softmax_generated_ess": torch.stack(
            generated_ess_values
        ).mean().detach(),
        "riesz_topk_softmax_negative_ess": torch.stack(
            negative_ess_values
        ).mean().detach(),
    }
    info.update(_temperature_metrics(temperatures, per_temperature_rms))
    if return_field:
        info.update(frozen_velocity=frozen_velocity, scale_inputs=scale_inputs)
    if len(temperatures) == 1:
        info["riesz_topk_softmax_tau"] = torch.tensor(
            temperatures[0], device=gen.device
        )
    return loss, info


if _COMPILE:
    riesz_topk_softmax = torch.compile(
        _riesz_topk_softmax_loss_impl, dynamic=True
    )
else:
    riesz_topk_softmax = _riesz_topk_softmax_loss_impl

# Keep the conventional ``*_loss`` spelling available to callers as well.
riesz_topk_softmax_loss = riesz_topk_softmax


def raw_field_diagnostics(info):
    """Scalar magnitudes for calls with no_field_rms_normalization=True."""
    prefix = 'riesz_topk_softmax_'
    result = {f'raw_{name}': info[prefix + name] for name in
              ('force_rms', 'attraction_rms', 'self_repulsion_rms')}
    result.update({name.replace(prefix, 'raw_'): value for name, value in info.items()
                   if name.startswith(prefix + 'force_rms_tau_')})
    return result



@torch.no_grad()
def mean_away(x: torch.Tensor, y: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Mean unit directions away from y; zero direction for exact collisions.

    Direct differences avoid cancellation near collisions. Chunking bounds the
    memory without approximating the field or smoothing the Riesz kernel.
    """
    result = torch.zeros_like(x)
    for start in range(0, len(y), chunk_size):
        delta = x[:, None, :] - y[None, start:start + chunk_size, :]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        delta.div_(torch.where(distance > 0, distance, torch.ones_like(distance)))
        result.add_(delta.sum(dim=1))
    return result / len(y)



@torch.no_grad()
def velocity(x: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Wasserstein negative gradient of half the biased empirical MMD squared."""
    return mean_away(x, x, chunk_size) - mean_away(x, target, chunk_size)



@torch.no_grad()
def mean_distance(x: torch.Tensor, y: torch.Tensor, chunk_size: int) -> float:
    total = 0.0
    for start in range(0, len(x), chunk_size):
        for stop in range(0, len(y), chunk_size):
            delta = x[start:start + chunk_size, None, :] - y[None, stop:stop + chunk_size, :]
            total += torch.linalg.vector_norm(delta, dim=-1).double().sum().item()
    return total / (len(x) * len(y))



def mmd_squared(x: torch.Tensor, y: torch.Tensor, yy: float, chunk_size: int) -> float:
    return 2 * mean_distance(x, y, chunk_size) - mean_distance(x, x, chunk_size) - yy



def backpropagate_objective(images, target, chunk_size, *, objective="riesz",
                            tau_list=(0.2, 0.05, 0.02), normalize_per_temperature=True):
    """Apply the chosen image-output gradient without changing its scaling."""
    if objective == "drifting":
        loss, _ = drifting_loss(images.unsqueeze(0), target.unsqueeze(0), tau_list=tau_list)
        loss.mean().backward()
    elif objective == "softmax-riesz":
        loss, _ = _riesz_topk_softmax_loss_impl(
            images.unsqueeze(0), target.unsqueeze(0), tau_list=tau_list,
            normalize_per_temperature=normalize_per_temperature, epsilon=1e-6)
        loss.mean().backward()
    elif objective == "riesz":
        images.backward(-velocity(images.detach(), target, chunk_size) / len(images))
    else:
        raise ValueError(f"Unknown objective: {objective}")


# W-Flow quadratic-cost adaptation; see WFLOW_LICENSE.txt.
# Reference: https://github.com/hanjq17/W-Flow/blob/main/drift_loss_ot.py
@torch.no_grad()
def sinkhorn_plan(source, support, *, epsilon, iterations=10):
    """Uniform balanced coupling for cost ||x-y||^2/2, with fixed iterations.

    Finite iteration counts need not exactly satisfy both marginals. Return
    the actual plan so callers can measure this approximation error.
    """
    if (source.ndim != 2 or support.ndim != 2 or source.shape[1] != support.shape[1]
            or not len(source) or not len(support)):
        raise ValueError('Sinkhorn needs nonempty [particles, dimension] populations')
    if not math.isfinite(epsilon) or epsilon <= 0 or iterations < 1:
        raise ValueError('Sinkhorn epsilon and iteration count must be positive')
    cost = 0.5 * torch.cdist(source, support).square()
    log_kernel = -cost / epsilon
    log_a, log_b = -math.log(len(source)), -math.log(len(support))
    log_v = support.new_zeros(len(support))
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    return (log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)).exp()


@torch.no_grad()
def sinkhorn_barycenter(source, support, *, epsilon, iterations=10):
    plan = sinkhorn_plan(source, support, epsilon=epsilon, iterations=iterations)
    rows = plan.sum(dim=1, keepdim=True)
    # Normalize actual row masses, as in the release, rather than assuming
    # exact convergence after the fixed iteration budget.
    projected = plan @ support / rows.clamp_min(1e-12)
    row_error = (rows.flatten() - 1 / len(source)).abs().max()
    col_error = (plan.sum(dim=0) - 1 / len(support)).abs().max()
    return projected, torch.maximum(row_error, col_error)


def sinkhorn_target_loss(features, target, reference, *, epsilon=0.05, iterations=10):
    """Regress toward x + T_real(x) - T_independent_generated(x).

    `reference` must be a separately sampled, detached generator population.
    The two maps share the same batch scale and entropic regularization.
    This scalar loss is a regression surrogate, not a Sinkhorn energy estimate.
    """
    if reference is None:
        raise ValueError('W-Flow requires an independent generated reference batch')
    with torch.no_grad():
        x, y, ref = features.detach().float(), target.detach().float(), reference.detach().float()
        dimension = x.shape[1]
        support = torch.cat((x, ref, y))
        distance_rms = (torch.cdist(x, support).square().mean() / (1 + 1e-8)).sqrt()
        scale = (distance_rms / math.sqrt(dimension)).clamp_min(1e-3)
        xs, ys, rs = x / scale, y / scale, ref / scale
        regularization = epsilon * dimension
        attraction, real_error = sinkhorn_barycenter(
            xs, ys, epsilon=regularization, iterations=iterations)
        repulsion, generated_error = sinkhorn_barycenter(
            xs, rs, epsilon=regularization, iterations=iterations)
        velocity = attraction - repulsion
        velocity_rms = velocity.square().mean().clamp_min(1e-8).sqrt()
        goal = xs + velocity / velocity_rms
        if not torch.isfinite(goal).all():
            raise RuntimeError('Nonfinite Sinkhorn regression target')
        info = dict(sinkhorn_velocity_raw_rms=velocity.square().mean().sqrt(),
                    sinkhorn_marginal_max_error=torch.maximum(real_error, generated_error),
                    sinkhorn_solver_iterations=xs.new_tensor(iterations))
    return (features.float() / scale - goal).square().mean(), info


@torch.no_grad()
def sinkhorn_cost_plan(cost, *, epsilon, iterations=1000, tolerance=1e-6):
    """Balanced log-domain Sinkhorn; tolerance=0 uses fixed iterations, no checks."""
    if cost.ndim != 2 or min(cost.shape) < 1:
        raise ValueError('Expected a nonempty cost matrix')
    if (not math.isfinite(epsilon) or epsilon <= 0 or iterations < 1
            or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Invalid Sinkhorn solver settings')
    # Row/column offsets leave a balanced coupling unchanged. Removing them
    # reduces cancellation in float32 dual potentials at small epsilon.
    centered = cost.detach() - cost.detach().amin(dim=1, keepdim=True)
    centered = centered - centered.amin(dim=0, keepdim=True)
    kernel = -centered / epsilon
    log_a, log_b = -math.log(cost.shape[0]), -math.log(cost.shape[1])
    v = cost.new_zeros(cost.shape[1])
    for step in range(1, iterations + 1):
        u = log_a - torch.logsumexp(kernel + v[None, :], dim=1)
        v = log_b - torch.logsumexp(kernel + u[:, None], dim=0)
        if tolerance and (step % 10 == 0 or step == iterations):
            plan = (kernel + u[:, None] + v[None, :]).exp()
            error = torch.maximum((plan.sum(1) - 1 / cost.shape[0]).abs().max(),
                                  (plan.sum(0) - 1 / cost.shape[1]).abs().max())
            if error.item() <= tolerance:
                break
    if not tolerance:
        # Compute the final diagnostic only; never synchronize for convergence.
        plan = (kernel + u[:, None] + v[None, :]).exp()
        error = torch.maximum((plan.sum(1) - 1 / cost.shape[0]).abs().max(),
                              (plan.sum(0) - 1 / cost.shape[1]).abs().max())
    return plan, error, step


def sinkhorn_ot_loss(features, target, reference, *, iterations=1000, tolerance=1e-6,
                     epsilon=None):
    """Direct detached-plan OT attraction minus independent generated repulsion.

    Priming-inspired epsilon mixture (.001: .75, .01: .25). Our adaptation uses
    batch-scaled pixel views and dimension-averaged quadratic costs, rather than
    calibrated frozen encoder features. No field or gradient RMS normalization.
    The scalar is a gradient surrogate, not the complete Sinkhorn divergence.
    """
    if reference is None:
        raise ValueError('OT requires independent generated reference samples')
    if epsilon is not None and (not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('Direct OT epsilon must be finite and positive')
    regularizers = ((.001, .75), (.01, .25)) if epsilon is None else ((epsilon, 1.),)
    with torch.no_grad():
        x, y, ref = features.detach().float(), target.detach().float(), reference.detach().float()
        dimension = x.shape[1]
        distance = torch.cdist(x, torch.cat((x, ref, y)), compute_mode='donot_use_mm_for_euclid_dist')
        scale = (distance.square().mean().sqrt() / math.sqrt(dimension)).clamp_min(1e-3)
    z = features.float() / (scale * math.sqrt(dimension))
    y, ref = y / (scale * math.sqrt(dimension)), ref / (scale * math.sqrt(dimension))
    costs = [0.5 * torch.cdist(z, support, compute_mode='donot_use_mm_for_euclid_dist').square()
             for support in (y, ref)]
    terms, errors, counts, row_errors = [], [], [], []
    velocity = torch.zeros_like(z.detach())
    for cost, support, sign in zip(costs, (y, ref), (1., -1.)):
        term = z.new_zeros(())
        for regularization, weight in regularizers:
            plan, error, count = sinkhorn_cost_plan(
                cost, epsilon=regularization, iterations=iterations, tolerance=tolerance)
            term = term + weight * (plan * cost).sum()
            with torch.no_grad():
                velocity.add_(sign * weight * (plan @ support - plan.sum(1, keepdim=True) * z.detach()))
            errors.append(error)
            counts.append(count)
            row_errors.append((plan.sum(1) * len(z) - 1).abs().max())
        terms.append(term)
    value = terms[0] - terms[1]
    if not torch.isfinite(value):
        raise RuntimeError('Nonfinite direct OT loss')
    return value, dict(sinkhorn_ot_attraction=terms[0].detach(),
                       sinkhorn_ot_repulsion=terms[1].detach(),
                       sinkhorn_velocity_raw_rms=(velocity * len(z)).square().mean().sqrt(),
                       sinkhorn_marginal_max_error=torch.stack(errors).max(),
                       sinkhorn_row_max_relative_error=torch.stack(row_errors).max(),
                       sinkhorn_solver_iterations=z.new_tensor(max(counts)))


# PyTorch adaptation of the official Drifting loss (MIT; DRIFTING_LICENSE.txt).
# Reference: https://github.com/lambertae/drifting/blob/main/drift_loss.py
# Revision: 8861066430db11d2fb0858d042c7845b99df49f1
# Upstream SHA256: 445ba0485e1d81be81aee4e1d5458d85548db171ecd310c1fc855ea6242bc782

def drifting_loss(gen, fixed_pos, fixed_neg=None, weight_gen=None,
                  weight_pos=None, weight_neg=None, *,
                  tau_list=(0.2, 0.05, 0.02), return_field=False):
    """Official Drifting affinity/force construction on [B, particles, D].

    Joint positive/negative affinities use the geometric mean of row and
    column softmaxes. Cross-population masses balance attraction and repulsion;
    displacements are not divided by pairwise distance. Each temperature is
    RMS-normalized before detached-target regression in scaled coordinates.
    Retains the release's finite diagonal mask and affinity floor exactly.
    """
    temperatures = tuple(float(t) for t in tau_list)
    if not temperatures or any(not math.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError('Drifting temperatures must be finite and positive')
    if gen.ndim != 3 or min(gen.shape) < 1:
        raise ValueError('gen must be nonempty [B, particles, D]')
    if fixed_neg is None:
        fixed_neg = gen[:, :0].detach()
    populations = (gen, fixed_neg, fixed_pos)
    weights = []
    for population, weight in zip(populations, (weight_gen, weight_neg, weight_pos)):
        if (population.ndim != 3 or population.shape[0] != gen.shape[0]
                or population.shape[-1] != gen.shape[-1]):
            raise ValueError('Populations must share batch and feature dimensions')
        if weight is None:
            weight = torch.ones_like(population[..., 0])
        if weight.shape != population.shape[:2] or not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError('Weights must be finite nonnegative [B, particles] arrays')
        weights.append(weight.detach().float())
    if fixed_pos.shape[1] < 1 or not (weights[2].sum(dim=1) > 0).all():
        raise ValueError('Each batch needs positive target mass')
    gen = gen.float()
    with torch.no_grad():
        anchor = gen.detach()
        support = torch.cat([anchor, fixed_neg.detach().float(), fixed_pos.detach().float()], dim=1)
        support_weights = torch.cat(weights, dim=1)
        # Match the release's sqrt(max(squared_distance, 1e-8)), including collisions.
        squared = (anchor.square().sum(-1, keepdim=True)
                   + support.square().sum(-1).unsqueeze(1)
                   - 2 * torch.bmm(anchor, support.transpose(1, 2)))
        distances = squared.clamp_min(1e-8).sqrt()
        scale = (distances * support_weights[:, None]).mean() / support_weights.mean()
        scale_inputs = (scale / math.sqrt(gen.shape[-1])).clamp_min(1e-3)
        xs, supports = anchor / scale_inputs, support / scale_inputs
        distance = distances / scale.clamp_min(1e-3)
        count = gen.shape[1]
        mask = torch.zeros_like(distance)
        mask[:, :, :count] = torch.eye(count, device=gen.device)
        distance = distance + 100 * mask
        split = count + fixed_neg.shape[1]
        field = torch.zeros_like(xs)
        info = {'scale': scale}
        for temperature in temperatures:
            logits = -distance / temperature
            affinity = (logits.softmax(-1) * logits.softmax(-2)).clamp_min(1e-6).sqrt()
            affinity = affinity * support_weights[:, None]
            negative, positive = affinity[:, :, :split], affinity[:, :, split:]
            coefficients = torch.cat((-negative * positive.sum(-1, keepdim=True),
                                      positive * negative.sum(-1, keepdim=True)), dim=-1)
            force = (torch.bmm(coefficients, supports)
                     - coefficients.sum(-1, keepdim=True) * xs)
            magnitude = force.square().mean()
            info[f'drifting_force_squared_tau_{temperature:g}'] = magnitude
            field = field + force / magnitude.clamp_min(1e-8).sqrt()
        goal = xs + field
        info['drifting_velocity_rms'] = field.square().mean().sqrt()
        if return_field:
            info.update(frozen_velocity=field, scale_inputs=scale_inputs)
    return (gen / scale_inputs - goal).square().mean(dim=(-1, -2)), info


# Official code uses the shorter spelling.
drift_loss = drifting_loss
