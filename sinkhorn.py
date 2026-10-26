"""Compatibility imports; all Sinkhorn implementations now live in loss.py."""
from loss import sinkhorn_plan, sinkhorn_barycenter, sinkhorn_target_loss

__all__ = ['sinkhorn_plan', 'sinkhorn_barycenter', 'sinkhorn_target_loss']
