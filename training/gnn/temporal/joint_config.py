"""Compatibility aliases for existing seed-job imports and configurations."""

from training.gnn.temporal.config import resolve_joint_job_config as resolve_config
from training.gnn.temporal.joint_finetuning import train_from_base as train
