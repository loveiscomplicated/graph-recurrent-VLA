"""Preserve the existing selected-M0 Mamba workflow entrypoints."""

from training.gnn.temporal.selected_estimator import aggregate, train_and_evaluate

__all__ = ["aggregate", "train_and_evaluate"]
