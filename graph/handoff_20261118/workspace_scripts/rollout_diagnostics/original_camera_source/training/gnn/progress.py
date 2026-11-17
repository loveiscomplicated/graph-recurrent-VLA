"""Batch-boundary progress reporting for GNN training and validation."""

from time import perf_counter

import torch


def format_duration(seconds):
    """Render elapsed times and estimates as hours, minutes and seconds."""
    if seconds is None:
        return "estimating"
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def batch_count(loader):
    """Keep unsized custom iterables usable, without consuming them."""
    try:
        return len(loader)
    except (TypeError, NotImplementedError):
        return None


class TrainingProgress:
    """Report wall-clock progress through the existing experiment logger.

    Train and validation rates are estimated separately once observed. Until
    then, the observed batch rate is used for the unobserved phase as well.
    Resume estimates use only measurements from the current invocation.
    """

    def __init__(self, experiment, epochs, train_loader, validation_loader,
                 interval_seconds=10.0, clock=perf_counter):
        self.experiment = experiment
        self.epochs = epochs
        self.sizes = {"train": batch_count(train_loader),
                      "validation": batch_count(validation_loader)}
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.started_at = clock()
        self.previous_rates = {}
        self.completed_epoch_seconds = []

    def begin_epoch(self, epoch):
        """Start timing before the first training batch is loaded."""
        self.epoch = epoch
        self.epoch_started_at = self.clock()
        self.counts = {name: 0 for name in self.sizes}
        self.phase_seconds = {name: 0.0 for name in self.sizes}

    def begin_phase(self, phase):
        """Print immediately when entering training or validation."""
        self.phase = phase
        self.phase_started_at = self.clock()
        self.metric_totals = {}
        self.metric_denominators = {}
        self.report(force=True)

    def batch_completed(self, totals=None, denominators=None):
        """Include all fetched batches, even batches filtered to None."""
        if totals is not None:
            self.metric_totals = totals
            self.metric_denominators = denominators
        self.counts[self.phase] += 1
        self.report()

    def finish_phase(self):
        """Capture phase duration including epoch metric aggregation."""
        self.phase_seconds[self.phase] = self.clock() - self.phase_started_at
        self.report(force=True)
        count = self.counts[self.phase]
        if count:
            self.previous_rates[self.phase] = self.phase_seconds[self.phase] / count

    def estimates(self, now):
        """Estimate remaining compute from observed phase rates and batch counts."""
        durations = dict(self.phase_seconds)
        durations[self.phase] = now - self.phase_started_at
        observed_batches = sum(self.counts.values())
        fallback_rate = None
        if observed_batches:
            fallback_rate = sum(durations.values()) / observed_batches
        epoch_remaining = 0.0
        estimated_epoch_seconds = 0.0
        for phase, size in self.sizes.items():
            if size is None:
                return None, None
            count = self.counts[phase]
            if count:
                rate = durations[phase] / count
            else:
                rate = self.previous_rates.get(phase, fallback_rate)
            if rate is None:
                return None, None
            epoch_remaining += max(0, size - count) * rate
            estimated_epoch_seconds += size * rate
        # Completed epochs include checkpoint/logging overhead. The current
        # partial epoch is used until the first full epoch has been measured.
        if self.completed_epoch_seconds:
            estimated_epoch_seconds = (
                sum(self.completed_epoch_seconds) / len(self.completed_epoch_seconds)
            )
        total_remaining = epoch_remaining + (self.epochs - self.epoch) * estimated_epoch_seconds
        return epoch_remaining, total_remaining

    def report(self, force=False):
        """Flush numeric metrics and a readable terminal line at the chosen cadence."""
        now = self.clock()
        if not force and now - self.last_report_at < self.interval_seconds:
            return
        epoch_remaining, total_remaining = self.estimates(now)
        elapsed = now - self.epoch_started_at
        metrics = {
            "epoch": self.epoch,
            "total_epochs": self.epochs,
            "epoch_elapsed_seconds": elapsed,
            "elapsed_seconds": now - self.started_at,
            "train_batches_completed": self.counts["train"],
            "validation_batches_completed": self.counts["validation"],
        }
        if epoch_remaining is not None:
            metrics["epoch_remaining_seconds"] = epoch_remaining
            metrics["total_remaining_seconds"] = total_remaining
        loss_text = ""
        if self.metric_totals:
            names = list(self.metric_totals)
            means = torch.stack([
                self.metric_totals[name] / self.metric_denominators[name] for name in names
            ]).detach().cpu().tolist()
            running_metrics = dict(zip(names, means))
            metrics.update({f"{self.phase}/{name}": value for name, value in running_metrics.items()})
            loss_text = f" loss={running_metrics['loss']:.6f}"
        self.experiment.log(metrics, step=self.epoch, prefix="progress")
        size = self.sizes[self.phase]
        print(
            f"Epoch {self.epoch}/{self.epochs} [{self.phase}] "
            f"batch {self.counts[self.phase]}/{size if size is not None else '?'}{loss_text} "
            f"elapsed={format_duration(elapsed)} "
            f"epoch_remaining={format_duration(epoch_remaining)} "
            f"total_remaining={format_duration(total_remaining)}",
            flush=True,
        )
        self.last_report_at = now

    def finish_epoch(self):
        """Return the completed epoch duration, including checkpoint writes."""
        duration = self.clock() - self.epoch_started_at
        self.completed_epoch_seconds.append(duration)
        self.experiment.log({"epoch_seconds": duration}, step=self.epoch, prefix="time")
        return duration
