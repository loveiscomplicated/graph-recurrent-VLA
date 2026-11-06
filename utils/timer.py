"""Opt-in wall-clock timing for named code regions and user-selected callables."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import inspect
from threading import Lock
from time import perf_counter
from typing import Callable, Iterator


@dataclass
class _TimingStats:
    """Keep aggregate seconds without retaining every individual measurement."""

    count: int = 0
    total_seconds: float = 0.0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0

    def record(self, elapsed_seconds: float) -> None:
        self.count += 1
        self.total_seconds += elapsed_seconds
        self.min_seconds = min(self.min_seconds, elapsed_seconds)
        self.max_seconds = max(self.max_seconds, elapsed_seconds)


class Timer:
    """Aggregate inclusive elapsed times under caller-chosen names.

    Use ``with timer.measure("pipeline")`` for a region, or
    ``timer.wrap(function, name="prediction")`` for a callable. No output or
    logging occurs automatically. An optional zero-argument ``synchronize``
    callback runs before each start and stop, e.g. ``torch.cuda.synchronize``.
    """

    def __init__(self, *, synchronize: Callable[[], None] | None = None):
        if synchronize is not None and not callable(synchronize):
            raise TypeError("synchronize must be a zero-argument callable or None")
        self._synchronize = synchronize
        self._stats: dict[str, _TimingStats] = {}
        self._lock = Lock()

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("measurement name must be a non-empty string")

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure a region, including time up to an exception if it raises.

        Nested regions are independent: the outer region includes the inner
        region's time. Synchronization callback errors propagate to the caller.
        """
        self._validate_name(name)
        if self._synchronize is not None:
            self._synchronize()
        started = perf_counter()
        try:
            yield
        finally:
            if self._synchronize is not None:
                self._synchronize()
            elapsed_seconds = perf_counter() - started
            with self._lock:
                stats = self._stats.setdefault(name, _TimingStats())
                stats.record(elapsed_seconds)

    def wrap(self, function: Callable | None = None, *, name: str) -> Callable:
        """Wrap a callable directly or as ``@timer.wrap(name="prediction")``.

        The original callable is not modified. Coroutine functions are timed
        through completion of their await. Generator functions are rejected:
        measuring iterator consumption requires an explicit ``measure`` region.
        """
        self._validate_name(name)
        if function is None:
            def decorate(target: Callable) -> Callable:
                return self.wrap(target, name=name)

            return decorate

        if not callable(function):
            raise TypeError("function must be callable")

        call_method = getattr(function, "__call__", None)
        targets = (function, call_method)
        if any(
            inspect.isgeneratorfunction(target) or inspect.isasyncgenfunction(target)
            for target in targets
        ):
            raise TypeError("generator functions require an explicit timer.measure region")

        if any(inspect.iscoroutinefunction(target) for target in targets):
            @wraps(function)
            async def async_wrapper(*args, **kwargs):
                with self.measure(name):
                    return await function(*args, **kwargs)

            return async_wrapper

        @wraps(function)
        def wrapper(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)

        return wrapper

    def summary(self) -> dict[str, dict[str, int | float]]:
        """Return an independent snapshot of completed measurements in seconds."""
        with self._lock:
            return {
                name: {
                    "count": stats.count,
                    "total_seconds": stats.total_seconds,
                    "mean_seconds": stats.total_seconds / stats.count,
                    "min_seconds": stats.min_seconds,
                    "max_seconds": stats.max_seconds,
                }
                for name, stats in self._stats.items()
            }

    def reset(self) -> None:
        """Clear aggregates; measurements currently in flight will record on exit."""
        with self._lock:
            self._stats.clear()
