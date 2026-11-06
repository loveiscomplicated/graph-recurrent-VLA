# Named elapsed-time measurements

`utils.timer.Timer` measures user-selected functions, callable objects, and code
regions with Python's monotonic `time.perf_counter()`. It uses only the standard
library and does not print, write files, modify the original callable, or start
an experiment automatically.

## Multiple functions and a complete pipeline

```python
from utils.timer import Timer

timer = Timer()
timed_preprocess = timer.wrap(preprocess, name="preprocess")
timed_model = timer.wrap(model, name="model")
timed_postprocess = timer.wrap(postprocess, name="postprocess")

for inputs in batches:
    with timer.measure("pipeline"):
        data = timed_preprocess(inputs)
        output = timed_model(data)
        result = timed_postprocess(output)

print(timer.summary())
timer.reset()
```

The callables and batches above are supplied by the caller. Functions, bound
methods, and callable objects such as PyTorch modules can be wrapped. A wrapped
module is a function: keep the original module for `.to()`, `.parameters()`,
checkpoint saving, and other module operations. To measure several statements
or a module that cannot be called directly, put those operations in a named
`with timer.measure("name")` region instead.

Decorators use the same aggregate storage:

```python
@timer.wrap(name="prediction")
def predict(inputs):
    return model(inputs)
```

`summary()` returns an independent dictionary snapshot containing only completed
measurements, keyed by name. Each entry contains `count`, `total_seconds`,
`mean_seconds`, `min_seconds`, and `max_seconds`. Reusing a name combines
measurements, including measurements from different functions. Samples are
aggregated rather than retained individually. `reset()` clears all aggregates;
an active measurement still records when it exits.

Nested timings are inclusive: a pipeline's elapsed time includes the functions
inside it. Concurrent aggregate updates are protected by a lock, but overlapping
measurements can count the same wall-clock time. Function returns and exceptions
are preserved; failed calls also contribute to the statistics. Coroutine
functions are timed during their awaited execution, including waiting time.
Generator functions are rejected by `wrap`; measure iterator consumption with a
`measure` region instead. A regular function returning a lazy iterator or
coroutine only measures creation of that object.

## Accelerator synchronization

By default the timer measures host elapsed time, which can cover only dispatch
for asynchronous GPU operations. Supply a zero-argument synchronization callback
to wait before starting the clock and before stopping it:

```python
import torch

timer = Timer(synchronize=lambda: torch.cuda.synchronize(device="cuda:0"))
# On an available MPS device: Timer(synchronize=torch.mps.synchronize)
```

The callback applies to every region on that timer, including nested regions.
Synchronization changes execution overlap and may wait for unrelated device
work; this is elapsed wall-clock timing, not a device-kernel profiler. Callback
errors propagate and may prevent recording a sample or replace a body exception.
Accelerator examples require a matching available backend. The timer tests use a
mock synchronization callback; actual CUDA/MPS execution was not verified for
this utility on the current host.

Run the timer contracts from the repository root:

```bash
.venv/bin/python -m unittest tests.test_timer -v
```
