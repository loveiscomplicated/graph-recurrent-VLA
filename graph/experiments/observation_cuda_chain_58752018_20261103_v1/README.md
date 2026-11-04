# Task 10-I — GPU packet → observations → GPU consistency

The preceding view-check change already moved observation construction into the
default CUDA path. This diagnostic verifies that complete prefix explicitly and
measures observation creation separately. It does not reintroduce EE→World→EE
roundtrips or duplicate the runtime implementation.

`benchmark.py` compares matching World/EE CPU and CUDA packets, already validated
and ready before timing. Hand observation creation returns Peg/Box observations
for1,105/0 points. Chain measurements include both source-camera calls: empty
cloned base and real hand. The resident endpoint keeps prepared sensor tensors
outside timing; a separate endpoint includes fresh sensor validation/upload.
Every CUDA endpoint waits for GPU completion; output arrays stay CUDA-resident.

```bash
WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/source python benchmark.py --inputs /path/to/inputs \
  --output /path/to/new/output --warmups 20 --repeats 100
```

Use the existing VastAI58752018 environment and preceding real512 inputs.
Sampling, packet production/validation and file IO are excluded. The old0.541ms
CPU EE→World adapter used a different frame/contract and is historical only.
Point identity/calibration checks remain in the actual helpers. CUDA compact
array layout is checked against already-validated producer packets; the CPU
helper retains its existing additional full observation validation.

The strengthened test forbids `.cpu()` and `.numpy()` during both observation
creation and GPU consistency and checks every array's device, World/EE and flags.
Shared ExperimentLogger, W&B disabled, no publishing, no filter/gate execution.

Full [report](../../../docs/visual-graph-integration/10i_observation_cuda_chain_20261103.md).
