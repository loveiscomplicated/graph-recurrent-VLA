"""Existing seed, RNG checkpoint and memory helpers for current training."""
import random
import resource
import sys
import numpy as np
import torch

def seed_all(seed,threads):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False

def rng_state(*,include_cuda=False):
    result=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    if include_cuda:result['cuda']=torch.cuda.get_rng_state_all()
    return result

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        if not torch.cuda.is_available():raise RuntimeError('CUDA checkpoint cannot resume on CPU')
        torch.cuda.set_rng_state_all(state['cuda'])

def memory(model,opt):
    result=dict(model_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        optimizer_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)),
        process_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        memory_scope='process RSS high-water including imports/graph collation; CUDA tensor allocator measured separately')
    if next(model.parameters()).is_cuda:
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result
