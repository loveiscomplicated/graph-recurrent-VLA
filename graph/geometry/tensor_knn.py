"""Device-only neighbor membership for the final bounded surface graph."""
import torch


@torch.inference_mode()
def neighbor_membership(xyz, k=8):
    if xyz.device.type != 'cuda' or xyz.dtype != torch.float64 or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('CUDA float64 [N,3] points required')
    n=len(xyz)
    if not bool((torch.isfinite(xyz)&(xyz.abs()<1e150)).all()):
        raise ValueError('finite bounded float64 coordinates required')
    if n>128 or k<1: raise ValueError('final graph requires <=128 rows/object and positive k')
    result=torch.zeros((n,n),device=xyz.device,dtype=torch.bool)
    if not n: return result
    import triton
    from graph.geometry.tensor_knn_kernels import build, query
    xyz=xyz.contiguous(); count=min(k+1,n); cap=2*n
    indices=torch.empty(n+1,device=xyz.device,dtype=torch.int32)
    nodes=torch.empty((cap,5),device=xyz.device,dtype=torch.int32)
    splits=torch.empty(cap,device=xyz.device,dtype=torch.float64)
    heaps=torch.empty((n,cap,5),device=xyz.device,dtype=torch.float64)
    neighbors=torch.empty((n,count,5),device=xyz.device,dtype=torch.float64)
    build[(1,)](xyz,indices,nodes,splits,n,triton.next_power_of_2(n),enable_fp_fusion=False,num_warps=1)
    query[(n,)](xyz,indices,nodes,splits,heaps,neighbors,result,n,count,cap,enable_fp_fusion=False,num_warps=1)
    return result
