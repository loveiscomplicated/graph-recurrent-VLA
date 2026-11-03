import numpy as np
import torch
from scipy.spatial import cKDTree
from graph.persistent_geometry.tensor_knn import neighbor_membership
from graph.persistent_geometry.tensor_knn_kernels import build
import triton

def check(x):
    x=np.asarray(x,np.float64);n=len(x);tree=cKDTree(x)
    gpu=torch.as_tensor(x,device='cuda');I=torch.empty(n+1,device='cuda',dtype=torch.int32)
    T=torch.empty((2*n,5),device='cuda',dtype=torch.int32); S=torch.empty(2*n,device='cuda',dtype=torch.float64)
    build[(1,)](gpu,I,T,S,n,triton.next_power_of_2(n),enable_fp_fusion=False,num_warps=1)
    ix=I[:n].cpu().numpy()
    if not np.array_equal(ix,tree.indices):
        print('permutation mismatch',n,ix,tree.indices,flush=True);raise AssertionError('tree permutation')
    actual=neighbor_membership(gpu).cpu().numpy();near=tree.query(x,k=min(9,n))[1]
    if n==1:near=near[:,None]
    expected=np.zeros((n,n),bool);expected[np.arange(n)[:,None],near]=True
    np.testing.assert_array_equal(actual,expected)

if __name__=='__main__':
    torch.set_num_threads(1);rng=np.random.default_rng(42);cases=0
    for n in (1,2,8,9,16,17,25,31,32,33,64,97,127,128):
        for i in range(5):
            for typ,x in enumerate((rng.normal(size=(n,3)),rng.integers(-2,3,size=(n,3)),np.zeros((n,3)))):
                print('checking',n,i,typ,flush=True);check(x);cases+=1
        print('passed',n,cases,flush=True)
    for nx in (4,5,8):
        x=np.array(np.meshgrid(np.arange(nx),np.arange(8),[0])).reshape(3,-1).T
        for _ in range(5):check(x[rng.permutation(len(x))]);cases+=1
    print('KNN PASSED',cases,flush=True)
