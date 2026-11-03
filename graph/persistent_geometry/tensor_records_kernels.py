"""Ordered greedy 3mm representatives. One device loop, bounded work buffers."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def cells(X, C, N, R: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    x = tl.load(X+i, i < N*3, other=0)
    r = tl.full((), R, tl.float64)
    # NumPy's divide then floor, not multiply by a rounded reciprocal.
    c = libdevice.floor(libdevice.div_rn(x, r)).to(tl.int64)
    tl.store(C+i, c, i < N*3)


@triton.jit
def greedy(X, O, C, OWNER, N, R: tl.constexpr, B: tl.constexpr):
    lane = tl.arange(0, B)
    radius = tl.full((), R, tl.float64)
    first = 0
    while first < N:
        x = tl.load(X+first*3); y = tl.load(X+first*3+1); z = tl.load(X+first*3+2)
        oid = tl.load(O+first)
        cx = tl.load(C+first*3); cy = tl.load(C+first*3+1); cz = tl.load(C+first*3+2)
        following = tl.full((), 2147483647, tl.int32)
        for chunk in range(tl.cdiv(N, B)):
            j = chunk*B+lane
            owner = tl.load(OWNER+j, j < N, other=0)
            free = (j < N) & (owner < 0)
            dx = tl.load(X+j*3, free, other=0)-x
            dy = tl.load(X+j*3+1, free, other=0)-y
            dz = tl.load(X+j*3+2, free, other=0)-z
            cell_ok = (tl.abs(tl.load(C+j*3, free, other=0)-cx) <= 1)
            cell_ok &= (tl.abs(tl.load(C+j*3+1, free, other=0)-cy) <= 1)
            cell_ok &= (tl.abs(tl.load(C+j*3+2, free, other=0)-cz) <= 1)
            distance = libdevice.sqrt((dx*dx+dy*dy)+dz*dz)
            hit = free & cell_ok & (tl.load(O+j, free, other=-1) == oid) & (distance < radius)
            updated = tl.where(hit, first, owner)
            tl.store(OWNER+j, updated, j < N)
            following = tl.minimum(following, tl.min(tl.where((j < N) & (updated < 0), j, 2147483647), 0))
        first = following
