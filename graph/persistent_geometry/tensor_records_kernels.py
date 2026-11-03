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


@triton.jit
def identity_keys(PIXELS, CAMERAS, PREFIX, LENGTHS, POWERS, KEYS, N,
                  WIDTH: tl.constexpr, LIMBS: tl.constexpr, B: tl.constexpr):
    """Big-endian seven-byte limbs of camera + ':' + decimal pixel + zero pad."""
    i = tl.program_id(0)*B+tl.arange(0,B)
    row = i//LIMBS; limb = i%LIMBS; valid = row < N
    pixel = tl.load(PIXELS+row, valid, other=0)
    camera = tl.load(CAMERAS+row, valid, other=0)
    length = tl.load(LENGTHS+camera, valid, other=0)
    digits = tl.full((B,),1,tl.int32)
    for exponent in tl.static_range(1,19):
        digits += (pixel >= 10**exponent).to(tl.int32)
    key = tl.full((B,),0,tl.int64)
    for byte in tl.static_range(7):
        position = limb*7+byte
        prefix = tl.load(PREFIX+camera*WIDTH+position, valid & (position < length), other=0).to(tl.int64)
        decimal_position = position-length
        exponent = digits-1-decimal_position
        power = tl.load(POWERS+tl.minimum(18,tl.maximum(exponent,0)))
        number = pixel//power%10+48
        value = tl.where(position < length, prefix,
            tl.where((decimal_position >= 0) & (decimal_position < digits), number, 0))
        key = (key << 8) | value
    tl.store(KEYS+i,key,valid)


@triton.jit
def final_fps(X, RANK, OUT, SCRATCH, N, COUNT: tl.constexpr, B: tl.constexpr):
    """Stable-ID seed/ties, FP64 NumPy three-term squared distance, one launch."""
    lane = tl.arange(0,B)
    first_rank = tl.full((),9223372036854775807,tl.int64)
    current = tl.full((),2147483647,tl.int32)
    for chunk in range(tl.cdiv(N,B)):
        i = chunk*B+lane
        rank = tl.load(RANK+i,i < N,other=9223372036854775807)
        local_rank = tl.min(rank,0)
        local_row = tl.min(tl.where((i < N) & (rank == local_rank),i,2147483647),0)
        current = tl.where(local_rank < first_rank,local_row,current)
        first_rank = tl.minimum(first_rank,local_rank)
    if N <= B:
        rank = tl.load(RANK+lane,lane < N,other=9223372036854775807)
        x = tl.load(X+lane*3,lane < N,other=0)
        y = tl.load(X+lane*3+1,lane < N,other=0)
        z = tl.load(X+lane*3+2,lane < N,other=0)
        distance = tl.full((B,),float('inf'),tl.float64)
        for step in range(COUNT):
            tl.store(OUT+step,current)
            if step+1 < COUNT:
                dx = x-tl.load(X+current*3)
                dy = y-tl.load(X+current*3+1)
                dz = z-tl.load(X+current*3+2)
                distance = tl.minimum(distance,(dx*dx+dy*dy)+dz*dz)
                distance = tl.where(lane == current,-1.,distance)
                best = tl.max(tl.where(lane < N,distance,-1.),0)
                best_rank = tl.min(tl.where((lane < N) & (distance == best),rank,9223372036854775807),0)
                current = tl.min(tl.where((lane < N) & (distance == best) & (rank == best_rank),lane,2147483647),0)
    else:
        # Bounded register footprint for larger post-dedup clouds.
        for step in range(COUNT):
            tl.store(OUT+step,current)
            if step+1 < COUNT:
                x = tl.load(X+current*3); y = tl.load(X+current*3+1); z = tl.load(X+current*3+2)
                best = tl.full((),-1.,tl.float64)
                best_rank = tl.full((),9223372036854775807,tl.int64)
                next_row = tl.full((),2147483647,tl.int32)
                for chunk in range(tl.cdiv(N,B)):
                    i = chunk*B+lane
                    dx = tl.load(X+i*3,i < N,other=0)-x
                    dy = tl.load(X+i*3+1,i < N,other=0)-y
                    dz = tl.load(X+i*3+2,i < N,other=0)-z
                    if step == 0:
                        old = tl.full((B,),float('inf'),tl.float64)
                    else:
                        old = tl.load(SCRATCH+i,i < N,other=-1.)
                    distance = tl.minimum(old,(dx*dx+dy*dy)+dz*dz)
                    distance = tl.where(i == current,-1.,distance)
                    tl.store(SCRATCH+i,distance,i < N)
                    rank = tl.load(RANK+i,i < N,other=9223372036854775807)
                    local = tl.max(tl.where(i < N,distance,-1.),0)
                    local_rank = tl.min(tl.where((i < N) & (distance == local),rank,9223372036854775807),0)
                    local_row = tl.min(tl.where((i < N) & (distance == local) & (rank == local_rank),i,2147483647),0)
                    take = (local > best) | ((local == best) & (local_rank < best_rank))
                    next_row = tl.where(take,local_row,next_row)
                    best_rank = tl.where(take,local_rank,best_rank)
                    best = tl.maximum(best,local)
                current = next_row
