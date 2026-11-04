"""Fused CUDA patch-local PCA5. Imported only for an explicit CUDA backend.

One warp per observed centre: exact 25-bit connectivity, float64 covariance,
closed-form symmetric eigensystem, and a guarded Torch/LAPACK fallback.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _pca5(XYZ, SUPPORT, CENTRES, NORMALS, VALID, COV, FALLBACK,
          PIXELS, OBJECTS, ROWS, COMPACT:tl.constexpr,
          H:tl.constexpr, W:tl.constexpr, ABS:tl.constexpr, REL:tl.constexpr):
    pid=tl.program_id(0)
    if COMPACT:
        source=tl.load(PIXELS+pid);cv=source//W;cu=source%W
        oid=tl.load(OBJECTS+pid)
    else:
        cv=tl.load(CENTRES+pid*2);cu=tl.load(CENTRES+pid*2+1)
    k=tl.arange(0,32);r=k//5;c=k%5
    v=cv+r-2;u=cu+c-2
    inside=(k<25)&(v>=0)&(v<H)&(u>=0)&(u<W)
    index=v*W+u
    right_inside=inside&(c<4)&(u+1<W)
    down_inside=inside&(r<4)&(v+1<H)
    if COMPACT:
        row=tl.load(ROWS+index,inside,other=-1)
        available=inside&(row>=0)
        available=available&(tl.load(OBJECTS+row,available,other=-1)==oid)
        rr=tl.load(ROWS+index+1,right_inside,other=-1)
        dr=tl.load(ROWS+index+W,down_inside,other=-1)
        rs=right_inside&(rr>=0);ds=down_inside&(dr>=0)
        rs=rs&(tl.load(OBJECTS+rr,rs,other=-1)==oid)
        ds=ds&(tl.load(OBJECTS+dr,ds,other=-1)==oid)
        rz=tl.load(XYZ+rr*3+2,rs,other=0).to(tl.float64)
        dz=tl.load(XYZ+dr*3+2,ds,other=0).to(tl.float64)
        x=tl.load(XYZ+row*3,available,other=0).to(tl.float64)
        y=tl.load(XYZ+row*3+1,available,other=0).to(tl.float64)
        z=tl.load(XYZ+row*3+2,available,other=0).to(tl.float64)
        centre_row=pid
    else:
        available=tl.load(SUPPORT+index,inside,other=0)
        x=tl.load(XYZ+index*3,inside,other=0).to(tl.float64)
        y=tl.load(XYZ+index*3+1,inside,other=0).to(tl.float64)
        z=tl.load(XYZ+index*3+2,inside,other=0).to(tl.float64)
        rs=tl.load(SUPPORT+index+1,right_inside,other=0)
        ds=tl.load(SUPPORT+index+W,down_inside,other=0)
        rz=tl.load(XYZ+(index+1)*3+2,right_inside,other=0).to(tl.float64)
        dz=tl.load(XYZ+(index+W)*3+2,down_inside,other=0).to(tl.float64)
        centre_row=cv*W+cu
    right_ok=available&rs&right_inside&(tl.abs(z-rz)<=ABS+REL*tl.minimum(z,rz))
    down_ok=available&ds&down_inside&(tl.abs(z-dz)<=ABS+REL*tl.minimum(z,dz))
    bits=1<<k
    right=tl.sum(tl.where(right_ok,bits,0),0)
    down=tl.sum(tl.where(down_ok,bits,0),0)
    reached=1<<12
    for _ in range(24):
        reached=reached|((reached&right)<<1)|((reached>>1)&right)|((reached&down)<<5)|((reached>>5)&down)
    component=(k<25)&((reached&bits)!=0)
    count=tl.sum(component.to(tl.int32),0)
    cx=tl.load(XYZ+centre_row*3).to(tl.float64)
    cy=tl.load(XYZ+centre_row*3+1).to(tl.float64)
    cz=tl.load(XYZ+centre_row*3+2).to(tl.float64)
    dx=tl.where(component,x-cx,0);dy=tl.where(component,y-cy,0);ddz=tl.where(component,z-cz,0)
    den=tl.maximum(count,1).to(tl.float64)
    mx=tl.sum(dx,0)/den;my=tl.sum(dy,0)/den;mz=tl.sum(ddz,0)/den
    xx=tl.sum(dx*dx,0)/den-mx*mx;yy=tl.sum(dy*dy,0)/den-my*my;zz=tl.sum(ddz*ddz,0)/den-mz*mz
    xy=tl.sum(dx*dy,0)/den-mx*my;xz=tl.sum(dx*ddz,0)/den-mx*mz;yz=tl.sum(dy*ddz,0)/den-my*mz
    for j in tl.static_range(9):
        value=tl.where(j==0,xx,tl.where(j==1,xy,tl.where(j==2,xz,tl.where(j==3,xy,tl.where(j==4,yy,tl.where(j==5,yz,tl.where(j==6,xz,tl.where(j==7,yz,zz))))))))
        tl.store(COV+pid*9+j,value)
    scale=tl.maximum(tl.maximum(tl.maximum(tl.abs(xx),tl.abs(yy)),tl.maximum(tl.abs(zz),tl.abs(xy))),tl.maximum(tl.abs(xz),tl.abs(yz)))
    s=tl.where(scale>0,scale,1.)
    a=xx/s;b=yy/s;d=zz/s;e=xy/s;f=xz/s;g=yz/s
    q=(a+b+d)/3.
    aa=a-q;bb=b-q;dd=d-q
    p=libdevice.sqrt((aa*aa+bb*bb+dd*dd+2*(e*e+f*f+g*g))/6.)
    pp=tl.where(p>0,p,1.)
    ca=aa/pp;cb=bb/pp;cd=dd/pp;ce=e/pp;cf=f/pp;cg=g/pp
    determinant=ca*(cb*cd-cg*cg)-ce*(ce*cd-cg*cf)+cf*(ce*cg-cb*cf)
    phi=libdevice.acos(tl.minimum(1.,tl.maximum(-1.,determinant/2.)))/3.
    largest=q+2*p*libdevice.cos(phi)
    smallest=q+2*p*libdevice.cos(phi+2.0943951023931953)
    middle=3*q-smallest-largest
    ra=a-smallest;rb=b-smallest;rd=d-smallest
    x1=e*g-f*rb;y1=f*e-ra*g;z1=ra*rb-e*e
    x2=e*rd-f*g;y2=f*f-ra*rd;z2=ra*g-e*f
    x3=rb*rd-g*g;y3=g*f-e*rd;z3=e*g-rb*f
    n1=x1*x1+y1*y1+z1*z1;n2=x2*x2+y2*y2+z2*z2;n3=x3*x3+y3*y3+z3*z3
    use1=(n1>=n2)&(n1>=n3);use2=n2>=n3
    nx=tl.where(use1,x1,tl.where(use2,x2,x3));ny=tl.where(use1,y1,tl.where(use2,y2,y3));nz=tl.where(use1,z1,tl.where(use2,z2,z3))
    length=libdevice.sqrt(nx*nx+ny*ny+nz*nz)
    safe_length=tl.where(length>0,length,1.)
    nx=nx/safe_length;ny=ny/safe_length;nz=nz/safe_length
    rx=a*nx+e*ny+f*nz-smallest*nx
    ry=e*nx+b*ny+g*nz-smallest*ny
    rz=f*nx+g*ny+d*nz-smallest*nz
    residual=libdevice.sqrt(rx*rx+ry*ry+rz*rz)
    threshold=tl.maximum(1e-16,largest*scale*1e-8)
    rank=(middle*scale>threshold)&((middle-smallest)*scale>threshold)
    delicate=(middle*scale<=100*threshold)|((middle-smallest)*scale<=100*threshold)
    fallback=(count>=6)&((length<1e-10)|(residual>1e-10)|delicate|(residual!=residual))
    good=(count>=6)&rank&~fallback
    flip=nx*cx+ny*cy+nz*cz>0
    sign=tl.where(flip,-1.,1.)
    tl.store(NORMALS+centre_row*3,nx*sign,good)
    tl.store(NORMALS+centre_row*3+1,ny*sign,good)
    tl.store(NORMALS+centre_row*3+2,nz*sign,good)
    tl.store(VALID+centre_row,good)
    tl.store(FALLBACK+pid,fallback)


def fused_normals(xyz,support,parameters):
    h,w=support.shape
    normals=torch.full_like(xyz,float('nan'));valid=torch.zeros_like(support)
    # torch.nonzero commonly returns column-major strides (1,N); the kernel
    # explicitly consumes adjacent (v,u) pairs.
    centres=support.nonzero().contiguous()
    if not len(centres):return normals,valid
    covariance=torch.empty((len(centres),3,3),dtype=torch.float64,device=xyz.device)
    fallback=torch.empty((len(centres),),dtype=torch.bool,device=xyz.device)
    _pca5[(len(centres),)](xyz,support,centres,normals,valid,covariance,fallback,
                           support,support,support,False,h,w,
                           parameters.depth_jump_abs_m,parameters.depth_jump_rel,num_warps=1,enable_fp_fusion=False)
    idx=fallback.nonzero().flatten()
    if len(idx):
        values,basis=torch.linalg.eigh(covariance[idx])
        threshold=torch.maximum(torch.full_like(values[:,2],1e-16),values[:,2]*1e-8)
        good=(values[:,1]>threshold)&(values[:,1]-values[:,0]>threshold)&torch.isfinite(values).all(1)
        vu=centres[idx[good]];vectors=basis[good,:,0]
        vectors/=torch.linalg.vector_norm(vectors,dim=1)[:,None]
        flip=(vectors*xyz[vu[:,0],vu[:,1]]).sum(1)>0
        vectors=torch.where(flip[:,None],-vectors,vectors)
        normals[vu[:,0],vu[:,1]]=vectors.to(torch.float32);valid[vu[:,0],vu[:,1]]=True
    return normals,valid


@triton.jit
def _features(XYZ,VD,MASK,SUPPORT,NORMALS,NV,JM,PV,DISC,MB,VB,IB,VAR,VV,COUNT,
              PIXELS,OBJECTS,ROWS,DEPTH,N:tl.constexpr,COMPACT:tl.constexpr,
              H:tl.constexpr,W:tl.constexpr,ABS:tl.constexpr,REL:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);inside=i<N
    if COMPACT:
        source=tl.load(PIXELS+i,inside,other=0);oid=tl.load(OBJECTS+i,inside,other=0)
        mask=inside;support=inside;vd=inside
    else:
        source=i
        vd=tl.load(VD+i,inside,other=0);mask=tl.load(MASK+i,inside,other=0)
        support=tl.load(SUPPORT+i,inside,other=0)
    v=source//W;u=source%W
    nv=tl.load(NV+i,inside,other=0)
    z=tl.where(vd,tl.load(XYZ+i*3+2,inside,other=0),0.)
    nx=tl.where(nv,tl.load(NORMALS+i*3,inside,other=0),0.)
    ny=tl.where(nv,tl.load(NORMALS+i*3+1,inside,other=0),0.)
    nz=tl.where(nv,tl.load(NORMALS+i*3+2,inside,other=0),0.)
    jump_m=tl.full((B,),0.,tl.float32);variation=tl.full((B,),0.,tl.float32)
    pair_valid=tl.full((B,),False,tl.int1);disc=pair_valid;mb=pair_valid;vb=pair_valid
    count=tl.full((B,),0,tl.int32)
    for direction in tl.static_range(4):
        offset=tl.where(direction==0,-1,tl.where(direction==1,1,tl.where(direction==2,-W,W)))
        available=inside&tl.where(direction==0,u>0,tl.where(direction==1,u+1<W,tl.where(direction==2,v>0,v+1<H)))
        q=source+offset
        if COMPACT:
            qraw=tl.load(DEPTH+q,available,other=0)
            qvd=available&(qraw>0)&(qraw<float('inf'))
            qz=tl.where(qvd,qraw,0.)
            qm=tl.load(MASK+oid*H*W+q,available,other=0)
            qr=tl.load(ROWS+q,available,other=-1)
            own=available&(qr>=0)
            own=own&(tl.load(OBJECTS+qr,own,other=-1)==oid)
            qnv=tl.load(NV+qr,own,other=0)
        else:
            qr=q
            qvd=tl.load(VD+q,available,other=0);qm=tl.load(MASK+q,available,other=0)
            qnv=tl.load(NV+q,available,other=0)
            qz=tl.where(qvd,tl.load(XYZ+q*3+2,available,other=0),0.)
        dz=tl.abs(z-qz);both_depth=available&vd&qvd
        pv=support&both_depth
        pair_valid=pair_valid|pv
        jump_m=tl.maximum(jump_m,tl.where(pv,dz,0.))
        threshold=tl.full((),ABS,tl.float32)+tl.full((),REL,tl.float32)*tl.minimum(z,qz)
        disc=disc|(support&both_depth&(dz>threshold))
        mb=mb|(available&mask&~qm);vb=vb|(available&support&~qvd)
        qx=tl.load(NORMALS+qr*3,available&qnv,other=0)
        qy=tl.load(NORMALS+qr*3+1,available&qnv,other=0)
        qnz=tl.load(NORMALS+qr*3+2,available&qnv,other=0)
        cx=ny*qnz-nz*qy;cy=nz*qx-nx*qnz;cz=nx*qy-ny*qx
        length=libdevice.sqrt(cx*cx+cy*cy+cz*cz)
        dot=nx*qx+ny*qy+nz*qnz
        both=available&nv&qnv
        angle=tl.where(both,libdevice.atan2(length,dot),0.)
        variation=tl.maximum(variation,angle);count=count+both.to(tl.int32)
    vv=nv&(count>0)
    image_boundary=mask&((v==0)|(v==H-1)|(u==0)|(u==W-1))
    tl.store(JM+i,tl.where(pair_valid,jump_m,float('nan')),inside)
    tl.store(PV+i,pair_valid,inside);tl.store(DISC+i,disc,inside)
    tl.store(MB+i,mb,inside);tl.store(VB+i,vb,inside);tl.store(IB+i,image_boundary,inside)
    tl.store(VAR+i,tl.where(vv,variation,float('nan')),inside);tl.store(VV+i,vv,inside);tl.store(COUNT+i,count,inside)


def fused_features(xyz,valid_depth,mask,support,normals,normal_valid,parameters):
    shape=mask.shape
    result=dict(object_mask=mask,valid_points=support,normals=normals,normal_valid=normal_valid)
    names=['depth_jump_m','depth_pair_valid','depth_discontinuity','mask_boundary','depth_validity_boundary',
           'image_boundary','normal_variation_rad','normal_variation_valid','normal_neighbor_count']
    for name in names:
        dtype=torch.float32 if name in ('depth_jump_m','normal_variation_rad') else torch.uint8 if name=='normal_neighbor_count' else torch.bool
        result[name]=torch.empty(shape,dtype=dtype,device=xyz.device)
    _features[(triton.cdiv(mask.numel(),256),)](xyz,valid_depth,mask,support,normals,normal_valid,
              *[result[name] for name in names],support,support,support,xyz,mask.numel(),False,
              *shape,parameters.depth_jump_abs_m,parameters.depth_jump_rel,256,
              num_warps=4,enable_fp_fusion=False)
    return result


def compact_rows(pixels, shape):
    """Raster index -> compact point row. This is an integer map, not XYZ."""
    rows=torch.full(shape,-1,dtype=torch.int64,device=pixels.device)
    rows.view(-1)[pixels]=torch.arange(len(pixels),device=pixels.device)
    return rows


def fused_normals_compact(points,pixels,objects,rows,parameters):
    normals=torch.full_like(points,float('nan'))
    valid=torch.zeros(len(points),dtype=torch.bool,device=points.device)
    if not len(points):return normals,valid
    covariance=torch.empty((len(points),3,3),dtype=torch.float64,device=points.device)
    fallback=torch.empty(len(points),dtype=torch.bool,device=points.device)
    _pca5[(len(points),)](points,rows,rows,normals,valid,covariance,fallback,
        pixels,objects,rows,True,*rows.shape,parameters.depth_jump_abs_m,
        parameters.depth_jump_rel,num_warps=1,enable_fp_fusion=False)
    idx=fallback.nonzero().flatten()
    if len(idx):
        values,basis=torch.linalg.eigh(covariance[idx])
        threshold=torch.maximum(torch.full_like(values[:,2],1e-16),values[:,2]*1e-8)
        good=(values[:,1]>threshold)&(values[:,1]-values[:,0]>threshold)&torch.isfinite(values).all(1)
        chosen=idx[good];vectors=basis[good,:,0]
        vectors/=torch.linalg.vector_norm(vectors,dim=1)[:,None]
        vectors=torch.where((vectors*points[chosen]).sum(1)[:,None]>0,-vectors,vectors)
        normals[chosen]=vectors.to(torch.float32);valid[chosen]=True
    return normals,valid


def fused_features_compact(points,pixels,objects,rows,depth,masks,normals,valid,parameters):
    n=len(points)
    result=dict(normals=normals,normal_valid=valid)
    names=['depth_jump_m','depth_pair_valid','depth_discontinuity','mask_boundary',
        'depth_validity_boundary','image_boundary','normal_variation_rad',
        'normal_variation_valid','normal_neighbor_count']
    for name in names:
        dtype=torch.float32 if name in ('depth_jump_m','normal_variation_rad') else torch.uint8 if name=='normal_neighbor_count' else torch.bool
        result[name]=torch.empty(n,dtype=dtype,device=points.device)
    if n:
        _features[(triton.cdiv(n,256),)](points,rows,masks,rows,normals,valid,
            *[result[name] for name in names],pixels,objects,rows,depth,n,True,
            *depth.shape,parameters.depth_jump_abs_m,parameters.depth_jump_rel,256,
            num_warps=4,enable_fp_fusion=False)
    return result
