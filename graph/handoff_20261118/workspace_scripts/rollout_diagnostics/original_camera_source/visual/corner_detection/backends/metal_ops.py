"""Fused MPS kernels operating directly on Torch tensor storage.

Scalar thresholds and sampling schedules match torch_ops. No GT, hidden
state, sensor-result caching, CPU geometry fallback or Metal readback is used.
CPU/CUDA continue to use the portable Torch implementations.
"""

from functools import lru_cache
import numpy as np
import torch


SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline float3 xyz_at(device const float* xyz, int p) {
    return float3(xyz[3*p], xyz[3*p+1], xyz[3*p+2]);
}
inline float3 normal_at(device const float* planes, int p) {
    return float3(planes[4*p], planes[4*p+1], planes[4*p+2]);
}
inline int pixel(int2 p) { return p.y*512+p.x; }
inline bool in_image(int2 p) { return all(p >= 0) && all(p < 512); }
inline float3 ray_at(float2 uv, device const float* k) {
    return float3((uv.x-k[2])/k[0], (uv.y-k[5])/k[4], 1.f);
}
inline float2 project_at(float3 p, device const float* k) {
    return p.xy/p.z*float2(k[0], k[4])+float2(k[2], k[5]);
}

kernel void backproject(device const float* depth, device const float* k,
    device float* xyz, uint i [[thread_position_in_grid]]) {
    float z=depth[i];
    xyz[3*i]=(float(i%512)-k[2])*z/k[0];
    xyz[3*i+1]=(float(i/512)-k[5])*z/k[4]; xyz[3*i+2]=z;
}

kernel void label_planes(device const float* xyz, device const bool* support,
    device const float* planes, device bool* masks, constant int& count,
    constant float& tolerance, uint i [[thread_position_in_grid]]) {
    float best=INFINITY; int winner=0; float3 point=xyz_at(xyz,i);
    for(int p=0;p<count;p++) {
        float d=abs(dot(point,normal_at(planes,p))+planes[4*p+3]);
        if(d<best) { best=d;winner=p; }
    }
    for(int p=0;p<count;p++) masks[p*262144+i]=support[i] && best<tolerance && p==winner;
}
kernel void morphology(device const bool* src, device bool* dst,
    constant int& erosion, uint i [[thread_position_in_grid]]) {
    int plane=i/262144, local=i%262144; int2 p=int2(local%512,local/512);
    bool result=bool(erosion);
    for(int y=-1;y<=1;y++) for(int x=-1;x<=1;x++) {
        int2 q=p+int2(x,y); bool v=in_image(q)?src[plane*262144+pixel(q)]:bool(erosion);
        result=erosion?(result&&v):(result||v);
    }
    dst[i]=result;
}

kernel void robust_lines(device const float* points, device const int* offsets,
    device float* out, uint i [[thread_position_in_grid]]) {
    int begin=offsets[i],end=offsets[i+1],count=end-begin;
    float2 center=0.f;
    for(int j=begin;j<end;j++) center+=float2(points[2*j],points[2*j+1]);
    center/=float(count);
    float a=0,b=0,c=0;
    for(int j=begin;j<end;j++) {
        float2 d=float2(points[2*j],points[2*j+1])-center;
        a+=d.x*d.x;b+=d.x*d.y;c+=d.y*d.y;
    }
    float angle=.5f*atan2(2.f*b,a-c);float2 normal=float2(-sin(angle),cos(angle));
    float2 old_center=center,old_normal=normal; int keep=0; float2 sum=0.f;
    for(int j=begin;j<end;j++) {
        float2 p=float2(points[2*j],points[2*j+1]);
        if(abs(dot(p-center,normal))<1.3f) {keep++;sum+=p;}
    }
    bool refit=keep>=5 && keep!=count;
    if(refit) {
        center=sum/float(keep);a=0;b=0;c=0;
        for(int j=begin;j<end;j++) {
            float2 p=float2(points[2*j],points[2*j+1]);
            if(abs(dot(p-old_center,old_normal))>=1.3f) continue;
            float2 d=p-center;a+=d.x*d.x;b+=d.x*d.y;c+=d.y*d.y;
        }
        angle=.5f*atan2(2.f*b,a-c);normal=float2(-sin(angle),cos(angle));
    }
    float error=0;
    for(int j=begin;j<end;j++) {
        float2 p=float2(points[2*j],points[2*j+1]);
        if(refit && abs(dot(p-old_center,old_normal))>=1.3f) continue;
        float residual=dot(p-center,normal);error+=residual*residual;
    }
    out[4*i]=normal.x;out[4*i+1]=normal.y;out[4*i+2]=-dot(normal,center);
    out[4*i+3]=sqrt(error/float(refit?keep:count));
}

inline bool visible_one(float3 p,float2 uv,device const bool* support,device const float* xyz) {
    if(!all(isfinite(uv)) || !all(isfinite(p))) return false;
    int2 center=int2(rint(uv));
    if(any(center<2)||any(center>=510))return false;
    for(int y=-2;y<=2;y++) for(int x=-2;x<=2;x++) {
        int2 q=center+int2(x,y);float2 d=float2(q)-uv;
        if(dot(d,d)>2.25f || !support[pixel(q)])continue;
        float3 gap=xyz_at(xyz,pixel(q))-p;
        if(dot(gap,gap)<.000016f) return true;
    }
    return false;
}
inline bool unoccluded_one(float3 p,float2 uv,device const float* depth,device const bool* valid) {
    if(!all(isfinite(uv)) || !all(isfinite(p)))return false;
    int2 corner=int2(floor(uv));
    if(any(corner<1)||any(corner>=510))return false;
    int count=0;
    for(int y=0;y<2;y++)for(int x=0;x<2;x++) {
        int j=pixel(corner+int2(x,y));count+=valid[j] && depth[j]<p.z-.020f;
    }
    return count<2;
}
kernel void visibility(device const float* points,device const float* uv,
    device const bool* support,device const float* xyz,device bool* out,
    uint i [[thread_position_in_grid]]) {
    out[i]=visible_one(xyz_at(points,i),float2(uv[2*i],uv[2*i+1]),support,xyz);
}
kernel void footprint(device const float* points,device const float* uv,
    device const float* depth,device const bool* valid,device bool* out,
    uint i [[thread_position_in_grid]]) {
    out[i]=unoccluded_one(xyz_at(points,i),float2(uv[2*i],uv[2*i+1]),depth,valid);
}

inline float3 snap_one(float3 point,int primary,device const float* planes,
    device const bool* eligible,int count,thread int& supporting) {
    supporting=1;if(count<2 || !all(isfinite(point)))return point;
    int second=-1,third=-1; float best=INFINITY;
    float3 a=normal_at(planes,primary);
    for(int i=0;i<count;i++) {
        float3 b=normal_at(planes,i);float distance=abs(dot(point,b)+planes[4*i+3]);
        if(eligible[i] && distance<=.004f && abs(dot(a,b))<.25f && distance<best) {best=distance;second=i;}
    }
    if(second>=0) {
        best=INFINITY;float3 b=normal_at(planes,second);
        for(int i=0;i<count;i++) {
            float3 c=normal_at(planes,i);float distance=abs(dot(point,c)+planes[4*i+3]);
            if(eligible[i] && distance<=.004f && abs(dot(a,c))<.25f && abs(dot(b,c))<.25f && distance<best) {best=distance;third=i;}
        }
    }
    float r0=dot(a,point)+planes[4*primary+3];float3 delta;
    if(second<0) delta=a*(r0/dot(a,a));
    else {
        float3 b=normal_at(planes,second);float r1=dot(b,point)+planes[4*second+3];
        if(third<0) {
            float aa=dot(a,a),ab=dot(a,b),bb=dot(b,b),det=aa*bb-ab*ab;
            delta=a*((r0*bb-r1*ab)/det)+b*((r1*aa-r0*ab)/det);
        } else {
            float3 c=normal_at(planes,third);float r2=dot(c,point)+planes[4*third+3];
            delta=(r0*cross(b,c)+r1*cross(c,a)+r2*cross(a,b))/dot(a,cross(b,c));
        }
    }
    if(dot(delta,delta)<.00002025f) {supporting=1+(second>=0)+(third>=0);return point-delta;}
    return point;
}
kernel void snap_points(device const float* points,device const long* primary,
    device const float* planes,device const bool* eligible,constant int& count,
    device float* out,device long* support,uint i [[thread_position_in_grid]]) {
    int supporting=1;float3 p=snap_one(xyz_at(points,i),int(primary[i]),planes,eligible,count,supporting);
    out[3*i]=p.x;out[3*i+1]=p.y;out[3*i+2]=p.z;support[i]=supporting;
}

inline bool edges_for_mask(float2 uv,float2 e0,float2 e1,float3 normal,float offset,
    int plane,device const bool* masks,device const bool* object_mask,
    device const float* depth,device const bool* valid,device const float* xyz,
    device const float* k,device const bool* neutral,thread int4& counts,int distance_mask) {
    counts=0;bool ok=all(isfinite(uv));
    for(int leg=0;leg<2;leg++) {
        float2 vector=(leg==0?e0:e1)-uv;float length=sqrt(dot(vector,vector));
        if(!isfinite(length))return false;
        if(length<3.f)ok=false;
        float2 direction=vector/max(length,1.e-12f),side=float2(-direction.y,direction.x);
        int2 pixels[12];int sums[2]={0,0};
        for(int sign=0;sign<2;sign++)for(int j=0;j<6;j++) {
            float along=1.5f+(min(length*.75f,8.f)-1.5f)*(float(j)/5.f);
            float2 raw=rint(uv+along*direction+(sign==0?-2.f:2.f)*side);
            if(any(raw<1.f)||any(raw>510.f))ok=false;
            int2 p=int2(clamp(raw,0.f,511.f));pixels[sign*6+j]=p;
            int index=pixel(p);
            sums[sign]+=distance_mask?(object_mask[index] && abs(dot(xyz_at(xyz,index),normal)+offset)<.0012f):masks[plane*262144+index];
        }
        int outside=sums[0]<sums[1]?0:1,physical=0,occluding=0;
        for(int j=0;j<6;j++) {
            int2 p=pixels[outside*6+j],q=pixels[(1-outside)*6+j];int ip=pixel(p),iq=pixel(q);
            float3 point=xyz_at(xyz,ip);float denominator=dot(ray_at(float2(p),k),normal);
            float predicted=abs(denominator)>.02f?-offset/denominator:NAN;
            float delta=depth[ip]-predicted;float3 gap=point-xyz_at(xyz,iq);
            bool away=abs(dot(point,normal)+offset)>.0012f;
            bool crease=(object_mask[ip]||!neutral[ip]) && away && dot(gap,gap)<.000144f;
            bool background=!valid[ip] || (delta>.002f && away);
            bool inside_mask=distance_mask?(object_mask[iq] && abs(dot(xyz_at(xyz,iq),normal)+offset)<.0012f):masks[plane*262144+iq];
            physical+=(crease||background) && valid[iq] && inside_mask;
            occluding+=valid[ip] && delta<-.002f && !crease;
        }
        counts[leg*2]=physical;counts[leg*2+1]=occluding;
        ok=ok && physical>=3 && occluding<=1;
    }
    return ok;
}
inline bool edges_one(float2 uv,float2 e0,float2 e1,float3 normal,float offset,
    int plane,device const bool* masks,device const bool* object_mask,
    device const float* depth,device const bool* valid,device const float* xyz,
    device const float* k,device const bool* neutral,thread int4& counts) {
    return edges_for_mask(uv,e0,e1,normal,offset,plane,masks,object_mask,depth,valid,xyz,k,neutral,counts,0);
}
kernel void physical_edges(device const float* uv,device const float* adjacent,
    device const float* planes,device const long* plane_ids,device const bool* masks,
    device const bool* object_mask,device const float* depth,device const bool* valid,
    device const float* xyz,device const float* k,device const bool* neutral,
    device bool* out,device long* evidence,uint i [[thread_position_in_grid]]) {
    int pi=int(plane_ids[i]);int4 counts;
    out[i]=edges_one(float2(uv[2*i],uv[2*i+1]),float2(adjacent[4*i],adjacent[4*i+1]),
        float2(adjacent[4*i+2],adjacent[4*i+3]),normal_at(planes,pi),planes[4*pi+3],pi,
        masks,object_mask,depth,valid,xyz,k,neutral,counts);
    for(int j=0;j<4;j++)evidence[4*i+j]=counts[j];
}

inline void write3(device float* out,uint i,float3 p) {
    out[3*i]=p.x;out[3*i+1]=p.y;out[3*i+2]=p.z;
}
inline void write2(device float* out,uint i,float2 p) {out[2*i]=p.x;out[2*i+1]=p.y;}
inline int contact_count(float3 point,float2 uv,int radius,device const bool* object_mask,
    device const float* xyz,device const float* depth,device const bool* valid,device const bool* neutral) {
    int2 center=int2(rint(uv));int count=0;
    for(int y=-radius;y<=radius;y++)for(int x=-radius;x<=radius;x++) {
        int2 q=center+int2(x,y);if(!in_image(q))continue;int j=pixel(q);
        float3 gap=xyz_at(xyz,j)-point;
        count+=valid[j] && neutral[j] && !object_mask[j] && dot(gap,gap)<.000036f;
    }
    return count;
}

kernel void boundary_batch(device const float* uv_in,device const float* adjacent,
    device const long* primary,device const int* method,device const float* planes,
    device const bool* eligible,constant int& plane_count,device const bool* masks,
    device const bool* object_mask,device const bool* support,device const float* depth,
    device const bool* valid,device const float* xyz,device const float* k,device const bool* neutral,
    constant float& maximum_cosine,constant float& maximum_gap,constant int& recovery,
    device float* points_out,device float* uv_out,device long* support_out,
    device bool* core_out,device bool* recovered_out,uint i [[thread_position_in_grid]]) {
    core_out[i]=false;recovered_out[i]=false;support_out[i]=1;
    write3(points_out,i,float3(0));write2(uv_out,i,float2(0));
    int pi=int(primary[i]);float3 normal=normal_at(planes,pi);float offset=planes[4*pi+3];
    float2 uv=float2(uv_in[2*i],uv_in[2*i+1]);float3 ray=ray_at(uv,k);float denom=dot(ray,normal);
    if(abs(denom)<.08f)return;
    float3 point=ray*(-offset/denom);if(point.z<=0.f || !all(isfinite(point)))return;
    int2 center=int2(rint(uv));if(any(center<3)||any(center>=510))return;
    float2 e0=float2(adjacent[4*i],adjacent[4*i+1]),e1=float2(adjacent[4*i+2],adjacent[4*i+3]);
    float3 r0=ray_at(e0,k),r1=ray_at(e1,k);
    float3 a=r0*(-offset/dot(r0,normal))-point,b=r1*(-offset/dot(r1,normal))-point;
    float cosine=abs(dot(a,b))/(sqrt(dot(a,a))*sqrt(dot(b,b))+1.e-12f);
    if(!isfinite(cosine))return;
    int observed=0,occluded=0;float gap=INFINITY;
    for(int y=-3;y<=3;y++)for(int x=-3;x<=3;x++) {
        int2 q=center+int2(x,y);if(!in_image(q))continue;int j=pixel(q);
        if(masks[pi*262144+j]) {observed++;float2 delta=float2(q)-uv;gap=min(gap,dot(delta,delta));}
        occluded+=valid[j] && depth[j]<point.z-.020f;
    }
    if(observed<5 || sqrt(gap)>maximum_gap || contact_count(point,uv,3,object_mask,xyz,depth,valid,neutral)>=3)return;
    int supporting;float3 snapped=snap_one(point,pi,planes,eligible,plane_count,supporting);
    float2 image=project_at(snapped,k);write3(points_out,i,snapped);write2(uv_out,i,image);support_out[i]=supporting;
    bool boundary=method[i]==0;
    core_out[i]=boundary && cosine<=.28f && occluded<3;
    if(!recovery || cosine>maximum_cosine)return;
    int4 evidence;bool physical=edges_one(uv,e0,e1,normal,offset,pi,masks,object_mask,depth,valid,xyz,k,neutral,evidence);
    if(!(boundary && cosine<=.28f) && !physical)return;
    if(occluded>=3 && !(physical && visible_one(point,uv,support,xyz)))return;
    float2 shift=image-uv;
    if(!boundary && (dot(shift,shift)>4.f || !visible_one(snapped,image,support,xyz)))return;
    if(method[i]==2 && (any(uv<8.f)||any(uv>=504.f)))return;
    recovered_out[i]=unoccluded_one(snapped,image,depth,valid);
}

kernel void envelope_batch(device const float* points_in,device const float* uv_in,
    device const float* adjacent,device const long* primary,device const float* planes,
    device const bool* eligible,constant int& plane_count,device const bool* masks,
    device const bool* object_mask,device const bool* support,device const float* depth,
    device const bool* valid,device const float* xyz,device const float* k,device const bool* neutral,
    device float* points_out,device float* uv_out,device long* support_out,device bool* selected,
    uint i [[thread_position_in_grid]]) {
    float3 point=xyz_at(points_in,i);float2 uv=float2(uv_in[2*i],uv_in[2*i+1]);int pi=int(primary[i]);
    int4 evidence;bool physical=edges_one(uv,float2(adjacent[4*i],adjacent[4*i+1]),
        float2(adjacent[4*i+2],adjacent[4*i+3]),normal_at(planes,pi),planes[4*pi+3],pi,
        masks,object_mask,depth,valid,xyz,k,neutral,evidence);
    bool keep=physical && visible_one(point,uv,support,xyz) && contact_count(point,uv,2,object_mask,xyz,depth,valid,neutral)<3;
    int count;float3 snapped=snap_one(point,pi,planes,eligible,plane_count,count);float2 image=project_at(snapped,k);
    write3(points_out,i,snapped);write2(uv_out,i,image);support_out[i]=count;float2 shift=image-uv;
    selected[i]=keep && all(image>=8.f) && all(image<504.f) && dot(shift,shift)<=4.f &&
        visible_one(snapped,image,support,xyz) && unoccluded_one(snapped,image,depth,valid);
}

kernel void junction_batch(device const float* points,device const long* indices,
    device const bool* masks,device const bool* cores,device const bool* support,
    device const float* xyz,device const float* depth,device const bool* valid,device const float* k,
    device float* uv_out,device bool* selected,device bool* recovered,uint i [[thread_position_in_grid]]) {
    float3 point=xyz_at(points,i);float2 uv=project_at(point,k);write2(uv_out,i,uv);
    selected[i]=false;recovered[i]=false;
    if(!all(isfinite(uv)) || point.z<=0.f)return;int2 center=int2(rint(uv));
    if(any(center<6)||any(center>=506))return;
    for(int leg=0;leg<3;leg++) {
        int plane=int(indices[3*i+leg]),count=0,core_count=0;
        for(int y=-4;y<=4;y++)for(int x=-4;x<=4;x++) {
            int j=pixel(center+int2(x,y));float3 gap=xyz_at(xyz,j)-point;
            count+=masks[plane*262144+j] && support[j] && dot(gap,gap)<.000036f;
        }
        if(count<3)return;
        for(int y=-6;y<=6;y++)for(int x=-6;x<=6;x++)core_count+=cores[plane*262144+pixel(center+int2(x,y))];
        if(core_count<2)return;
    }
    int occluded=0;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
        int j=pixel(center+int2(x,y));occluded+=valid[j] && depth[j]<point.z-.004f;
    }
    selected[i]=occluded<2;recovered[i]=selected[i] && unoccluded_one(point,uv,depth,valid);
}

kernel void box_rgb_batch(device const float* uv_in,device const long* primary,
    device const float* axes,device const float* planes,device const bool* masks,
    device const bool* support,device const float* depth,device const bool* valid,
    device const float* xyz,device const float* k,device const bool* neutral,
    device float* points_out,device bool* selected,device bool* holes,uint i [[thread_position_in_grid]]) {
    selected[i]=false;holes[i]=false;write3(points_out,i,float3(0));
    float2 uv=float2(uv_in[2*i],uv_in[2*i+1]);int pi=int(primary[i]);
    float3 normal=normal_at(planes,pi);float offset=planes[4*pi+3];float3 ray=ray_at(uv,k);
    float denominator=dot(ray,normal);if(abs(denominator)<=.1f)return;
    float3 point=ray*(-offset/denominator);write3(points_out,i,point);
    if(point.z<=0.f || !visible_one(point,uv,support,xyz) || !unoccluded_one(point,uv,depth,valid))return;
    float3 axis0=xyz_at(axes,2*i),axis1=xyz_at(axes,2*i+1);
    int occupied=0,empty=0,occupied_id=0,empty_id=0;
    for(int q=0;q<4;q++) {
        float sx=q>=2?1.f:-1.f,sy=q%2?1.f:-1.f;int count=0;
        for(int j=0;j<4;j++) {
            float a=j>=2?.004f:.002f,b=j%2?.004f:.002f;
            float3 probe=point+((sx*a)*axis0+(sy*b)*axis1);
            float2 raw=rint(project_at(probe,k));if(any(raw<1.f)||any(raw>510.f))return;
            int index=pixel(int2(raw));
            count+=support[index] && abs(dot(xyz_at(xyz,index),normal)+offset)<.0012f;
        }
        if(count>=3) {occupied++;occupied_id=q;}
        if(count<=1) {empty++;empty_id=q;}
    }
    bool exterior=occupied==1 && empty==3,interior=occupied==3 && empty==1;holes[i]=interior;
    if(!exterior && !interior)return;
    int q=exterior?occupied_id:empty_id;
    float sx=q>=2?1.f:-1.f,sy=q%2?1.f:-1.f;
    float2 e0=project_at(point+(sx*.015f)*axis0,k),e1=project_at(point+(sy*.015f)*axis1,k);
    int4 evidence;bool physical=edges_one(uv,e0,e1,normal,offset,pi,masks,support,depth,valid,xyz,k,neutral,evidence);
    selected[i]=physical && contact_count(point,uv,2,support,xyz,depth,valid,neutral)<3;
}

kernel void box_cube_batch(device const float* base_points,device const float* rotation,
    device const float* signs,device const float* features,constant int& feature_count,
    device const float* planes,device const bool* eligible,constant int& plane_count,
    device const bool* masks,device const bool* support,device const float* depth,
    device const bool* valid,device const float* xyz,device const float* k,device const bool* neutral,
    device float* points_out,device float* uv_out,device bool* selected,uint i [[thread_position_in_grid]]) {
    selected[i]=false;float3 point=xyz_at(base_points,i);int nearest=-1;float best=INFINITY;
    for(int p=0;p<plane_count;p++)if(eligible[p]) {
        float d=abs(dot(point,normal_at(planes,p))+planes[4*p+3]);if(d<best){best=d;nearest=p;}
    }
    if(nearest<0) {write3(points_out,i,point);write2(uv_out,i,float2(0));return;}
    int supporting;if(best<.003f)point=snap_one(point,nearest,planes,eligible,plane_count,supporting);
    float2 uv=project_at(point,k);int closest=-1;best=INFINITY;
    for(int f=0;f<feature_count;f++) {
        float2 delta=uv-float2(features[2*f],features[2*f+1]);float d=dot(delta,delta);
        if(d<best) {best=d;closest=f;}
    }
    if(closest>=0 && best<6.25f) {
        int pi=-1;float distance=INFINITY;
        for(int p=0;p<plane_count;p++)if(eligible[p]) {
            float d=abs(dot(point,normal_at(planes,p))+planes[4*p+3]);if(d<distance){distance=d;pi=p;}
        }
        float2 feature=float2(features[2*closest],features[2*closest+1]);
        float3 ray=ray_at(feature,k);float denominator=dot(ray,normal_at(planes,pi));
        if(abs(denominator)>.1f) {
            float3 refined=ray*(-planes[4*pi+3]/denominator),gap=refined-point;
            if(dot(gap,gap)<.000025f) {point=refined;uv=feature;}
        }
    }
    write3(points_out,i,point);write2(uv_out,i,uv);
    bool image_corner=false;
    for(int f=0;f<feature_count;f++) {
        float2 d=uv-float2(features[2*f],features[2*f+1]);image_corner=image_corner || dot(d,d)<6.25f;
    }
    bool observed=visible_one(point,uv,support,xyz);
    if(!observed && image_corner && all(isfinite(uv))) {
        int2 center=int2(rint(uv));
        for(int y=-3;y<=3;y++)for(int x=-3;x<=3;x++) {
            int2 q=center+int2(x,y);if(!in_image(q))continue;int j=pixel(q);
            float3 d=xyz_at(xyz,j)-point;observed=observed || (support[j] && dot(d,d)<.000036f);
        }
    }
    if(!observed || !unoccluded_one(point,uv,depth,valid))return;
    bool physical=false;
    for(int axis=0;axis<3;axis++) {
        float3 column=float3(rotation[axis],rotation[3+axis],rotation[6+axis]);int matching=-1;
        for(int p=0;p<plane_count;p++)if(eligible[p] && abs(dot(normal_at(planes,p),column))>.97f &&
            abs(dot(point,normal_at(planes,p))+planes[4*p+3])<.0025f){matching=p;break;}
        if(matching<0)continue;
        float2 adjacent[2];int count=0;
        for(int tangent=0;tangent<3;tangent++)if(tangent!=axis) {
            float3 direction=float3(rotation[tangent],rotation[3+tangent],rotation[6+tangent]);
            adjacent[count++]=project_at(point-(signs[3*i+tangent]*.015f)*direction,k);
        }
        int4 evidence;
        physical=edges_for_mask(uv,adjacent[0],adjacent[1],normal_at(planes,matching),planes[4*matching+3],
            matching,masks,support,depth,valid,xyz,k,neutral,evidence,1);
        if(physical)break;
    }
    selected[i]=physical || image_corner;
}

kernel void score_planes(device const float* points,device const int* sample,
    device const int* triples,device float* hypotheses,device int* votes,
    constant int& samples,constant float& tolerance,uint i [[thread_position_in_grid]]) {
    float3 a=xyz_at(points,sample[triples[3*i]]);
    float3 b=xyz_at(points,sample[triples[3*i+1]]);
    float3 c=xyz_at(points,sample[triples[3*i+2]]);
    float3 normal=cross(b-a,c-a);float norm=sqrt(dot(normal,normal));
    normal/=max(norm,1.e-7f);float offset=-dot(normal,a);int count=0;
    if(norm>1.e-7f)for(int j=0;j<samples;j++)
        count+=abs(dot(xyz_at(points,sample[j]),normal)+offset)<tolerance;
    hypotheses[4*i]=normal.x;hypotheses[4*i+1]=normal.y;
    hypotheses[4*i+2]=normal.z;hypotheses[4*i+3]=offset;votes[i]=count;
}
kernel void select_plane(device const float* points,device const float* hypotheses,
    device const long* winner,device bool* selected,constant float& tolerance,
    uint i [[thread_position_in_grid]]) {
    int p=int(winner[0]);selected[i]=abs(dot(xyz_at(points,i),normal_at(hypotheses,p))+hypotheses[4*p+3])<tolerance;
}

// One cooperative group computes a centred covariance, the symmetric 3x3
// eigensystem, and optionally the next inlier mask. No host size readback is
// needed between the three refits. All array reads precede mask updates.
kernel void weighted_plane(device const float* points,device bool* selected,
    device float* out,device int* out_count,constant int& count,
    constant float& tolerance,constant int& minimum,constant int& update,
    uint t [[thread_index_in_threadgroup]]) {
    threadgroup float4 means[256];threadgroup float4 cov1[256];threadgroup float2 cov2[256];
    threadgroup float3 center;threadgroup float4 fitted;threadgroup int active;
    float4 total=0.f;
    for(int i=t;i<count;i+=256)if(selected[i])total+=float4(xyz_at(points,i),1.f);
    means[t]=total;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int stride=128;stride>0;stride/=2) {
        if(t<uint(stride))means[t]+=means[t+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(t==0) {active=int(means[0].w);center=means[0].xyz/max(means[0].w,1.f);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 first=0.f;float2 second=0.f;
    for(int i=t;i<count;i+=256)if(selected[i]) {
        float3 d=xyz_at(points,i)-center;
        first+=float4(d.x*d.x,d.x*d.y,d.x*d.z,d.y*d.y);
        second+=float2(d.y*d.z,d.z*d.z);
    }
    cov1[t]=first;cov2[t]=second;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int stride=128;stride>0;stride/=2) {
        if(t<uint(stride)) {cov1[t]+=cov1[t+stride];cov2[t]+=cov2[t+stride];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(t==0) {
        float4 c=cov1[0];float2 d=cov2[0];
        float a[3][3]={{c.x,c.y,c.z},{c.y,c.w,d.x},{c.z,d.x,d.y}};
        float v[3][3]={{1,0,0},{0,1,0},{0,0,1}};
        for(int iteration=0;iteration<24;iteration++) {
            int p=0,q=1;float maximum=abs(a[0][1]);
            if(abs(a[0][2])>maximum) {p=0;q=2;maximum=abs(a[0][2]);}
            if(abs(a[1][2])>maximum) {p=1;q=2;maximum=abs(a[1][2]);}
            if(maximum<1.e-12f)break;
            float tau=(a[q][q]-a[p][p])/(2.f*a[p][q]);
            float tangent=(tau>=0.f?1.f:-1.f)/(abs(tau)+sqrt(1.f+tau*tau));
            float cosine=1.f/sqrt(1.f+tangent*tangent),sine=tangent*cosine;
            float apq=a[p][q];a[p][p]-=tangent*apq;a[q][q]+=tangent*apq;
            a[p][q]=0.f;a[q][p]=0.f;
            for(int r=0;r<3;r++) {
                if(r!=p && r!=q) {
                    float rp=a[r][p],rq=a[r][q];
                    a[r][p]=cosine*rp-sine*rq;a[p][r]=a[r][p];
                    a[r][q]=sine*rp+cosine*rq;a[q][r]=a[r][q];
                }
                float vp=v[r][p],vq=v[r][q];
                v[r][p]=cosine*vp-sine*vq;v[r][q]=sine*vp+cosine*vq;
            }
        }
        int smallest=0;
        if(a[1][1]<a[smallest][smallest])smallest=1;
        if(a[2][2]<a[smallest][smallest])smallest=2;
        float3 normal=normalize(float3(v[0][smallest],v[1][smallest],v[2][smallest]));
        fitted=float4(normal,-dot(normal,center));
        out[0]=fitted.x;out[1]=fitted.y;out[2]=fitted.z;out[3]=fitted.w;out_count[0]=active;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(update)for(int i=t;i<count;i+=256)
        selected[i]=active>=minimum && abs(dot(xyz_at(points,i),fitted.xyz)+fitted.w)<tolerance;
}
"""


@lru_cache(maxsize=1)
def library():
    compiler = getattr(torch.mps, "compile_shader", None)
    if compiler is None:
        raise RuntimeError(
            "Metal backend requires torch.mps.compile_shader; use a compatible PyTorch build or device='cpu'"
        )
    return compiler(SOURCE)


def unproject(depth, k):
    out = torch.empty((*depth.shape, 3), device=depth.device, dtype=torch.float32)
    library().backproject(
        depth.contiguous(), k.contiguous(), out, threads=depth.numel()
    )
    return out


def plane_masks(xyz, support, planes, tolerance):
    shape = (len(planes), 512, 512)
    raw = torch.empty(shape, dtype=torch.bool, device=xyz.device)
    temp = torch.empty_like(raw)
    masks = torch.empty_like(raw)
    cores = torch.empty_like(raw)
    lib = library()
    lib.label_planes(
        xyz.contiguous(),
        support.contiguous(),
        planes.contiguous(),
        raw,
        len(planes),
        float(tolerance),
        threads=262144,
    )
    lib.morphology(raw, temp, 0, threads=raw.numel())
    lib.morphology(temp, masks, 1, threads=raw.numel())
    lib.morphology(masks, cores, 1, threads=raw.numel())
    return masks, cores


def fit_lines(parts, device):
    sizes = np.array([len(p) for p in parts], np.int32)
    offsets = np.r_[np.int32(0), sizes.cumsum(dtype=np.int32)]
    packed = np.concatenate(parts).astype(np.float32)
    points = torch.as_tensor(packed, device=device)
    indices = torch.as_tensor(offsets, device=device)
    out = torch.empty((len(parts), 4), device=device)
    library().robust_lines(points, indices, out, threads=len(parts))
    return out.cpu().numpy()


def visible(points, uv, support, xyz):
    out = torch.empty(len(points), dtype=torch.bool, device=points.device)
    if len(points):
        library().visibility(
            points.contiguous(),
            uv.contiguous(),
            support.contiguous(),
            xyz.contiguous(),
            out,
            threads=len(points),
        )
    return out


def unoccluded(points, uv, depth, valid):
    out = torch.empty(len(points), dtype=torch.bool, device=points.device)
    if len(points):
        library().footprint(
            points.contiguous(),
            uv.contiguous(),
            depth.contiguous(),
            valid.contiguous(),
            out,
            threads=len(points),
        )
    return out


def snap(points, primary, planes, eligible):
    out = torch.empty_like(points)
    counts = torch.empty(len(points), dtype=torch.int64, device=points.device)
    if len(points):
        library().snap_points(
            points.contiguous(),
            primary.contiguous(),
            planes.contiguous(),
            eligible.contiguous(),
            len(planes),
            out,
            counts,
            threads=len(points),
        )
    return out, counts


def physical_edges(
    uv, adjacent, planes, plane_ids, masks, object_mask, depth, valid, xyz, k, neutral
):
    out = torch.empty(len(uv), dtype=torch.bool, device=uv.device)
    counts = torch.empty((len(uv), 2, 2), dtype=torch.int64, device=uv.device)
    if len(uv):
        library().physical_edges(
            uv.contiguous(),
            adjacent.contiguous(),
            planes.contiguous(),
            plane_ids.contiguous(),
            masks.contiguous(),
            object_mask.contiguous(),
            depth.contiguous(),
            valid.contiguous(),
            xyz.contiguous(),
            k.contiguous(),
            neutral.contiguous(),
            out,
            counts,
            threads=len(uv),
        )
    return out, counts


def fit_planes(points, parameters, seed, maximum):
    current = points.contiguous()
    result = []
    rng = np.random.default_rng(seed)
    lib = library()
    tolerance = float(parameters.plane_tolerance_m)
    for _ in range(maximum):
        if len(current) < parameters.minimum_plane_pixels:
            break
        indices = rng.choice(
            len(current), min(len(current), 1300), replace=False
        ).astype(np.int32)
        triplets = rng.choice(
            len(indices), (parameters.ransac_trials, 3), replace=True
        ).astype(np.int32)
        sample = torch.as_tensor(indices, device=points.device)
        triples = torch.as_tensor(triplets, device=points.device)
        hypotheses = points.new_empty((parameters.ransac_trials, 4))
        votes = torch.empty(
            parameters.ransac_trials, dtype=torch.int32, device=points.device
        )
        lib.score_planes(
            current,
            sample,
            triples,
            hypotheses,
            votes,
            len(indices),
            tolerance,
            threads=parameters.ransac_trials,
        )
        winner = votes.argmax()
        selected = torch.empty(len(current), dtype=torch.bool, device=points.device)
        lib.select_plane(
            current, hypotheses, winner, selected, tolerance, threads=len(current)
        )
        plane = points.new_empty(4)
        count = torch.empty(1, dtype=torch.int32, device=points.device)
        for _ in range(3):
            lib.weighted_plane(
                current,
                selected,
                plane,
                count,
                len(current),
                tolerance,
                parameters.minimum_plane_pixels,
                1,
                threads=256,
                group_size=256,
            )
        lib.weighted_plane(
            current,
            selected,
            plane,
            count,
            len(current),
            tolerance,
            parameters.minimum_plane_pixels,
            0,
            threads=256,
            group_size=256,
        )
        if int(count.cpu()[0]) < parameters.minimum_plane_pixels:
            break
        result.append(plane)
        current = current[~selected]
    return torch.stack(result) if result else points.new_empty((0, 4))


def boundary_batch(
    uv,
    adjacent,
    primary,
    methods,
    data,
    object_mask,
    depth,
    valid,
    xyz,
    k,
    neutral,
    parameters,
):
    points = uv.new_empty((len(uv), 3))
    image = torch.empty_like(uv)
    count = torch.empty(len(uv), dtype=torch.int64, device=uv.device)
    core = torch.empty(len(uv), dtype=torch.bool, device=uv.device)
    recovered = torch.empty_like(core)
    library().boundary_batch(
        uv,
        adjacent,
        primary,
        methods,
        data.planes,
        data.eligible,
        len(data.planes),
        data.masks,
        object_mask,
        data.support,
        depth,
        valid,
        xyz,
        k,
        neutral,
        float(parameters.maximum_angle_cosine),
        float(parameters.maximum_support_gap_px),
        int(data.recovery),
        points,
        image,
        count,
        core,
        recovered,
        threads=len(uv),
    )
    return points, image, count, core, recovered


def envelope_batch(
    points, uv, adjacent, primary, data, object_mask, depth, valid, xyz, k, neutral
):
    out = torch.empty_like(points)
    image = torch.empty_like(uv)
    count = torch.empty(len(uv), dtype=torch.int64, device=uv.device)
    selected = torch.empty(len(uv), dtype=torch.bool, device=uv.device)
    library().envelope_batch(
        points,
        uv,
        adjacent,
        primary,
        data.planes,
        data.eligible,
        len(data.planes),
        data.masks,
        object_mask,
        data.support,
        depth,
        valid,
        xyz,
        k,
        neutral,
        out,
        image,
        count,
        selected,
        threads=len(uv),
    )
    return out, image, count, selected


def junction_batch(points, indices, data, xyz, depth, valid, k):
    uv = points.new_empty((len(points), 2))
    selected = torch.empty(len(points), dtype=torch.bool, device=points.device)
    recovered = torch.empty_like(selected)
    library().junction_batch(
        points,
        indices,
        data.masks,
        data.cores,
        data.support,
        xyz,
        depth,
        valid,
        k,
        uv,
        selected,
        recovered,
        threads=len(points),
    )
    return uv, selected, recovered


def box_rgb_batch(uv, primary, axes, data, depth, valid, xyz, k, neutral):
    points = uv.new_empty((len(uv), 3))
    selected = torch.empty(len(uv), dtype=torch.bool, device=uv.device)
    holes = torch.empty_like(selected)
    library().box_rgb_batch(
        uv,
        primary,
        axes,
        data.planes,
        data.masks,
        data.support,
        depth,
        valid,
        xyz,
        k,
        neutral,
        points,
        selected,
        holes,
        threads=len(uv),
    )
    return points, selected, holes


def box_cube_batch(
    points, rotation, signs, features, data, depth, valid, xyz, k, neutral
):
    out = torch.empty_like(points)
    image = points.new_empty((len(points), 2))
    selected = torch.empty(len(points), dtype=torch.bool, device=points.device)
    library().box_cube_batch(
        points,
        rotation,
        signs,
        features,
        len(features),
        data.planes,
        data.eligible,
        len(data.planes),
        data.masks,
        data.support,
        depth,
        valid,
        xyz,
        k,
        neutral,
        out,
        image,
        selected,
        threads=len(points),
    )
    return out, image, selected
