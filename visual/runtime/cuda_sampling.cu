// Experimental FP32 PCA5 and source-index FPS. No GT, point pre-reduction,
// model inference, or change to the configured depth/orientation thresholds.
#include <math.h>
#include <stdint.h>

__device__ bool link(float a, float b, float ab, float rel) {
    return fabsf(a-b) <= ab+rel*fminf(a,b);
}

extern "C" __global__ void pca5(const float* xyz, const uint8_t* support,
                                float* normals, uint8_t* valid,
                                int h, int w, float ab, float rel) {
    int k=blockIdx.x*blockDim.x+threadIdx.x, np=h*w;
    if(k>=2*np) return;
    valid[k]=0; normals[3*k]=NAN; normals[3*k+1]=NAN; normals[3*k+2]=NAN;
    if(!support[k])return;
    int oid=k/np, c=k%np, cv=c/w, cu=c%w;
    uint32_t available=0, right=0, down=0;
    for(int j=0;j<25;j++) {
        int v=cv+j/5-2,u=cu+j%5-2;
        if(v<0||u<0||v>=h||u>=w)continue;
        int s=v*w+u;
        if(!support[oid*np+s])continue;
        available|=1u<<j;
        if(j%5<4&&u+1<w&&support[oid*np+s+1]&&link(xyz[3*s+2],xyz[3*(s+1)+2],ab,rel))right|=1u<<j;
        if(j/5<4&&v+1<h&&support[oid*np+s+w]&&link(xyz[3*s+2],xyz[3*(s+w)+2],ab,rel))down|=1u<<j;
    }
    uint32_t reached=1u<<12;
    for(int i=0;i<24;i++) {
        uint32_t next=reached|((reached&right)<<1)|((reached>>1)&right)|((reached&down)<<5)|((reached>>5)&down);
        if(next==reached)break;
        reached=next;
    }
    int n=__popc(reached);
    if(n<6)return;
    float sx=0,sy=0,sz=0,xx=0,xy=0,xz=0,yy=0,yz=0,zz=0;
    for(int j=0;j<25;j++)if(reached&(1u<<j)) {
        int s=(cv+j/5-2)*w+cu+j%5-2;
        float x=xyz[3*s]-xyz[3*c], y=xyz[3*s+1]-xyz[3*c+1], z=xyz[3*s+2]-xyz[3*c+2];
        sx+=x;sy+=y;sz+=z;xx+=x*x;xy+=x*y;xz+=x*z;yy+=y*y;yz+=y*z;zz+=z*z;
    }
    float inv=1.f/n;sx*=inv;sy*=inv;sz*=inv;
    float a[9]={xx*inv-sx*sx,xy*inv-sx*sy,xz*inv-sx*sz,
                xy*inv-sx*sy,yy*inv-sy*sy,yz*inv-sy*sz,
                xz*inv-sx*sz,yz*inv-sy*sz,zz*inv-sz*sz};
    float scale=0;for(int i=0;i<9;i++)scale=fmaxf(scale,fabsf(a[i]));
    if(!(scale>0))return;
    for(int i=0;i<9;i++)a[i]/=scale;
    float q[9]={1,0,0,0,1,0,0,0,1};
    // Small symmetric Jacobi solver avoids large cuSOLVER/Torch dependencies.
    for(int sweep=0;sweep<8;sweep++)for(int pair=0;pair<3;pair++) {
        int p=pair==2?1:0,r=pair==0?1:2;
        float off=a[3*p+r];if(fabsf(off)<1e-12f)continue;
        float tau=(a[3*r+r]-a[3*p+p])/(2*off);
        float t=copysignf(1.f,tau)/(fabsf(tau)+hypotf(1.f,tau));
        float co=rsqrtf(1+t*t),si=t*co;
        a[3*p+p]-=t*off;a[3*r+r]+=t*off;a[3*p+r]=a[3*r+p]=0;
        for(int j=0;j<3;j++)if(j!=p&&j!=r) {
            float x=a[3*j+p],y=a[3*j+r];
            a[3*j+p]=a[3*p+j]=co*x-si*y;a[3*j+r]=a[3*r+j]=si*x+co*y;
        }
        for(int j=0;j<3;j++) {
            float x=q[3*j+p],y=q[3*j+r];q[3*j+p]=co*x-si*y;q[3*j+r]=si*x+co*y;
        }
    }
    int lo=0,mi=1,hi=2,tmp;
    if(a[3*lo+lo]>a[3*mi+mi]){tmp=lo;lo=mi;mi=tmp;}
    if(a[3*mi+mi]>a[3*hi+hi]){tmp=mi;mi=hi;hi=tmp;}
    if(a[3*lo+lo]>a[3*mi+mi]){tmp=lo;lo=mi;mi=tmp;}
    float threshold=fmaxf(1e-16f/scale,a[3*hi+hi]*1e-8f);
    if(!(a[3*mi+mi]>threshold&&a[3*mi+mi]-a[3*lo+lo]>threshold))return;
    float x=q[lo],y=q[3+lo],z=q[6+lo],length=sqrtf(x*x+y*y+z*z);
    if(!(length>0)||!isfinite(length))return;
    float sign=(x*xyz[3*c]+y*xyz[3*c+1]+z*xyz[3*c+2])>0?-1.f:1.f;
    normals[3*k]=sign*x/length;normals[3*k+1]=sign*y/length;normals[3*k+2]=sign*z/length;valid[k]=1;
}

extern "C" __global__ void fps(const float* xyz,float* distance,int* output,
                               const int* initial,int n,int m,int ni,int first) {
    __shared__ float best[256];__shared__ int index[256];__shared__ int cur;
    int tid=threadIdx.x;
    for(int j=tid;j<n;j+=256)distance[j]=INFINITY;
    if(tid==0)cur=ni?initial[0]:first;
    __syncthreads();
    for(int i=0;i<m;i++) {
        if(tid==0) { if(i<ni)cur=initial[i]; output[i]=cur; }
        __syncthreads();
        float x=xyz[3*cur],y=xyz[3*cur+1],z=xyz[3*cur+2];
        float bv=-INFINITY;int bi=2147483647;
        for(int j=tid;j<n;j+=256) {
            float dx=xyz[3*j]-x,dy=xyz[3*j+1]-y,dz=xyz[3*j+2]-z;
            float v=fminf(distance[j],dx*dx+dy*dy+dz*dz);
            if(j==cur)v=-INFINITY;distance[j]=v;
            if(v>bv||(v==bv&&j<bi)){bv=v;bi=j;}
        }
        best[tid]=bv;index[tid]=bi;__syncthreads();
        for(int s=128;s;s>>=1) {
            if(tid<s) {
                float v=best[tid+s];int j=index[tid+s];
                if(v>best[tid]||(v==best[tid]&&j<index[tid])) {best[tid]=v;index[tid]=j;}
            }
            __syncthreads();
        }
        if(tid==0)cur=index[0];__syncthreads();
    }
}

extern "C" __global__ void features(const float* xyz,const uint8_t* raw,const uint8_t* support,
    const float* normal,const uint8_t* normal_valid,float* depth_jump,float* variation,
    uint8_t* flags,uint8_t* neighbor_count,float* angle_h,float* angle_v,
    int h,int w,float ab,float rel) {
    int k=blockIdx.x*blockDim.x+threadIdx.x,np=h*w;
    if(k>=2*np)return;
    int oid=k/np,p=k%np,v=p/w,u=p%w;
    bool mask=raw[k],observed=support[k],nv=normal_valid[k];
    float z=xyz[3*p+2],maxdepth=0,maxangle=0;
    bool pair=false,jump=false,maskedge=false,invalidedge=false;
    int count=0;angle_h[k]=NAN;angle_v[k]=NAN;
    for(int direction=0;direction<4;direction++) {
        int yy=v+(direction==2?-1:direction==3?1:0);
        int xx=u+(direction==0?-1:direction==1?1:0);
        if(yy<0||xx<0||yy>=h||xx>=w)continue;
        int q=yy*w+xx,oq=oid*np+q;
        float zq=xyz[3*q+2];bool qvalid=isfinite(zq)&&zq>0;
        if(observed&&qvalid) {
            float dz=fabsf(z-zq);pair=true;maxdepth=fmaxf(maxdepth,dz);
            if(!link(z,zq,ab,rel))jump=true;
        }
        maskedge |= mask&&!raw[oq]; invalidedge |= observed&&!qvalid;
        if(nv&&normal_valid[oq]) {
            float ax=normal[3*k],ay=normal[3*k+1],az=normal[3*k+2];
            float bx=normal[3*oq],by=normal[3*oq+1],bz=normal[3*oq+2];
            float cx=ay*bz-az*by,cy=az*bx-ax*bz,cz=ax*by-ay*bx;
            float angle=atan2f(sqrtf(cx*cx+cy*cy+cz*cz),ax*bx+ay*by+az*bz);
            maxangle=fmaxf(maxangle,angle);count++;
            if(direction==1)angle_h[k]=angle*(180.f/3.14159265358979323846f);
            if(direction==3)angle_v[k]=angle*(180.f/3.14159265358979323846f);
        }
    }
    bool varvalid=nv&&count>0;
    depth_jump[k]=pair?maxdepth:NAN;variation[k]=varvalid?maxangle:NAN;
    neighbor_count[k]=(uint8_t)count;
    flags[k]=(pair?1:0)|(jump?2:0)|(maskedge?4:0)|(invalidedge?8:0)
             |((mask&&(v==0||u==0||v==h-1||u==w-1))?16:0)|(varvalid?32:0);
}
