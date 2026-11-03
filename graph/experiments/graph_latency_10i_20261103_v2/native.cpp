// Task-local, single-threaded exact-order loops. No fast-math or FP contraction.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

struct Cell {
    int64_t x,y,z,object;
    bool operator==(const Cell& b) const {return x==b.x && y==b.y && z==b.z && object==b.object;}
};
struct Hash {
    size_t operator()(const Cell& c) const {
        size_t h=0;
        for(auto v:{c.x,c.y,c.z,c.object}) h^=std::hash<int64_t>{}(v)+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
        return h;
    }
};

extern "C" int64_t dedup(const double* xyz,const int64_t* objects,int64_t n,double radius,int64_t* labels,
                         const int64_t* overrides,int64_t known,int64_t* unknown,int64_t capacity) {
    std::unordered_map<Cell,std::vector<int64_t>,Hash> buckets;
    std::unordered_map<int64_t,bool> boundary;
    for(int64_t k=0;k<known;++k) boundary[overrides[3*k]*n+overrides[3*k+1]]=overrides[3*k+2];
    int64_t uncertain=0;
    for(int64_t i=0;i<n;++i) {
        Cell c{int64_t(std::floor(xyz[3*i]/radius)),int64_t(std::floor(xyz[3*i+1]/radius)),
               int64_t(std::floor(xyz[3*i+2]/radius)),objects[i]};
        int64_t first=i;
        for(int x=-1;x<=1;++x) for(int y=-1;y<=1;++y) for(int z=-1;z<=1;++z) {
            auto found=buckets.find({c.x+x,c.y+y,c.z+z,c.object});
            if(found==buckets.end()) continue;
            for(auto j:found->second) {
                double dx=xyz[3*i]-xyz[3*j],dy=xyz[3*i+1]-xyz[3*j+1],dz=xyz[3*i+2]-xyz[3*j+2];
                double d=std::sqrt((dx*dx+dy*dy)+dz*dz);
                bool hit=d<radius;
                if(std::abs(d-radius)<1e-14) {
                    auto found=boundary.find(i*n+j);
                    if(found!=boundary.end()) hit=found->second;
                    else {
                        if(uncertain>=capacity) return -1;
                        unknown[2*uncertain]=i;unknown[2*uncertain+1]=j;++uncertain;
                    }
                }
                if(hit) first=std::min(first,j);
            }
        }
        labels[i]=first;
        if(first==i) buckets[c].push_back(i);
    }
    return uncertain;
}

extern "C" void fps(const double* xyz,const int64_t* order,int64_t n,int64_t count,
                     const int64_t* initial,int64_t used,int64_t* chosen) {
    std::vector<double> distance(n,INFINITY);
    std::vector<bool> selected(n,false);
    for(int64_t k=0;k<used;++k) {chosen[k]=initial[k];selected[initial[k]]=true;}
    auto update=[&](int64_t j) {
        for(int64_t i=0;i<n;++i) {
            double dx=xyz[3*i]-xyz[3*j],dy=xyz[3*i+1]-xyz[3*j+1],dz=xyz[3*i+2]-xyz[3*j+2];
            distance[i]=std::min(distance[i],(dx*dx+dy*dy)+dz*dz);
        }
    };
    for(int64_t k=0;k<used;++k) update(chosen[k]);
    if(!used && count) {chosen[0]=order[0];selected[order[0]]=true;update(order[0]);used=1;}
    for(int64_t k=used;k<count;++k) {
        double best=-INFINITY;int64_t j=-1;
        for(int64_t r=0;r<n;++r) {auto i=order[r];if(!selected[i] && distance[i]>best) {best=distance[i];j=i;}}
        chosen[k]=j;selected[j]=true;update(j);
    }
}

extern "C" void representatives(const double* xyz,const int64_t* order,const int64_t* starts,
                                 int64_t cells,int64_t* reps) {
    for(int64_t c=0;c<cells;++c) {
        double mean[3]={0,0,0};int64_t lo=starts[c],hi=starts[c+1];
        for(auto r=lo;r<hi;++r) for(int k=0;k<3;++k) mean[k]+=xyz[3*order[r]+k];
        for(int k=0;k<3;++k) mean[k]/=double(hi-lo);
        double best=INFINITY;int64_t chosen=order[lo];
        for(auto r=lo;r<hi;++r) {
            auto i=order[r];double dx=xyz[3*i]-mean[0],dy=xyz[3*i+1]-mean[1],dz=xyz[3*i+2]-mean[2];
            double d=(dx*dx+dy*dy)+dz*dz;
            if(d<best) {best=d;chosen=i;}
        }
        reps[c]=chosen;
    }
}
