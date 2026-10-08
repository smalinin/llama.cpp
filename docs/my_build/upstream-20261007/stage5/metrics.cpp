#include <cmath>
#include <cstddef>
#include <algorithm>

extern "C" void stage5_metrics(const float * a,const float * b,size_t n,int masked,double * out) {
    double err=0,norm=0,maxabs=0,mask_changes=0,nonfinite=0;
    for (size_t i=0;i<n;++i) {
        const double p=a[i],q=b[i];
        if (!std::isfinite(p) || !std::isfinite(q)) {++nonfinite;continue;}
        if (masked && (p<-1e8 || q<-1e8)) {mask_changes+=p!=q;continue;}
        const double d=p-q;err+=d*d;norm+=p*p;maxabs=std::max(maxabs,std::abs(d));
    }
    out[0]=err;out[1]=norm;out[2]=maxabs;out[3]=mask_changes;out[4]=nonfinite;
}
