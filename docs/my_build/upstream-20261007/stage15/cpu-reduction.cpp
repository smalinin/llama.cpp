#include <cmath>
#include <cstddef>
extern "C" void stage15_reduce(const float * down, const float * weights, float * rounded, float * fused, size_t rows, int experts) {
    for (size_t row=0;row<rows;++row) {
        volatile float first=down[row]*weights[0];
        float a=first,b=first;
        for (int expert=1;expert<experts;++expert) {
            volatile float product=down[size_t(expert)*rows+row]*weights[expert];
            a=a+product;
            b=std::fma(down[size_t(expert)*rows+row],weights[expert],b);
        }
        rounded[row]=a;fused[row]=b;
    }
}
