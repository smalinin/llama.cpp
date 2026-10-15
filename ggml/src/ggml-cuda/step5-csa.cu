#include "step5-csa.cuh"

extern "C" __device__ float __nv_sinf(float);
extern "C" __device__ float __nv_cosf(float);
extern "C" __device__ float __nv_powf(float, float);

static __device__ float csa_bf16(float x) {
    unsigned bits = __float_as_uint(x);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return __uint_as_float(bits & 0xffff0000U);
}

static __device__ float csa_e4m3(float x) {
    const float a = fabsf(x);
    if (a == 0 || !isfinite(a)) { return x; }
    if (a > 464) { return nanf(""); }
    int exponent;
    frexpf(a, &exponent);
    const float step = ldexpf(1.0f, max(exponent-4, -9));
    return copysignf(fminf(448.0f, rintf(a/step)*step), x);
}

static __global__ void csa_index_norm(const float * x, const float * weight, const float * bias,
        const int * pos, float * out, int heads, float eps, float theta, bool subtract) {
    const int row = blockIdx.x, d = threadIdx.x;
    __shared__ float sum[256], squares[256], normalized[256];
    const float raw = csa_bf16(x[(size_t) row*256+d]);
    sum[d] = raw; squares[d] = raw*raw;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (d < stride) { sum[d] += sum[d+stride]; squares[d] += squares[d+stride]; }
        __syncthreads();
    }
    const float mean = subtract ? sum[0]/256 : 0;
    const float rs = rsqrtf(squares[0]/256-mean*mean+eps);
    const float scale = csa_bf16(weight[d])+(subtract ? 0 : 1);
    const float offset = bias ? csa_bf16(bias[d]) : 0;
    normalized[d] = csa_bf16((raw-mean)*rs*scale+offset);
    __syncthreads();
    float value = normalized[d];
    if (d < 32) {
        const int j = d%16;
        // Fast trig loses precision at long positions under -use_fast_math.
        const float angle = pos[row/heads]*__fdiv_rn(1.0f, __nv_powf(theta,float(j)/16));
        const float c = csa_bf16(__nv_cosf(angle)), s = csa_bf16(__nv_sinf(angle));
        value = d < 16 ? normalized[j]*c-normalized[j+16]*s : normalized[j]*s+normalized[j+16]*c;
    }
    out[(size_t) row*256+d] = csa_bf16(value);
}

struct csa_cache {
    const char * data;
    size_t nb0, nb1, nb2, nb3;
    int cells, type;

    __device__ float load(int slot, int head, int d) const {
        const char * p = data + (slot/cells)*nb3 + (slot%cells)*nb2 + head*nb1 + d*nb0;
        if (type == GGML_TYPE_F32) { return *(const float *) p; }
        if (type == GGML_TYPE_F16) { return __half2float(*(const half *) p); }
        return __uint_as_float((unsigned) *(const uint16_t *) p << 16);
    }
};

static csa_cache csa_view(const ggml_tensor * t) {
    return {(const char *) t->data, t->nb[0], t->nb[1], t->nb[2], t->nb[3], (int) t->ne[2], (int) t->type};
}

static __device__ float csa_warp_sum(float x) {
    for (int offset = 16; offset; offset /= 2) { x += __shfl_xor_sync(0xffffffff, x, offset); }
    return x;
}

static __global__ void csa_summary(csa_cache kz, const int * slots, float * out, int positions, int regions) {
    const int r = blockIdx.x, seq = blockIdx.y, d = threadIdx.x;
    __shared__ float maximum[256];
    int cell[8];
    bool valid = true;
    float m = -INFINITY;
    for (int t = 0; t < 8; ++t) {
        cell[t] = slots[seq*positions+8*r+t];
        valid = valid && cell[t] >= 0;
        if (cell[t] >= 0) { m = fmaxf(m, kz.load(cell[t], 0, 256+d)); }
    }
    maximum[d] = m;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (d < stride) { maximum[d] = fmaxf(maximum[d], maximum[d+stride]); }
        __syncthreads();
    }
    float num = 0, den = 0;
    if (valid) {
        for (int t = 0; t < 8; ++t) {
            const float weight = expf(kz.load(cell[t],0,256+d)-maximum[0]);
            num += weight*kz.load(cell[t],0,d); den += weight;
        }
    }
    out[((size_t) seq*regions+r)*256+d] = valid ? csa_e4m3(csa_bf16(den > 0 ? num/fmaxf(den, 1e-20f) : 0)) : nanf("");
}

static __global__ void csa_quantize(const float * in, half * out, size_t n, bool quantize) {
    const size_t i = (size_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) { out[i] = __float2half(quantize ? csa_e4m3(in[i]) : in[i]); }
}

static __global__ void csa_scores(const float * dots, const float * w, const float * summary,
        const int * seq, const int * pos, float * out, int nr, int nq, int sequence) {
    const size_t i = (size_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= (size_t) nr*4*nq) { return; }
    const int r = i%nr, group = i/nr%4, qi = i/(nr*4);
    if (seq[qi] != sequence) { return; }
    float score = -INFINITY;
    if (r < pos[qi]/8 && isfinite(summary[((size_t) sequence*nr+r)*256])) {
        score = 0;
        for (int h = 0; h < 4; ++h) {
            const int head = qi*16+group*4+h;
            score += fmaxf(0, dots[(size_t) head*nr+r])*w[head]*0.5f;
        }
    }
    out[i] = score;
}

static __device__ unsigned csa_key(float x) {
    if (x == 0) { x = 0; }
    const unsigned bits = __float_as_uint(x);
    return bits ^ ((bits >> 31) ? 0xffffffffU : 0x80000000U);
}

static __global__ void csa_topk(const float * scores, const int * pos, int * out, int regions, int budget) {
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid%32, warp = tid/32;
    const int visible = min(pos[row/4]/8, regions), width = (budget+1)*8;
    scores += (size_t) row*regions; out += (size_t) row*width;
    for (int i = tid; i < width; i += 256) { out[i] = -1; }
    __shared__ int counts[8], equal_counts[8], rank, packed, equal_before;
    __shared__ unsigned prefix, mask;
    if (tid == 0) { rank = budget; prefix = mask = 0; packed = equal_before = 0; }
    __syncthreads();
    if (visible > budget) {
        for (int bit = 31; bit >= 0; --bit) {
            int count = 0;
            for (int r = tid; r < visible; r += 256) {
                const unsigned key = csa_key(scores[r]);
                count += (key & mask) == prefix && ((key >> bit) & 1U);
            }
            count = (int) csa_warp_sum((float) count);
            if (lane == 0) { counts[warp] = count; }
            __syncthreads();
            if (tid == 0) {
                int total = 0;
                for (int w = 0; w < 8; ++w) { total += counts[w]; }
                if (total >= rank) { prefix |= 1U << bit; } else { rank -= total; }
                mask |= 1U << bit;
            }
            __syncthreads();
        }
    }
    for (int start = 0; start < visible; start += 256) {
        const int r = start+tid;
        const unsigned key = r < visible ? csa_key(scores[r]) : 0;
        const bool eq = r < visible && key == prefix;
        const unsigned equal_bits = __ballot_sync(0xffffffff, eq);
        if (lane == 0) { equal_counts[warp] = __popc(equal_bits); }
        __syncthreads();
        int eq_rank = equal_before + __popc(equal_bits & ((1U << lane)-1));
        for (int w = 0; w < warp; ++w) { eq_rank += equal_counts[w]; }
        const bool chosen = r < visible && (visible <= budget || key > prefix || (eq && eq_rank < rank));
        const unsigned selected = __ballot_sync(0xffffffff, chosen);
        if (lane == 0) { counts[warp] = __popc(selected); }
        __syncthreads();
        int index = packed + __popc(selected & ((1U << lane)-1));
        for (int w = 0; w < warp; ++w) { index += counts[w]; }
        if (chosen) { for (int t = 0; t < 8; ++t) { out[index*8+t] = r*8+t; } }
        __syncthreads();
        if (tid == 0) {
            for (int w = 0; w < 8; ++w) { packed += counts[w]; equal_before += equal_counts[w]; }
        }
        __syncthreads();
    }
    if (tid < 8 && pos[row/4]/8*8+tid <= pos[row/4]) { out[packed*8+tid] = pos[row/4]/8*8+tid; }
}

static __global__ void csa_attention(const float * q, csa_cache k, csa_cache v, const int * ids,
        const int * slots, const int * seq, float * out, int dim, int heads, int width, int positions, float scale) {
    const int qi = blockIdx.x, head = blockIdx.y, tid = threadIdx.x, lane = tid%32, warp = tid/32;
    const int group = head/(heads/4);
    q += ((size_t) qi*heads+head)*dim;
    ids += ((size_t) qi*4+group)*width;
    slots += (size_t) seq[qi]*positions;
    float qr[8], acc[8] = {}, maximum = -INFINITY, total = 0;
    for (int j = 0; j < 8; ++j) { qr[j] = lane+32*j < dim ? q[lane+32*j] : 0; }
    for (int i = warp; i < width; i += 8) {
        const int p = ids[i];
        if (p < 0 || p >= positions) { continue; }
        const int slot = slots[p];
        if (slot < 0) { continue; }
        float dot = 0;
        for (int j = 0; j < 8; ++j) { if (lane+32*j < dim) { dot += qr[j]*k.load(slot,group,lane+32*j); } }
        dot = csa_warp_sum(dot)*scale;
        const float m = fmaxf(maximum, dot), old = expf(maximum-m), weight = expf(dot-m);
        maximum = m; total = total*old+weight;
        for (int j = 0; j < 8; ++j) {
            if (lane+32*j < dim) { acc[j] = acc[j]*old+weight*v.load(slot,group,lane+32*j); }
        }
    }
    __shared__ float maxima[8], totals[8], partial[8][256], coefficients[8], denominator;
    if (lane == 0) { maxima[warp] = maximum; totals[warp] = total; }
    for (int j = 0; j < 8; ++j) { if (lane+32*j < dim) { partial[warp][lane+32*j] = acc[j]; } }
    __syncthreads();
    if (tid == 0) {
        float m = -INFINITY;
        for (int w = 0; w < 8; ++w) { m = fmaxf(m, maxima[w]); }
        float sum = 0;
        for (int w = 0; w < 8; ++w) {
            coefficients[w] = totals[w] > 0 ? expf(maxima[w]-m) : 0;
            sum += totals[w]*coefficients[w];
        }
        denominator = sum;
    }
    __syncthreads();
    if (tid < dim) {
        float value = 0;
        for (int w = 0; w < 8; ++w) { value += partial[w][tid]*coefficients[w]; }
        out[((size_t) qi*heads+head)*dim+tid] = denominator > 0 ? value/denominator : 0;
    }
}

void ggml_cuda_op_step5_csa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto stream = ctx.stream();
    const auto * a = dst->src[0];
    const auto * b = dst->src[1];
    if (dst->op == GGML_OP_STEP5_INDEX_NORM) {
        float op[3]; memcpy(op, dst->op_params, sizeof(op));
        csa_index_norm<<<a->ne[1]*a->ne[2],256,0,stream>>>((const float *) a->data,(const float *) b->data,
                dst->src[2] ? (const float *) dst->src[2]->data : nullptr, (const int *) dst->src[3]->data,
                (float *) dst->data,a->ne[1],op[0],op[1],op[2] != 0);
    } else if (dst->op == GGML_OP_STEP5_CSA_SUMMARY) {
        csa_summary<<<dim3(dst->ne[1],dst->ne[2]),256,0,stream>>>(csa_view(a), (const int *) b->data,
                (float *) dst->data, b->ne[0], dst->ne[1]);
    } else if (dst->op == GGML_OP_STEP5_CSA_SCORES) {
        const auto * summary = dst->src[2];
        const int nr = summary->ne[1], nq = a->ne[2], columns = nq*16;
        const size_t ns = ggml_nelements(summary), nquery = ggml_nelements(a);
        auto * dots = (float *) dst->data + (size_t) nr*4*nq;
        auto * qs = (half *) (dots + (size_t) nr*columns);
        auto * sums = qs + nquery;
        GGML_ASSERT((char *) (sums + ns) <= (char *) dst->data + ggml_nbytes(dst));
        csa_quantize<<<(nquery+255)/256,256,0,stream>>>((const float *) a->data, qs, nquery, true);
        csa_quantize<<<(ns+255)/256,256,0,stream>>>((const float *) summary->data, sums, ns, false);
        const float alpha = 1, beta = 0;
        auto handle = ctx.cublas_handle();
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        for (int seq = 0; seq < summary->ne[2]; ++seq) {
            CUBLAS_CHECK(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,nr,columns,256,&alpha,
                    sums+(size_t) seq*nr*256,CUDA_R_16F,256,qs,CUDA_R_16F,256,
                    &beta,dots,CUDA_R_32F,nr,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            csa_scores<<<((size_t) nr*4*nq+255)/256,256,0,stream>>>(dots, (const float *) b->data,
                    (const float *) summary->data, (const int *) dst->src[3]->data, (const int *) dst->src[4]->data,
                    (float *) dst->data, nr, nq, seq);
        }
    } else if (dst->op == GGML_OP_STEP5_CSA_TOPK) {
        csa_topk<<<dst->ne[1]*dst->ne[2],256,0,stream>>>((const float *) a->data, (const int *) b->data,
                (int *) dst->data,a->ne[0],ggml_get_op_params_i32(dst,0));
    } else {
        float scale; memcpy(&scale, dst->op_params, sizeof(scale));
        csa_attention<<<dim3(a->ne[2],a->ne[1]),256,0,stream>>>((const float *) a->data,csa_view(b),csa_view(dst->src[2]),
                (const int *) dst->src[3]->data,(const int *) dst->src[4]->data,(const int *) dst->src[5]->data,
                (float *) dst->data,a->ne[0],a->ne[1],dst->src[3]->ne[0],dst->src[4]->ne[0],scale);
    }
}
