#include "step5-csa.cuh"
#include "mma.cuh"

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
        const char * p = data + (slot/cells)*nb3 + (slot%cells)*nb2 + head*nb1;
        if (type == GGML_TYPE_Q8_0) {
            const auto & block = *(const block_q8_0 *) (p + (d/QK8_0)*nb0);
            return __half2float(block.d)*block.qs[d % QK8_0];
        }
        p += d*nb0;
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

static __device__ float csa_float(float x) { return x; }
static __device__ float csa_float(half x) { return __half2float(x); }
static __device__ void csa_store(float * p, size_t i, float x) { p[i] = x; }
static __device__ void csa_store(half * p, size_t i, float x) { p[i] = __float2half(x); }

template<typename T>
static __global__ void csa_summary(csa_cache kz, const int * slots, T * out, int positions, int regions,
        T * cache, int64_t * tags, const int64_t * versions, int cache_regions) {
    const int r = blockIdx.x, seq = blockIdx.y, d = threadIdx.x;
    const int64_t version = versions ? versions[(size_t) seq*(regions+1)+r+1] : 0;
    const size_t cached_row = versions ? versions[(size_t) seq*(regions+1)]*cache_regions+r : 0;
    if (cache && version != 0 && tags[cached_row] == version) {
        if (out + ((size_t) seq*regions+r)*256 != cache + cached_row*256) {
            out[((size_t) seq*regions+r)*256+d] = cache[cached_row*256+d];
        }
        return;
    }
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
    const float value = valid ? csa_e4m3(csa_bf16(den > 0 ? num/fmaxf(den, 1e-20f) : 0)) : nanf("");
    csa_store(out, ((size_t) seq*regions+r)*256+d, value);
    if (cache) {
        if (out + ((size_t) seq*regions+r)*256 != cache + cached_row*256) {
            csa_store(cache, cached_row*256+d, value);
        }
        __syncthreads();
        if (d == 0) { tags[cached_row] = version; }
    }
}

static __global__ void csa_quantize(const float * in, half * out, size_t n, bool quantize) {
    const size_t i = (size_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) { out[i] = __float2half(quantize ? csa_e4m3(in[i]) : in[i]); }
}

template<typename T>
static __global__ void csa_scores(const float * dots, const float * w, const T * summary,
        const int * seq, const int * pos, float * out, int nr, int nq, int sequence) {
    const size_t i = (size_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= (size_t) nr*4*nq) { return; }
    const int r = i%nr, group = i/nr%4, qi = i/(nr*4);
    if (seq[qi] != sequence) { return; }
    float score = -INFINITY;
    if (r < pos[qi]/8 && isfinite(csa_float(summary[((size_t) sequence*nr+r)*256]))) {
        score = 0;
        for (int h = 0; h < 4; ++h) {
            const int head = qi*16+group*4+h;
            score += fmaxf(0, dots[(size_t) head*nr+r])*w[head]*0.5f;
        }
    }
    out[i] = score;
}



template<typename T>
static __global__ void csa_scores_fused(const half * q, const float * w, const T * summary,
        const int * seq, const int * pos, float * out, int nr, int nq) {
#if defined(AMPERE_MMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    constexpr int rows_per_warp = 2, k_step = 32, queries = 8;
    constexpr int rows = rows_per_warp*32, queries_per_warp = queries/2;
    constexpr int stride = k_step/2+4;
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp*32+lane;
    const int row0 = blockIdx.x*rows, query0 = blockIdx.y*queries, sequence = blockIdx.z;
    int visible = -1;
#pragma unroll
    for (int i = 0; i < queries; ++i) {
        if (query0+i < nq && seq[query0+i] == sequence) { visible = max(visible,pos[query0+i]/8); }
    }
    if (visible < 0) { return; }
    if (row0 >= visible) {
        for (int i = tid; i < rows*queries*4; i += 128) {
            const int r = row0+i%rows, group = i/rows%4, qi = query0+i/(rows*4);
            if (r < nr && qi < nq && seq[qi] == sequence) { out[((size_t) qi*4+group)*nr+r] = -INFINITY; }
        }
        return;
    }
    __shared__ half2 tile_s[rows][stride];
    __shared__ half2 tile_q[queries*16][stride];
    typedef tile<16, 8, half2> tile_A;
    typedef tile<8, 8, half2> tile_B;
    typedef tile<16, 8, float> tile_C;
    tile_C acc[queries_per_warp][rows_per_warp][2];
    const int local_row = (warp%2)*rows_per_warp*16;
    const int local_query = (warp/2)*queries_per_warp;
    for (int k0 = 0; k0 < 256; k0 += k_step) {
        for (int i = tid; i < rows*k_step/2; i += 128) {
            const int r = row0+i/(k_step/2), d = k0+2*(i%(k_step/2));
            half2 value = __float2half2_rn(0);
            if (r < nr) {
                if constexpr (std::is_same_v<T, half>) { value = *(const half2 *) (summary+((size_t) sequence*nr+r)*256+d); }
                else {
                    const float2 v = *(const float2 *) (summary+((size_t) sequence*nr+r)*256+d);
                    value = __floats2half2_rn(v.x,v.y);
                }
            }
            tile_s[i/(k_step/2)][i%(k_step/2)] = value;
        }
        for (int i = tid; i < queries*16*k_step/2; i += 128) {
            const int head = query0*16+i/(k_step/2), d = k0/2+i%(k_step/2);
            tile_q[i/(k_step/2)][i%(k_step/2)] = head < nq*16 ? ((const half2 *) q)[(size_t) head*128+d] : __float2half2_rn(0);
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < k_step/2; k += 8) {
            tile_A a[rows_per_warp];
#pragma unroll
            for (int r = 0; r < rows_per_warp; ++r) { load_ldmatrix(a[r], &tile_s[local_row+r*16][k], stride); }
#pragma unroll
            for (int iq = 0; iq < queries_per_warp; ++iq) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    tile_B b;
                    load_ldmatrix(b, &tile_q[(local_query+iq)*16+h*8][k], stride);
#pragma unroll
                    for (int r = 0; r < rows_per_warp; ++r) { mma(acc[iq][r][h],a[r],b); }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int iq = 0; iq < queries_per_warp; ++iq) {
        const int qi = query0+local_query+iq;
#pragma unroll
        for (int tile_r = 0; tile_r < rows_per_warp; ++tile_r) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const float d0 = acc[iq][tile_r][h].x[r*2], d1 = acc[iq][tile_r][h].x[r*2+1];
                    const float d2 = __shfl_xor_sync(0xffffffff,d0,1), d3 = __shfl_xor_sync(0xffffffff,d1,1);
                    const int region = row0+local_row+tile_r*16+lane/4+r*8;
                    if (lane%2 == 0 && region < nr && qi < nq && seq[qi] == sequence) {
                        const int group = h*2+lane%4/2;
                        float score = -INFINITY;
                        if (region < pos[qi]/8 && isfinite((float) summary[((size_t) sequence*nr+region)*256])) {
                            const float * weights = w+qi*16+group*4;
                            score = 0;
                            score += fmaxf(0,d0)*weights[0]*0.5f;
                            score += fmaxf(0,d1)*weights[1]*0.5f;
                            score += fmaxf(0,d2)*weights[2]*0.5f;
                            score += fmaxf(0,d3)*weights[3]*0.5f;
                        }
                        out[((size_t) qi*4+group)*nr+region] = score;
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(q,w,summary,seq,pos,out,nr,nq);
    NO_DEVICE_CODE;
#endif
}

template<typename T>
static __global__ void csa_scores_fused_fallback(const half * q, const float * w, const T * summary,
        const int * seq, const int * pos, float * out, int nr, int nq) {
    const size_t i = (size_t) blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= (size_t) nr*4*nq) { return; }
    const int r = i%nr, group = i/nr%4, qi = i/(nr*4);
    const T * row = summary+((size_t) seq[qi]*nr+r)*256;
    float score = -INFINITY;
    if (r < pos[qi]/8 && isfinite(csa_float(row[0]))) {
        score = 0;
        for (int h = 0; h < 4; ++h) {
            const int head = qi*16+group*4+h;
            float dot = 0;
            for (int d = 0; d < 256; ++d) { dot += __half2float(q[(size_t) head*256+d])*__half2float(__float2half(csa_float(row[d]))); }
            score += fmaxf(0,dot)*w[head]*0.5f;
        }
    }
    out[i] = score;
}

template<typename T>
static void csa_launch_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const half * q, const T * summary) {
    const int nr = dst->src[2]->ne[1], nq = dst->src[0]->ne[2];
    const auto * w = (const float *) dst->src[1]->data;
    const auto * seq = (const int *) dst->src[3]->data;
    const auto * pos = (const int *) dst->src[4]->data;
    if (ampere_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        csa_scores_fused<<<dim3((nr+63)/64,(nq+7)/8,dst->src[2]->ne[2]),dim3(32,4),0,ctx.stream()>>>(q,w,summary,seq,pos,(float *) dst->data,nr,nq);
    } else {
        csa_scores_fused_fallback<<<((size_t) nr*4*nq+255)/256,256,0,ctx.stream()>>>(q,w,summary,seq,pos,(float *) dst->data,nr,nq);
    }
}

static __device__ unsigned csa_key(float x) {
    if (x == 0) { x = 0; }
    const unsigned bits = __float_as_uint(x);
    return bits ^ ((bits >> 31) ? 0xffffffffU : 0x80000000U);
}

template<bool physical>
static __device__ int csa_selected_row(int p, const int * slots, int positions) {
    return physical ? (p < positions ? slots[p] : -1) : p;
}

template<bool physical, bool compact = false>
static __global__ void csa_topk(const float * scores, const int * pos, int * out, int regions, int budget,
        const int * slots, const int * seq, int positions, int queries, int width) {
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid%32, warp = tid/32;
    const int visible = min(pos[row/4]/8, regions);
    scores += (size_t) row*regions;
    if constexpr (!compact) {
        out += (size_t) (physical ? (row%4)*queries+row/4 : row)*width;
        if constexpr (physical) { slots += (size_t) seq[row/4]*positions; }
        for (int i = tid; i < width; i += 256) { out[i] = -1; }
    }
    __shared__ int selected_regions[compact ? 1024 : 1];
    __shared__ int counts[8], equal_counts[8], rank, packed, equal_before;
    __shared__ unsigned prefix, mask;
    if (tid == 0) { rank = budget; prefix = mask = 0; packed = equal_before = 0; }
    __syncthreads();
    if (visible > budget) {
        __shared__ int histogram[8*16], bins[16], done;
        for (int shift = 28; shift >= 0; shift -= 4) {
            if (tid < 8*16) { histogram[tid] = 0; }
            __syncthreads();
            for (int start = 0; start < visible; start += 256) {
                const int r = start+tid;
                const unsigned key = r < visible ? csa_key(scores[r]) : 0;
                const bool active = r < visible && (key & mask) == prefix;
                const unsigned digit = (key >> shift) & 15U;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                const unsigned peers = __match_any_sync(0xffffffff, active ? digit : 16U);
                if (active && lane == __ffs(peers)-1) { atomicAdd(&histogram[warp*16+digit], __popc(peers)); }
#else
                if (active) { atomicAdd(&histogram[warp*16+digit], 1); }
#endif
            }
            __syncthreads();
            if (tid < 16) {
                int total = 0;
                for (int w = 0; w < 8; ++w) { total += histogram[w*16+tid]; }
                bins[tid] = total;
            }
            __syncthreads();
            if (tid == 0) {
                int digit = 15;
                while (digit > 0 && rank > bins[digit]) { rank -= bins[digit--]; }
                prefix |= (unsigned) digit << shift;
                mask |= 15U << shift;
                // Stop when the remaining bucket is entirely selected.
                done = rank == bins[digit];
            }
            __syncthreads();
            if (done) { break; }
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
        if (chosen) {
            if constexpr (compact) { selected_regions[index] = r; }
            else { for (int t = 0; t < 8; ++t) { out[index*8+t] = csa_selected_row<physical>(r*8+t, slots, positions); } }
        }
        __syncthreads();
        if (tid == 0) {
            for (int w = 0; w < 8; ++w) { packed += counts[w]; equal_before += equal_counts[w]; }
        }
        __syncthreads();
    }
    if constexpr (compact) {
        out += (size_t) ((row%4)*queries+row/4)*width;
        slots += (size_t) seq[row/4]*positions;
        const int current = pos[row/4]/8*8, count = packed*8 + pos[row/4]%8 + 1;
        for (int i = tid; i < width; i += 256) {
            const int p = i < packed*8 ? selected_regions[i/8]*8+i%8 : current+i-packed*8;
            out[i] = i < count ? csa_selected_row<true>(p, slots, positions) : -1;
        }
    } else {
        if (tid < 8 && pos[row/4]/8*8+tid <= pos[row/4]) { out[packed*8+tid] = csa_selected_row<physical>(pos[row/4]/8*8+tid, slots, positions); }
    }
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

static __global__ void csa_rows(const int * selected, const int * slots, const int * seq, int * out,
        int width, int padded, int queries, int positions) {
    const size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (size_t) padded*queries*4) { return; }
    const int col = i%padded, qi = i/padded%queries, group = i/((size_t) padded*queries);
    const int p = col < width ? selected[((size_t) qi*4+group)*width+col] : -1;
    out[i] = p >= 0 && p < positions ? slots[(size_t) seq[qi]*positions+p] : -1;
}

void ggml_cuda_op_step5_csa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto stream = ctx.stream();
    const auto * a = dst->src[0];
    const auto * b = dst->src[1];
    if (dst->op == GGML_OP_STEP5_CSA_ROWS) {
        csa_rows<<<(ggml_nelements(dst)+255)/256,256,0,stream>>>((const int *) a->data,(const int *) b->data,
                (const int *) dst->src[2]->data,(int *) dst->data,a->ne[0],dst->ne[0],a->ne[2],b->ne[0]);
        return;
    }
    if (dst->op == GGML_OP_STEP5_INDEX_NORM) {
        float op[3]; memcpy(op, dst->op_params, sizeof(op));
        csa_index_norm<<<a->ne[1]*a->ne[2],256,0,stream>>>((const float *) a->data,(const float *) b->data,
                dst->src[2] ? (const float *) dst->src[2]->data : nullptr, (const int *) dst->src[3]->data,
                (float *) dst->data,a->ne[1],op[0],op[1],op[2] != 0);
    } else if (dst->op == GGML_OP_STEP5_CSA_SUMMARY) {
        if (dst->type == GGML_TYPE_F16) {
            csa_summary<<<dim3(dst->ne[1],dst->ne[2]),256,0,stream>>>(csa_view(a), (const int *) b->data,
                    (half *) dst->data, b->ne[0], dst->ne[1],
                    dst->src[2] ? (half *) dst->src[2]->data : nullptr,
                    dst->src[3] ? (int64_t *) dst->src[3]->data : nullptr,
                    dst->src[4] ? (const int64_t *) dst->src[4]->data : nullptr,
                    dst->src[2] ? dst->src[2]->ne[1] : 0);
        } else {
            csa_summary<<<dim3(dst->ne[1],dst->ne[2]),256,0,stream>>>(csa_view(a), (const int *) b->data,
                    (float *) dst->data, b->ne[0], dst->ne[1],
                    dst->src[2] ? (float *) dst->src[2]->data : nullptr,
                    dst->src[3] ? (int64_t *) dst->src[3]->data : nullptr,
                    dst->src[4] ? (const int64_t *) dst->src[4]->data : nullptr,
                    dst->src[2] ? dst->src[2]->ne[1] : 0);
        }
    } else if (dst->op == GGML_OP_STEP5_CSA_SCORES) {
        const auto * summary = dst->src[2];
        const int nr = summary->ne[1], nq = a->ne[2], columns = nq*16;
        const size_t ns = ggml_nelements(summary), nquery = ggml_nelements(a);
        if (ggml_get_op_params_i32(dst, 0)) {
            auto * qs = (half *) ((float *) dst->data + (size_t) nr*4*nq);
            GGML_ASSERT((char *) (qs+nquery) <= (char *) dst->data+ggml_nbytes(dst));
            csa_quantize<<<(nquery+255)/256,256,0,stream>>>((const float *) a->data,qs,nquery,true);
            if (summary->type == GGML_TYPE_F16) {
                csa_launch_fused(ctx,dst,qs,(const half *) summary->data);
            } else {
                csa_launch_fused(ctx,dst,qs,(const float *) summary->data);
            }
            return;
        }
        auto * dots = (float *) dst->data + (size_t) nr*4*nq;
        auto * qs = (half *) (dots + (size_t) nr*columns);
        auto * summary_scratch = qs + nquery;
        const bool f16 = summary->type == GGML_TYPE_F16;
        const half * sums = f16 ? (const half *) summary->data : summary_scratch;
        GGML_ASSERT((char *) (summary_scratch + (f16 ? 0 : ns)) <= (char *) dst->data + ggml_nbytes(dst));
        csa_quantize<<<(nquery+255)/256,256,0,stream>>>((const float *) a->data, qs, nquery, true);
        if (!f16) {
            csa_quantize<<<(ns+255)/256,256,0,stream>>>((const float *) summary->data, summary_scratch, ns, false);
        }
        const float alpha = 1, beta = 0;
        auto handle = ctx.cublas_handle();
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        for (int seq = 0; seq < summary->ne[2]; ++seq) {
            CUBLAS_CHECK(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,nr,columns,256,&alpha,
                    sums+(size_t) seq*nr*256,CUDA_R_16F,256,qs,CUDA_R_16F,256,
                    &beta,dots,CUDA_R_32F,nr,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            if (f16) {
                csa_scores<<<((size_t) nr*4*nq+255)/256,256,0,stream>>>(dots, (const float *) b->data,
                        (const half *) summary->data, (const int *) dst->src[3]->data, (const int *) dst->src[4]->data,
                        (float *) dst->data, nr, nq, seq);
            } else {
                csa_scores<<<((size_t) nr*4*nq+255)/256,256,0,stream>>>(dots, (const float *) b->data,
                        (const float *) summary->data, (const int *) dst->src[3]->data, (const int *) dst->src[4]->data,
                        (float *) dst->data, nr, nq, seq);
            }
        }
    } else if (dst->op == GGML_OP_STEP5_CSA_TOPK) {
        const auto * slots = dst->src[2];
        if (slots && ggml_get_op_params_i32(dst,0) <= 1024) {
            csa_topk<true,true><<<dst->ne[1]*dst->ne[2],256,0,stream>>>((const float *) a->data, (const int *) b->data,
                    (int *) dst->data, a->ne[0], ggml_get_op_params_i32(dst,0), (const int *) slots->data,
                    (const int *) dst->src[3]->data, slots->ne[0], a->ne[2], dst->ne[0]);
        } else if (slots) {
            csa_topk<true><<<dst->ne[1]*dst->ne[2],256,0,stream>>>((const float *) a->data, (const int *) b->data,
                    (int *) dst->data, a->ne[0], ggml_get_op_params_i32(dst,0), (const int *) slots->data,
                    (const int *) dst->src[3]->data, slots->ne[0], a->ne[2], dst->ne[0]);
        } else {
            csa_topk<false><<<dst->ne[1]*dst->ne[2],256,0,stream>>>((const float *) a->data, (const int *) b->data,
                    (int *) dst->data, a->ne[0], ggml_get_op_params_i32(dst,0), nullptr, nullptr, 0, a->ne[2], dst->ne[0]);
        }
    } else {
        float scale; memcpy(&scale, dst->op_params, sizeof(scale));
        csa_attention<<<dim3(a->ne[2],a->ne[1]),256,0,stream>>>((const float *) a->data,csa_view(b),csa_view(dst->src[2]),
                (const int *) dst->src[3]->data,(const int *) dst->src[4]->data,(const int *) dst->src[5]->data,
                (float *) dst->data,a->ne[0],a->ne[1],dst->src[3]->ne[0],dst->src[4]->ne[0],scale);
    }
}
