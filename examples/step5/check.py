#!/usr/bin/env python3
"""Compare GGML Step-5 CSA operations with the independent CPU oracle."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
import torch
from reference import bf16, e4m3, compress, scores, select, attention, index_norm, rope

P, I, L, F, S = C.c_void_p, C.c_int, C.c_int64, C.c_float, C.c_size_t


class Init(C.Structure):
    _fields_ = [('mem_size', S), ('mem_buffer', P), ('no_alloc', C.c_bool)]


class Backend:
    def __init__(self, directory, name):
        self.base = C.CDLL(str(directory / 'libggml-base.so'), mode=C.RTLD_GLOBAL)
        self.cpu = C.CDLL(str(directory / 'libggml-cpu.so'), mode=C.RTLD_GLOBAL)
        def bind(lib, name, ret, args):
            f = getattr(lib, name); f.restype = ret; f.argtypes = args
            return f
        self.init = bind(self.base, 'ggml_init', P, [Init])
        self.free = bind(self.base, 'ggml_free', None, [P])
        self.new = bind(self.base, 'ggml_new_tensor_4d', P, [P, I, L, L, L, L])
        self.graph = bind(self.base, 'ggml_new_graph', P, [P])
        self.expand = bind(self.base, 'ggml_build_forward_expand', None, [P, P])
        self.summary = bind(self.base, 'ggml_step5_csa_summary', P, [P, P, P])
        self.scores = bind(self.base, 'ggml_step5_csa_scores', P, [P]*6)
        self.topk = bind(self.base, 'ggml_step5_csa_topk', P, [P, P, P, I])
        self.attn = bind(self.base, 'ggml_step5_csa_attn', P, [P]*7+[F])
        self.alloc = bind(self.cpu, 'ggml_backend_alloc_ctx_tensors', P, [P, P])
        self.buffer_free = bind(self.cpu, 'ggml_backend_buffer_free', None, [P])
        self.set = bind(self.cpu, 'ggml_backend_tensor_set', None, [P, P, S, S])
        self.get = bind(self.cpu, 'ggml_backend_tensor_get', None, [P, P, S, S])
        self.compute = bind(self.cpu, 'ggml_backend_graph_compute', I, [P, P])
        if name == 'CPU':
            self.backend = bind(self.cpu, 'ggml_backend_cpu_init', P, [])()
            bind(self.cpu, 'ggml_backend_cpu_set_n_threads', None, [P, I])(self.backend, 8)
        else:
            self.cuda = C.CDLL(str(directory / 'libggml-cuda.so'), mode=C.RTLD_GLOBAL)
            self.backend = bind(self.cuda, 'ggml_backend_cuda_init', P, [I])(int(name.removeprefix('CUDA')))
        assert self.backend

    def run(self, q, w, k, z, main_q, main_k, main_v, positions, budget, sequences, shuffle=False, unified=False):
        ctx = self.init(Init(16*1024*1024, None, True))
        inputs = []
        def tensor(x):
            array = np.ascontiguousarray(x)
            types = {np.dtype('float32'): 0, np.dtype('float16'): 1, np.dtype('int32'): 26}
            dims = list(reversed(array.shape)) + [1]*(4-array.ndim)
            t = self.new(ctx, types[array.dtype], *dims)
            inputs.append((t, array))
            return t
        nq, groups, heads, _ = q.shape
        nseq, length, _ = k.shape
        capacity = (length+7)//8*8
        generator = np.random.default_rng(51)
        slots = np.full((nseq, capacity), -1, dtype=np.int32)
        kz = np.zeros((nseq, capacity, 1, 512), dtype=np.float16)
        mk = np.zeros((nseq, capacity, 4, main_q.shape[-1]), dtype=np.float16)
        mv = np.zeros_like(mk)
        for s in range(nseq):
            order = generator.permutation(length) if shuffle else np.arange(length)
            slots[s, :length] = s*capacity+order
            kz[s, order, 0, :256] = k[s].numpy().astype(np.float16)
            kz[s, order, 0, 256:] = z[s].numpy().astype(np.float16)
            mk[s, order] = main_k[s].numpy().astype(np.float16)
            mv[s, order] = main_v[s].numpy().astype(np.float16)
        if unified:
            kz = kz.reshape(1,nseq*capacity,1,512)
            mk = mk.reshape(1,nseq*capacity,4,main_q.shape[-1])
            mv = mv.reshape(mk.shape)
        tq = tensor(q.flatten(1, 2).numpy())
        tw = tensor(w.flatten(1, 2).numpy())
        ts = tensor(sequences.numpy().astype(np.int32)); tp = tensor(positions.numpy().astype(np.int32))
        tm = tensor(slots); tkz = tensor(kz)
        summary = self.summary(ctx, tkz, tm)
        logits = self.scores(ctx, tq, tw, summary, ts, tp)
        selected = self.topk(ctx, logits, tp, budget)
        output = self.attn(ctx, tensor(main_q.flatten(1, 2).numpy()), tensor(mk), tensor(mv), selected, tm, ts, main_q.shape[-1]**-.5)
        gf = self.graph(ctx)
        self.expand(gf, output)
        buffer = self.alloc(ctx, self.backend); assert buffer
        for t, array in inputs:
            self.set(t, array.ctypes.data, 0, array.nbytes)
        assert self.compute(self.backend, gf) == 0
        def read(t, shape, dtype=np.float32):
            out = np.empty(shape, dtype=dtype)
            self.get(t, out.ctypes.data, 0, out.nbytes)
            return torch.from_numpy(out)
        actual_summary = read(summary, (nseq, capacity//8, 256))[:, :length//8]
        actual_scores = read(logits, (nq, groups, capacity//8))
        actual_selected = read(selected, (nq, groups, (budget+1)*8), np.int32)
        actual_output = read(output, main_q.shape)
        self.buffer_free(buffer); self.free(ctx)
        return actual_summary, actual_scores, actual_selected, actual_output


def check_norm(backend):
    ctx = backend.init(Init(16*1024*1024, None, True))
    op = backend.base.ggml_step5_index_norm
    op.restype = P; op.argtypes = [P, P, P, P, P, F, F, C.c_bool]
    generator = torch.Generator().manual_seed(913)
    q = torch.randn((3,16,256),generator=generator)
    k = torch.randn((3,1,256),generator=generator)
    qw = torch.randn(256,generator=generator)*.05
    kw = 1+torch.randn(256,generator=generator)*.05
    kb = torch.randn(256,generator=generator)*.1
    positions = torch.tensor([0,4096,143999],dtype=torch.int32)
    inputs = []
    def tensor(x):
        a = x.contiguous().numpy()
        dims = list(reversed(a.shape))+[1]*(4-a.ndim)
        t = backend.new(ctx,26 if a.dtype==np.int32 else 0,*dims)
        inputs.append((t,a)); return t
    tq,tk,tqw,tkw,tkb,tp = map(tensor,(q,k,qw,kw,kb,positions))
    oq = op(ctx,tq,tqw,None,tp,1e-5,1e7,False)
    ok = op(ctx,tk,tkw,tkb,tp,1e-5,1e7,True)
    gf = backend.graph(ctx); backend.expand(gf,oq); backend.expand(gf,ok)
    buffer = backend.alloc(ctx,backend.backend); assert buffer
    for t,a in inputs: backend.set(t,a.ctypes.data,0,a.nbytes)
    assert backend.compute(backend.backend,gf)==0
    refs = index_norm(q,k,qw,kw,kb)
    report = {}
    for name,t,expected in zip(('Q','K'),(oq,ok),refs):
        expected = rope(expected,positions)
        actual = np.empty(expected.shape,dtype=np.float32)
        backend.get(t,actual.ctypes.data,0,actual.nbytes)
        error = np.abs(actual-expected.numpy())
        assert np.all(error<=np.maximum(np.abs(expected.numpy())/128,1e-4)), (name,float(error.max()))
        report[name] = {'max_abs':float(error.max()),'different_values':int(np.count_nonzero(error))}
    backend.buffer_free(buffer); backend.free(ctx)
    return report


def check_case(backend, length, budget=512, ties=False, multi=False, underflow=False, fast=False, early=False):
    generator = torch.Generator().manual_seed(73)
    nseq, nq, dim = (2 if multi else 1), 3, 192
    q = bf16(torch.randn((nq, 4, 4, 256), generator=generator))
    w = torch.randn((nq, 4, 4), generator=generator)
    if ties: w.zero_()
    k = bf16(torch.randn((nseq, length, 256), generator=generator))
    z = bf16(torch.randn((nseq, length, 256), generator=generator))
    if underflow:
        z.zero_(); z[:,0,0] = 200
    main_q = torch.randn((nq, 4, 16, dim), generator=generator)
    if fast: main_q = bf16(main_q)
    main_k = torch.randn((nseq, length, 4, dim), generator=generator).half().float()
    main_v = torch.randn((nseq, length, 4, dim), generator=generator).half().float()
    pos = torch.tensor([max(0, length-3), max(0, length-2), length-1], dtype=torch.int32)
    if early: pos = torch.tensor([0,7,8],dtype=torch.int32)
    seq = torch.tensor([0, int(multi), 0], dtype=torch.int32)
    expected_summary = torch.stack([compress(ks, zs) for ks, zs in zip(k, z)])
    expected_scores = torch.stack([scores(q[i:i+1], w[i:i+1], expected_summary[s])[0] for i, s in enumerate(seq)])
    expected_selected = select(expected_scores, pos, budget)
    expected_output = torch.cat([attention(main_q[i:i+1], main_k[s], main_v[s], expected_selected[i:i+1]) for i, s in enumerate(seq)])
    summary, logits, selected, output = backend.run(q, w, k, z, main_q, main_k, main_v, pos, budget, seq, multi, fast and multi)
    summary_diff = summary != expected_summary
    for si in range(nseq):
        kc = k[si,:length//8*8].view(length//8,8,256)
        zc = z[si,:length//8*8].view(length//8,8,256)
        weights = (zc-zc.amax(dim=(1,2),keepdim=True)).exp()
        raw = (weights*kc).sum(1)/weights.sum(1).clamp_min(1e-20)
        low, high = e4m3(bf16(raw-1e-6)), e4m3(bf16(raw+1e-6))
        assert ((summary[si] >= low) & (summary[si] <= high)).all(), 'summary outside FP32 rounding interval'
    # FP32 reductions can straddle a BF16 midpoint before E4M3 conversion.
    # Test the score kernel separately using those already verified summaries.
    numerical_scores = torch.stack([scores(q[i:i+1],w[i:i+1],summary[si])[0] for i,si in enumerate(seq)])
    maximum_score_error = 0.
    for i, p in enumerate(pos):
        visible = int(p)//8
        if visible:
            error = (logits[i, :, :visible]-numerical_scores[i, :, :visible]).abs().max().item()
            maximum_score_error = max(maximum_score_error, error)
            torch.testing.assert_close(logits[i, :, :visible], numerical_scores[i, :, :visible], rtol=2e-5, atol=5e-4)
        assert torch.isneginf(logits[i, :, visible:]).all()
    torch.testing.assert_close(selected, expected_selected, rtol=0, atol=0)
    error = output-expected_output
    nmse = float(error.square().sum()/expected_output.square().sum())
    if fast:
        # Tensor-core Flash Attention uses F16 probability and accumulator tiles.
        bound = float(main_v.abs().max())/1024+1e-5
        assert torch.isfinite(output).all() and nmse < 1e-6 and float(error.abs().max()) < bound, (nmse,float(error.abs().max()),bound)
        if early:
            torch.testing.assert_close(output[0],expected_output[0],rtol=5e-4,atol=1e-6)
    else:
        torch.testing.assert_close(output, expected_output, rtol=5e-4, atol=1e-5)
    return {'length': length, 'budget': budget, 'ties': ties, 'sequences': nseq, 'underflow': underflow, 'indexed_flash': fast, 'early': early, 'attention_nmse': nmse,
            'summary_exact': not bool(summary_diff.any()), 'summary_rounding_boundaries': int(summary_diff.sum()), 'selected_exact': True,
            'score_max_abs': maximum_score_error, 'attention_max_abs': (output-expected_output).abs().max().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libdir', type=Path, required=True)
    parser.add_argument('--backend', default='CPU')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    backend = Backend(args.libdir.resolve(), args.backend)
    report = {'backend': args.backend, 'norm_rope': check_norm(backend), 'cases': []}
    cases = [(73, 3, False, False)]
    cases += [(length, 512, False, False) for length in (1,7,8,9,511,512,513,4095,4096,4103,4104,4105)]
    cases += [(4112,512,True,False), (4112,512,False,True)]
    cases = [(*case,False,False) for case in cases]+[(73,3,False,False,True,False)]
    cases += [(8192,512,False,False,False,True), (8192,512,False,True,False,True)]
    for length, budget, ties, multi, underflow, fast in cases:
        record = check_case(backend, length, budget, ties, multi, underflow, fast)
        report['cases'].append(record)
        print(json.dumps(record), flush=True)
    for multi in (False,True):
        record = check_case(backend,8192,512,False,multi,False,True,True)
        report['cases'].append(record)
        print(json.dumps(record),flush=True)
    report['passed'] = True
    if args.output: args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
