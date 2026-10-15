#!/usr/bin/env python3
"""Independent numerical oracle for the StepFun Step-5 region indexer.

Equations follow inference/model.py and inference/kernel.py at HF revision
364b3b03fed2477aa0953fe855a2c4c8c48ee3cc. This script never loads the full model.
"""
import argparse
import json
from pathlib import Path
import torch

REVISION = '364b3b03fed2477aa0953fe855a2c4c8c48ee3cc'


def bf16(x):
    return x.to(torch.bfloat16).float()


def e4m3(x):
    return x.to(torch.float8_e4m3fn).float()


def index_norm(q, k, q_weight, k_weight, k_bias, eps=1e-5):
    q, k = bf16(q), bf16(k)
    q = bf16(q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
             * (1 + bf16(q_weight)))
    mean = k.mean(-1, keepdim=True)
    variance = k.square().mean(-1, keepdim=True) - mean.square()
    k = bf16((k - mean) * torch.rsqrt(variance + eps) * bf16(k_weight) + bf16(k_bias))
    return q, k


def rope(x, positions, theta=1e7, rotary=32):
    half = rotary // 2
    frequencies = theta ** (-torch.arange(0, rotary, 2).float() / rotary)
    angle = positions.float().view(-1, 1) * frequencies
    cosine, sine = angle.cos(), angle.sin()
    while cosine.ndim < x.ndim:
        cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    result = x.clone()
    a, b = x[..., :half], x[..., half:rotary]
    result[..., :half] = a * cosine - b * sine
    result[..., half:rotary] = a * sine + b * cosine
    return bf16(result)


def compress(k, z, region=8):
    count = k.shape[0] // region
    k, z = k[:count*region].view(count, region, -1), z[:count*region].view(count, region, -1)
    shift = z.amax(dim=(1, 2), keepdim=True)
    weight = (z - shift).exp()
    mean = (weight * k).sum(1) / weight.sum(1)
    return e4m3(bf16(mean))


def scores(q, weights, summary):
    # Four independent provider groups, each with four signed head weights.
    dots = torch.einsum('qghd,rd->qghr', e4m3(q), e4m3(summary))
    return (dots.relu() * (weights * 0.5)[..., None]).sum(2)


def select(logits, positions, topk=512, region=8):
    nq, groups, _ = logits.shape
    selected = torch.full((nq, groups, (topk+1)*region), -1, dtype=torch.int32)
    for qi, pos in enumerate(positions.tolist()):
        complete = pos // region
        for group in range(groups):
            order = torch.argsort(logits[qi, group, :complete], descending=True, stable=True)
            chosen = order[:topk].sort().values
            tokens = (chosen[:, None]*region + torch.arange(region)).flatten()
            tokens = torch.cat((tokens, torch.arange(complete*region, pos+1)))
            selected[qi, group, :tokens.numel()] = tokens.int()
    return selected


def attention(q, k, v, selected):
    nq, groups, heads, dim = q.shape
    out = torch.empty_like(q)
    for qi in range(nq):
        for group in range(groups):
            indices = selected[qi, group]
            indices = indices[indices >= 0].long()
            logits = q[qi, group] @ k[indices, group].T / dim**0.5
            out[qi, group] = logits.softmax(-1) @ v[indices, group]
    return out


def write_tensor(path, tensor):
    tensor.detach().contiguous().numpy().tofile(path)


def fixture(directory, length=73, topk=3):
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(73)
    dim, groups, heads = 256, 4, 4
    positions = torch.tensor([length-3, length-2, length-1], dtype=torch.int32)
    q = bf16(torch.randn((3, groups, heads, dim), generator=generator))
    k = bf16(torch.randn((length, dim), generator=generator))
    z = bf16(torch.randn((length, dim), generator=generator))
    w = torch.randn((3, groups, heads), generator=generator)
    summary = compress(k, z)
    logits = scores(q, w, summary)
    selected = select(logits, positions, topk=topk)
    geometry = {'tokens': length, 'queries': 3, 'proxy': dim, 'groups': groups, 'heads_per_group': heads, 'topk': topk, 'region': 8}
    for name, tensor in {'q': q, 'k': k, 'z': z, 'w': w, 'positions': positions,
                         'summary': summary, 'scores': logits, 'selected': selected}.items():
        write_tensor(directory / (name + '.bin'), tensor)
    (directory / 'geometry.json').write_text(json.dumps(geometry, indent=2)+'\n')
    return geometry


def self_test():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1)
    dim, groups, heads = 256, 4, 4
    k = bf16(torch.randn((4112, dim), generator=generator))
    z = bf16(torch.randn((4112, dim), generator=generator))
    q = bf16(torch.randn((1, groups, heads, dim), generator=generator))
    w = torch.randn((1, groups, heads), generator=generator)
    main_q = torch.randn((1, groups, 2, 8), generator=generator)
    main_k = torch.randn((4112, groups, 8), generator=generator)
    main_v = torch.randn((4112, groups, 8), generator=generator)
    checks = []
    summary = compress(k, z)
    logits = scores(q, w, summary)
    for length in (1,7,8,9,511,512,513,4095,4096,4103,4104,4105):
        pos = torch.tensor([length-1], dtype=torch.int32)
        ids = select(logits, pos)
        for g in range(groups):
            valid = ids[0,g][ids[0,g]>=0]
            assert valid.numel() <= 4104
            assert int(valid.max()) == length-1
            assert torch.all(valid[1:]>valid[:-1])
            assert torch.all(valid <= length-1)
            if length<=4104:
                assert torch.equal(valid, torch.arange(length).int())
        if length<=4104:
            sparse_out = attention(main_q,main_k,main_v,ids)
            dense = torch.stack([(main_q[0,g] @ main_k[:length,g].T / 8**0.5).softmax(-1) @ main_v[:length,g] for g in range(groups)])[None]
            torch.testing.assert_close(sparse_out,dense,rtol=1e-6,atol=1e-6)
        checks.append({'length':length,'passed':True})
    ties = torch.zeros((1,groups,514))
    ids=select(ties,torch.tensor([4111],dtype=torch.int32))
    assert torch.equal(ids[0,0,:4096],torch.arange(4096).int())
    assert torch.equal(ids[0,0,4096:],torch.arange(4104,4112).int())
    # Future regions must never change selection for earlier queries.
    altered=logits.clone();altered[:,:,4:]=1e9
    pos=torch.tensor([31],dtype=torch.int32)
    assert torch.equal(select(logits,pos),select(altered,pos))
    # A constant per-coordinate Z offset must not change CSA in exact arithmetic.
    shifted=compress(k[:32],z[:32]+2)
    torch.testing.assert_close(summary[:4],shifted,rtol=0,atol=0)
    # Group-local choices: another group's scores must not change group zero.
    changed=logits.clone();changed[:,1:]=-changed[:,1:]
    pos=torch.tensor([4104],dtype=torch.int32)
    assert torch.equal(select(logits,pos)[:,0],select(changed,pos)[:,0])
    # Zero-centered Q scale and conventional K scale intentionally differ.
    q0,k0=index_norm(q,k[:1],torch.zeros(dim),torch.ones(dim),torch.zeros(dim))
    assert torch.isfinite(q0).all() and torch.isfinite(k0).all() and q0.abs().sum()>0
    return {'reference_revision':REVISION,'boundary_checks':checks,'stable_ties':True,'causal_selection':True,'provider_isolation':True,'zero_centered_norm':True,'passed':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fixtures',type=Path)
    args=parser.parse_args()
    report=self_test()
    if args.fixtures: report['fixture']=fixture(args.fixtures)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
