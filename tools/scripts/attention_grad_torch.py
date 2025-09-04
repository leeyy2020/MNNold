#!/usr/bin/env python3
"""
Generate PyTorch reference gradients for MNN attention grad numeric test.

Usage:
  python3 tools/scripts/attention_grad_torch.py /path/to/dump_dir

The dump directory should be produced by running:
  ./build/attention_grad_test.out --dump /tmp/attn_dump

This script reads case_N_meta.txt, q.txt, k.txt, v.txt, do.txt and writes
qg_torch.txt, kg_torch.txt, vg_torch.txt for each case.
"""
import os
import sys
import math
from typing import Tuple

try:
    import torch
except Exception as e:
    print("[ERROR] PyTorch is required: pip install torch", file=sys.stderr)
    raise


def read_txt(path: str):
    with open(path, 'r') as f:
        data = [float(x) for x in f.read().strip().split()]
    return data


def write_txt(path: str, arr):
    with open(path, 'w') as f:
        f.write(" ".join(str(float(x)) for x in arr) + "\n")


def load_case(prefix: str) -> Tuple[Tuple[int, int, int, int, int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # meta
    with open(prefix + 'meta.txt', 'r') as f:
        parts = f.read().strip().split()
        seq_len, kv_seq_len, num_head, kv_num_head, head_dim = map(int, parts)
    # tensors
    q = torch.tensor(read_txt(prefix + 'q.txt'), dtype=torch.float32)
    k = torch.tensor(read_txt(prefix + 'k.txt'), dtype=torch.float32)
    v = torch.tensor(read_txt(prefix + 'v.txt'), dtype=torch.float32)
    try:
        do = torch.tensor(read_txt(prefix + 'do.txt'), dtype=torch.float32)
    except FileNotFoundError:
        do = torch.ones(seq_len * num_head * head_dim, dtype=torch.float32)

    # reshape to MNN layout: [1, seq_len, num_head, head_dim], [1, kv_seq_len, kv_num_head, head_dim]
    q = q.view(1, seq_len, num_head, head_dim).clone().detach().requires_grad_(True)
    k = k.view(1, kv_seq_len, kv_num_head, head_dim).clone().detach().requires_grad_(True)
    v = v.view(1, kv_seq_len, kv_num_head, head_dim).clone().detach().requires_grad_(True)
    do = do.view(1, seq_len, num_head, head_dim)
    return (seq_len, kv_seq_len, num_head, kv_num_head, head_dim), q, k, v, do


def attention_forward(q, k, v, do, seq_len, kv_seq_len, num_head, kv_num_head, head_dim):
    # Expand K/V heads to align with Q heads according to kv grouping
    group = num_head // kv_num_head
    # Build index mapping: head h uses kv_h = h // group
    idx = torch.arange(num_head) // group  # [num_head], values in [0, kv_num_head)
    # Gather along kv_head dim (dim=2)
    k_full = torch.index_select(k, dim=2, index=idx)
    v_full = torch.index_select(v, dim=2, index=idx)
    # scores: [1, seq_len, num_head, kv_seq_len]
    # scores[b,i,h,j] = q[b,i,h,:] @ k_full[b,j,h,:]
    scores = torch.einsum('bihd,bjhd->bihj', q, k_full)
    scores = scores * (1.0 / math.sqrt(float(head_dim)))
    probs = torch.softmax(scores, dim=-1)
    # y[b,i,h,d] = sum_j probs[b,i,h,j] * v_full[b,j,h,d]
    y = torch.einsum('bihj,bjhd->bihd', probs, v_full)
    # Loss = sum(y * do)
    loss = (y * do).sum()
    return loss, (k_full, v_full, idx)


def run_case(dump_dir: str, case_id: int) -> bool:
    prefix = os.path.join(dump_dir, f"case_{case_id}_")
    if not os.path.exists(prefix + 'meta.txt'):
        return False
    (seq_len, kv_seq_len, num_head, kv_num_head, head_dim), q, k, v, do = load_case(prefix)
    loss, (k_full, v_full, idx) = attention_forward(q, k, v, do, seq_len, kv_seq_len, num_head, kv_num_head, head_dim)
    loss.backward()

    # q.grad shape matches q; k.grad, v.grad match originals
    # Save grads in the same flat order as MNN test expects
    write_txt(prefix + 'qg_torch.txt', q.grad.flatten().tolist())

    # Accumulate grads from expanded k_full/v_full back to original k/v
    # Build zeros grads and scatter-add from expanded head axis
    k_grad = torch.zeros_like(k)
    v_grad = torch.zeros_like(v)
    # For each head h, add grad from k_full[..., h, :] to kv_h in k_grad
    for h in range(num_head):
        kv_h = int(idx[h].item())
        k_grad[:, :, kv_h, :] += k_full.grad[:, :, h, :]
        v_grad[:, :, kv_h, :] += v_full.grad[:, :, h, :]

    write_txt(prefix + 'kg_torch.txt', k_grad.flatten().tolist())
    write_txt(prefix + 'vg_torch.txt', v_grad.flatten().tolist())
    print(f"[torch] Wrote grads for case {case_id}")
    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: attention_grad_torch.py /path/to/dump_dir", file=sys.stderr)
        sys.exit(1)
    dump_dir = sys.argv[1]
    # iterate cases until meta missing
    cid = 0
    found_any = False
    while run_case(dump_dir, cid):
        found_any = True
        cid += 1
    if not found_any:
        print(f"No cases found under {dump_dir}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()

