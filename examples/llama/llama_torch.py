# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Torch reference for llama prefill, one step, layer-0 detail + full-model logits.

Uses the same GGUF file as zyx (no download), reads config.json and
tokenizer.json (no hardcode), runs on cuda:0 fp16. Python gguf shapes are
already [out, in], so no transpose is needed here; Rust remap .t() only
compensates its own loader dim order. Linear is x @ W.t() on both sides.
"""

import json
import os

import torch
import torch.nn.functional as F
from gguf import GGUFReader
from tokenizers import Tokenizer

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "llama-3.2-3b")
GGUF_PATH = os.path.join(BASE, "Llama-3.2-3B-Instruct-f16.gguf")
CONFIG_PATH = os.path.join(BASE, "config.json")
TOKENIZER_PATH = os.path.join(BASE, "tokenizer.json")


def load_gguf_weights(path):
    reader = GGUFReader(path)
    weights = {}
    for t in reader.tensors:
        import numpy as np

        arr = np.asarray(t.data)
        torch_t = torch.from_numpy(arr.copy())
        weights[t.name] = torch_t
    return weights


def precompute_rope(cfg):
    head_dim = cfg["head_dim"] if cfg.get("head_dim") is not None else cfg["hidden_size"] // cfg["num_attention_heads"]
    theta = cfg["rope_theta"]
    import math

    inv_freq = [1.0 / (theta ** (i / head_dim)) for i in range(0, head_dim, 2)]
    scaling = cfg.get("rope_scaling")
    if scaling is not None:
        low_wavelen = scaling["original_max_position_embeddings"] / scaling["low_freq_factor"]
        high_wavelen = scaling["original_max_position_embeddings"] / scaling["high_freq_factor"]
        factor = scaling["factor"]
        fixed = []
        for f in inv_freq:
            wavelen = 2.0 * math.pi / f
            if wavelen > low_wavelen:
                fixed.append(f / factor)
            elif wavelen < high_wavelen:
                fixed.append(f)
            else:
                smooth = (scaling["original_max_position_embeddings"] / wavelen - scaling["low_freq_factor"]) / (
                    scaling["high_freq_factor"] - scaling["low_freq_factor"]
                )
                fixed.append((1.0 - smooth) * f / factor + smooth * f)
        inv_freq = fixed
    max_pos = cfg["max_position_embeddings"]
    t = torch.arange(max_pos, dtype=torch.float32).unsqueeze(1)
    freqs = t @ torch.tensor(inv_freq, dtype=torch.float32).unsqueeze(0)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin, start_pos):
    # x: [B, H, L, D], cos/sin: [max_pos, half]
    half = x.shape[-1] // 2
    seq_len = x.shape[2]
    c = cos[start_pos : start_pos + seq_len, :].to(x.dtype)
    s = sin[start_pos : start_pos + seq_len, :].to(x.dtype)
    # broadcast to [1, 1, L, half]
    c = c.unsqueeze(0).unsqueeze(0)
    s = s.unsqueeze(0).unsqueeze(0)
    x1 = x[..., :half]
    x2 = x[..., half:]
    # standard NeoX rotate-half
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    # x: [B, H_kv, L, D] -> [B, H, L, D]
    return x.repeat_interleave(n_rep, dim=1)


def rmsnorm(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps)
    return (normed * weight.float()).to(x.dtype)


def attention_layer0(xs, w, cfg, cos, sin, start_pos):
    # xs: [B, L, D] fp16 cuda
    num_heads = cfg["num_attention_heads"]
    num_kv = cfg.get("num_key_value_heads") or num_heads
    head_dim = cfg["head_dim"] if cfg.get("head_dim") is not None else cfg["hidden_size"] // num_heads
    b, seq, _ = xs.shape
    q = xs @ w["blk.0.attn_q.weight"].t()
    k = xs @ w["blk.0.attn_k.weight"].t()
    v = xs @ w["blk.0.attn_v.weight"].t()
    q = q.view(b, seq, num_heads, head_dim).transpose(1, 2)
    k = k.view(b, seq, num_kv, head_dim).transpose(1, 2)
    v = v.view(b, seq, num_kv, head_dim).transpose(1, 2)
    q = apply_rope(q, cos, sin, start_pos)
    k = apply_rope(k, cos, sin, start_pos)
    k = repeat_kv(k, num_heads // num_kv)
    v = repeat_kv(v, num_heads // num_kv)
    scale = 1.0 / (head_dim**0.5)
    attn = (q.float() @ k.float().transpose(-2, -1)) * scale
    if seq > 1:
        mask = torch.full((seq, seq), float("-inf")).triu(diagonal=1)
        attn = attn + mask.to(attn.device, attn.dtype)
    probs = F.softmax(attn, dim=-1).to(xs.dtype)
    out = (probs.float() @ v.float()).to(xs.dtype)
    out = out.transpose(1, 2).reshape(b, seq, num_heads * head_dim)
    out = out @ w["blk.0.attn_output.weight"].t()
    return {"q_rope": q, "k_rope": k, "attn_probs": probs, "attn_out": out}


def mlp_layer0(xs, w):
    gate = xs @ w["blk.0.ffn_gate.weight"].t()
    up = xs @ w["blk.0.ffn_up.weight"].t()
    hidden = F.silu(gate.float()).to(xs.dtype) * up
    down = hidden @ w["blk.0.ffn_down.weight"].t()
    return {"gate": gate, "up": up, "down": down}


def show(name, t):
    f = t.detach().float().cpu().flatten()[:8].tolist()
    vals = " ".join(f"{v:.6g}" for v in f)
    print(f"{name} shape={list(t.shape)} dtype={t.dtype} min={t.float().min().item():.6g} max={t.float().max().item():.6g} mean={t.float().mean().item():.6g} first8=[{vals}]", flush=True)


def main_prefill():
    device = torch.device("cuda:0")
    dtype = torch.float16
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    ids = tok.encode("Hello", add_special_tokens=True).ids
    print(f"prompt ids={ids}", flush=True)
    weights = load_gguf_weights(GGUF_PATH)
    # move needed tensors to cuda fp16 (norms cast to fp16 to match zyx)
    needed_prefixes = ("token_embd.weight", "output_norm.weight") + tuple(f"blk.{i}." for i in range(cfg["num_hidden_layers"]))
    w = {}
    for name, t in weights.items():
        if name.startswith(needed_prefixes):
            if t.dtype == torch.float32:
                w[name] = t.to(device, dtype)
            else:
                w[name] = t.to(device, dtype)
    cos, sin = precompute_rope(cfg)
    cos = cos.to(device, dtype)
    sin = sin.to(device, dtype)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    xs = F.embedding(input_ids, w["token_embd.weight"].float()).to(dtype)
    show("embed_out", xs)
    eps = cfg["rms_norm_eps"]
    # layer 0 detail
    h = rmsnorm(xs, w["blk.0.attn_norm.weight"], eps)
    show("l0_input_norm", h)
    attn = attention_layer0(h, w, cfg, cos, sin, 0)
    show("l0_q_rope", attn["q_rope"])
    show("l0_k_rope", attn["k_rope"])
    show("l0_attn_probs", attn["attn_probs"])
    show("l0_attn_out", attn["attn_out"])
    xs_res = xs + attn["attn_out"].to(dtype)
    h2 = rmsnorm(xs_res, w["blk.0.ffn_norm.weight"], eps)
    show("l0_post_norm", h2)
    mlp = mlp_layer0(h2, w)
    show("l0_gate", mlp["gate"])
    show("l0_up", mlp["up"])
    show("l0_mlp_down", mlp["down"])
    xs = xs_res + mlp["down"].to(dtype)
    show("l0_block_out", xs)
    # remaining layers, no per-stage dump except final
    for i in range(1, cfg["num_hidden_layers"]):
        p = f"blk.{i}."
        h = rmsnorm(xs, w[p + "attn_norm.weight"], eps)
        b, seq, _ = h.shape
        nh = cfg["num_attention_heads"]
        nkv = cfg.get("num_key_value_heads") or nh
        hd = cfg["head_dim"] if cfg.get("head_dim") is not None else cfg["hidden_size"] // nh
        q = (h @ w[p + "attn_q.weight"].t()).view(b, seq, nh, hd).transpose(1, 2)
        k = (h @ w[p + "attn_k.weight"].t()).view(b, seq, nkv, hd).transpose(1, 2)
        v = (h @ w[p + "attn_v.weight"].t()).view(b, seq, nkv, hd).transpose(1, 2)
        q = apply_rope(q, cos, sin, 0)
        k = apply_rope(k, cos, sin, 0)
        k = repeat_kv(k, nh // nkv)
        v = repeat_kv(v, nh // nkv)
        a = (q.float() @ k.float().transpose(-2, -1)) * (1.0 / (hd**0.5))
        if seq > 1:
            a = a + torch.full((seq, seq), float("-inf"), device=device).triu(1).to(a.dtype)
        pr = F.softmax(a, dim=-1).to(dtype)
        o = (pr.float() @ v.float()).to(dtype).transpose(1, 2).reshape(b, seq, nh * hd)
        o = o @ w[p + "attn_output.weight"].t()
        xs = xs + o.to(dtype)
        h2 = rmsnorm(xs, w[p + "ffn_norm.weight"], eps)
        g = h2 @ w[p + "ffn_gate.weight"].t()
        u = h2 @ w[p + "ffn_up.weight"].t()
        d = (F.silu(g.float()).to(dtype) * u) @ w[p + "ffn_down.weight"].t()
        xs = xs + d.to(dtype)
    xs = rmsnorm(xs, w["output_norm.weight"], eps)
    show("final_norm", xs)
    last = xs[:, -1:, :]
    # tied head
    logits = last.float() @ w["token_embd.weight"].float().t()
    show("logits_last", logits)
    top5 = torch.topk(logits.flatten(), 5)
    print(f"top5 ids={top5.indices.cpu().tolist()} vals={top5.values.cpu().tolist()}", flush=True)


if __name__ == "__main__":
    main_prefill()
