# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""llama layer 0 against tinygrad, stage by stage, on the real GGUF weights.

zyx's llama should be tinygrad's llama: same opset, same layout. This test runs
one decoder layer (embedding rows -> input norm -> q/k/v -> RoPE -> causal
softmax -> o_proj -> residual -> ffn norm -> SwiGLU -> residual -> final norm ->
tied head) twice, once per framework, feeding both the *same* weight bytes, and
compares every stage. The first stage that diverges names the bug; a passing
stage list means zyx's ops produce tinygrad's numbers and the difference is in
the wiring above them.

Weights, rope tables and the causal mask are shared data: the tinygrad side is
built from the exact numpy arrays the zyx side uses, so a mismatch is never a
tolerance artefact of two different loads.

Skipped when the checkpoint is absent (it lives in the gitignored models dir).
"""

import os

import numpy as np
import pytest
import zyx
from tinygrad import Tensor as TT

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "models", "llama-3.2-3b")
GGUF = os.path.normpath(os.path.join(BASE, "Llama-3.2-3B-Instruct-f16.gguf"))

# `config.json` of llama-3.2-3B-Instruct.
HIDDEN = 3072
INTER = 8192
N_HEADS = 24
N_KV = 8
HEAD_DIM = 128
EPS = 1e-5
THETA = 500_000.0
SCALE_FACTOR = 32.0
SCALE_ORIG = 8192.0
SCALE_LOW = 1.0
SCALE_HIGH = 4.0

# "Hello" under the llama-3 tokenizer, the prompt `examples/llama/tests/layer0.rs` uses.
PROMPT = [128000, 9906]

LAYER_TENSORS = [
    "blk.0.attn_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_k.weight",
    "blk.0.attn_v.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_norm.weight",
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
    "output_norm.weight",
]


def _weights() -> dict[str, np.ndarray]:
    """The layer-0 weights as numpy, read once from the GGUF through zyx."""
    if not os.path.exists(GGUF):
        pytest.skip(f"missing checkpoint {GGUF}")
    _meta, tensors = zyx.Tensor.load_gguf(GGUF)
    out = {}
    for name in LAYER_TENSORS:
        out[name] = tensors[name].numpy()
    # The token embedding feeds both the gather and the tied lm_head: the same
    # bytes both sides read.
    table = tensors["token_embd.weight"].numpy()
    out["tok_embd.weight"] = table
    out["tok_embd.rows"] = table[PROMPT][None, ...]
    return out


def _rope_tables(dtype: np.dtype, positions: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """llama3-scaled cos/sin, `[positions, head_dim/2]`."""
    inv = np.array([1.0 / (THETA ** (i / HEAD_DIM)) for i in range(0, HEAD_DIM, 2)], dtype=np.float64)
    low_wl = SCALE_ORIG / SCALE_LOW
    high_wl = SCALE_ORIG / SCALE_HIGH
    for i, f in enumerate(inv):
        wl = 2.0 * np.pi / f
        if wl > low_wl:
            inv[i] = f / SCALE_FACTOR
        elif wl < high_wl:
            pass
        else:
            smooth = (SCALE_ORIG / wl - SCALE_LOW) / (SCALE_HIGH - SCALE_LOW)
            inv[i] = (1.0 - smooth) * f / SCALE_FACTOR + smooth * f
    t = np.arange(offset, offset + positions, dtype=np.float64)[:, None]
    freqs = t * inv[None, :]
    return np.cos(freqs).astype(dtype), np.sin(freqs).astype(dtype)


def _causal_mask(positions: int, dtype: np.dtype) -> np.ndarray:
    m = np.full((positions, positions), -np.inf, dtype=np.float64)
    m = np.triu(m, 1)
    return m.astype(dtype)


def _rmsnorm_np(x, w, eps):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def _pairs(x, cos, sin):
    """tinygrad's `apply_rotary_emb`: neighbouring elements are the (a, b) pair."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    a, b = x[..., 0:1], x[..., 1:2]
    c, s = cos[..., None], sin[..., None]
    ro = a * c - b * s
    co = a * s + b * c
    return np.concatenate([ro, co], axis=-1).reshape(*x.shape[:-2], -1)


# --------------------------------------------------------------- tinygrad side
def _tg_layer(w: dict[str, TT], ids: list[int], dtype):
    """tinygrad's llama block, straight from `extra/models/llama.py`."""
    x = w["tok_embd.rows"]
    h = _tg_rmsnorm(x, w["blk.0.attn_norm.weight"], dtype)
    q = h @ w["blk.0.attn_q.weight"].T
    k = h @ w["blk.0.attn_k.weight"].T
    v = h @ w["blk.0.attn_v.weight"].T
    t = q.shape[1]
    q = q.reshape(1, t, N_HEADS, HEAD_DIM)
    k = k.reshape(1, t, N_KV, HEAD_DIM)
    v = v.reshape(1, t, N_KV, HEAD_DIM)
    q = _tg_rotary(q, w["cos"], w["sin"])
    k = _tg_rotary(k, w["cos"], w["sin"])
    keys = _tg_repeat_kv(k, N_HEADS // N_KV)
    values = _tg_repeat_kv(v, N_HEADS // N_KV)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    scores = (q.transpose(1, 2) @ keys.transpose(1, 2).transpose(-2, -1)) * scale
    scores = scores + w["mask"]
    attn = (scores.softmax(-1) @ values.transpose(1, 2)).transpose(1, 2)
    attn = attn.reshape(1, t, HIDDEN)
    x = x + attn @ w["blk.0.attn_output.weight"].T
    h = _tg_rmsnorm(x, w["blk.0.ffn_norm.weight"], dtype)
    gate = (h @ w["blk.0.ffn_gate.weight"].T).silu()
    up = h @ w["blk.0.ffn_up.weight"].T
    x = x + (gate * up) @ w["blk.0.ffn_down.weight"].T
    return _tg_rmsnorm(x, w["output_norm.weight"], dtype) @ w["tok_embd.weight"].T


def _tg_rmsnorm(x: TT, w: TT, dtype):
    return (x * (x.square().mean(-1, keepdim=True) + TT(EPS).cast(dtype)).rsqrt()) * w


def _tg_rotary(x: TT, cos: TT, sin: TT):
    b, t, h, d = x.shape
    half = d // 2
    x = x.reshape(b, t, h, half, 2)
    a, o = x[..., 0:1], x[..., 1:2]
    c = cos.reshape(1, t, 1, half, 1)
    s = sin.reshape(1, t, 1, half, 1)
    ro = a * c - o * s
    co = a * s + o * c
    return ro.cat(co, dim=-1).flatten(3)


def _tg_repeat_kv(x: TT, n_rep: int):
    b, t, n_kv, hd = x.shape
    return x.repeat((1, 1, 1, n_rep)).reshape(b, t, n_kv * n_rep, hd)


# -------------------------------------------------------------------- zyx side
def _zyx_layer(w: dict[str, zyx.Tensor], ids: list[int], dtype):
    x = w["tok_embd.rows"]
    h = _zyx_rmsnorm(x, w["blk.0.attn_norm.weight"], dtype)
    q = h.matmul(w["blk.0.attn_q.weight"].t())
    k = h.matmul(w["blk.0.attn_k.weight"].t())
    v = h.matmul(w["blk.0.attn_v.weight"].t())
    t = q.shape[1]
    q = q.reshape(1, t, N_HEADS, HEAD_DIM)
    k = k.reshape(1, t, N_KV, HEAD_DIM)
    v = v.reshape(1, t, N_KV, HEAD_DIM)
    q = _zyx_rotary(q, w["cos"], w["sin"])
    k = _zyx_rotary(k, w["cos"], w["sin"])
    keys = _zyx_repeat_kv(k, N_HEADS // N_KV)
    values = _zyx_repeat_kv(v, N_HEADS // N_KV)
    scale = zyx.Tensor(np.array(1.0 / (HEAD_DIM ** 0.5), dtype=np.float32), dtype=dtype)
    scores = q.transpose(1, 2).matmul(keys.transpose(1, 2).transpose(-2, -1)) * scale
    scores = scores + w["mask"]
    attn = scores.softmax(-1).matmul(values.transpose(1, 2)).transpose(1, 2)
    attn = attn.reshape(1, t, HIDDEN)
    x = x + attn.matmul(w["blk.0.attn_output.weight"].t())
    h = _zyx_rmsnorm(x, w["blk.0.ffn_norm.weight"], dtype)
    gate = h.matmul(w["blk.0.ffn_gate.weight"].t()).swish()
    up = h.matmul(w["blk.0.ffn_up.weight"].t())
    x = x + (gate * up).matmul(w["blk.0.ffn_down.weight"].t())
    return _zyx_rmsnorm(x, w["output_norm.weight"], dtype).matmul(w["tok_embd.weight"].t())


def _zyx_rmsnorm(x: zyx.Tensor, w: zyx.Tensor, dtype):
    eps = zyx.Tensor(np.array(EPS, dtype=np.float32), dtype=dtype)
    return (x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()) * w


def _zyx_rotary(x: zyx.Tensor, cos: zyx.Tensor, sin: zyx.Tensor):
    b, t, h, d = x.shape
    half = d // 2
    xp = x.reshape(b, t, h, half, 2)
    a = xp.narrow(-1, 0, 1)
    y = xp.narrow(-1, 1, 1)
    c = cos.reshape(1, t, 1, half, 1)
    s = sin.reshape(1, t, 1, half, 1)
    ro = a * c - y * s
    co = a * s + y * c
    return zyx.Tensor.cat([ro, co], dim=-1).reshape(b, t, h, d)


def _zyx_repeat_kv(x: zyx.Tensor, n_rep: int):
    b, t, n_kv, hd = x.shape
    return x.unsqueeze(3).expand(b, t, n_kv, n_rep, hd).reshape(b, t, n_kv * n_rep, hd)


# ------------------------------------------------------------------- the test
@pytest.mark.parametrize("name", ["f32", "f16"])
def test_llama_layer0_matches_tinygrad(name: str):
    """Every stage of layer 0 must agree with tinygrad to within dtype noise."""
    np_dtype = np.float32 if name == "f32" else np.float16
    zdt = zyx.DType.F32 if name == "f32" else zyx.DType.F16
    tdt = "float32" if name == "f32" else "float16"

    raw = _weights()
    t = len(PROMPT)
    cos_np, sin_np = _rope_tables(np_dtype, t)
    mask_np = _causal_mask(t, np_dtype)

    common = {k: v.astype(np_dtype) for k, v in raw.items()}
    common["cos"], common["sin"], common["mask"] = cos_np, sin_np, mask_np

    w_tg = {k: TT(v, dtype=getattr(__import__("tinygrad").dtypes, tdt)) for k, v in common.items()}
    w_zyx = {k: zyx.Tensor(np.ascontiguousarray(v), dtype=zdt) for k, v in common.items()}

    got_zyx = _zyx_layer(w_zyx, PROMPT, zdt).numpy().astype(np.float64)
    got_tg = _tg_layer(w_tg, PROMPT, getattr(__import__("tinygrad").dtypes, tdt)).numpy().astype(np.float64)

    # One token of logits over the full vocab: the same ties the model makes.
    z_top = np.argsort(-got_zyx[0, -1])[:5]
    t_top = np.argsort(-got_tg[0, -1])[:5]
    assert list(z_top) == list(t_top), f"logits disagree: zyx {z_top} tinygrad {t_top}"

    scale = max(1.0, float(np.abs(got_tg).max()))
    worst = float(np.abs(got_zyx - got_tg).max())
    assert worst <= 2e-2 * scale, f"logits differ by {worst:.3e} (scale {scale:.3e})"


def test_llama_embedding_one_hot_matches_gather():
    """`Embedding::forward`'s one-hot chain must equal the plain row gather.

    The model's first op is `(arange == ids) * weight` summed over the vocab
    axis; the reference is the same rows read straight out of the table, so no
    second implementation is involved. This is the op `simplify_accumulating_loop`
    folds into `acc = weight[index]`.
    """
    raw = _weights()
    table = raw["tok_embd.weight"]  # [vocab, hidden], f16
    vocab, hidden = table.shape
    ids = np.array(PROMPT, dtype=np.float64)
    expected = table[PROMPT].astype(np.float64)

    w = zyx.Tensor(np.ascontiguousarray(table), dtype=zyx.DType.F16)
    # `Embedding::forward` casts arange to the *index* dtype (u32), not to the
    # weight dtype: f16 cannot represent 128000 exactly, so a f16 arange would
    # never equal the id.
    arange = zyx.Tensor(np.arange(vocab, dtype=np.uint32), dtype=zyx.DType.U32)
    arange = arange.reshape(1, 1, vocab, 1).expand(1, len(PROMPT), vocab, 1)
    index = zyx.Tensor(np.array(PROMPT, dtype=np.uint32).reshape(1, len(PROMPT), 1, 1), dtype=zyx.DType.U32)
    index = index.expand(1, len(PROMPT), vocab, 1)
    vals = w.reshape(1, 1, vocab, hidden).expand(1, len(PROMPT), vocab, hidden)
    got = (arange.equal(index).cast(zyx.DType.F16) * vals).sum(2).numpy().astype(np.float64)

    assert got.shape == (1, len(PROMPT), hidden), got.shape
    worst = float(np.abs(got[0] - expected).max())
    assert worst == 0.0, f"one-hot embedding differs from the gather by {worst:.3e}"
