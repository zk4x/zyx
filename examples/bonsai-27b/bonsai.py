# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Ternary-Bonsai-2-27B (`qwen35` trunk, PQ2_0 weights) in Python.

A line-for-line port of the Rust reference in `src/`: `config.rs` -> `Config`,
`prism.rs` -> `Prism`, `weights.rs` -> `QWeight`, `model.rs` -> `Bonsai`, and
`main.rs` -> `main()`. Every dtype, layout and graph decision is the Rust one;
the Python-side code differs only in host type conversions (`int(...)`,
`.numpy()`) that Rust does with `.item::<T>()`/`.to_vec::<T>()`.

The forward is ordinary tensor arithmetic on a `Tape`, exactly like the Rust
example: the trace is frozen per sequence length and replayed on every later
step. Weight tensors stay in their gguf PQ2_0 super-block form;
`QWeight.dequantize` is part of the traced graph, so the egraph decides
whether to fuse the dequant into the projection or materialize it.

Every dtype choice follows llama.cpp: activations, norms, conv, delta-net
state, logits and the rotation are F32; the full-attention K/V cache is F16.

Run: `python3 bonsai.py MODEL.gguf --tokenizer-file tokenizer.json --prompt Hello`
"""

import argparse
import math
import os
import random
import sys
import time

import zyx
from zyx import DType, Tensor

# Elements covered by one PQ2_0 super-block.
PQ2_BLOCK = 128
# Gated-delta-rule chunk length used by llama.cpp's chunked prefill.
CHUNK = 64

# Byte width of one PQ2_0 super-block row.
PQ2_ROW = 34


class Config:
    """Hyperparameters of the `qwen35` trunk, read from the GGUF metadata.

    Nothing here is hardcoded to the 27B checkpoint: every dimension comes out
    of the file, so a different qwen35 export loads with the same code. The
    only derived values are the ones the architecture defines
    (`key_dim = state_size * group_count`, `conv_dim = 2*key_dim + inner`).
    """

    def __init__(
        self,
        n_blocks,
        hidden,
        ffn,
        n_head,
        n_kv,
        head_dim,
        rot_dim,
        rope_theta,
        eps,
        full_attention_interval,
        ssm_state,
        ssm_k_heads,
        ssm_v_heads,
        ssm_v_head_dim,
        conv_kernel,
    ):
        self.n_blocks = n_blocks
        self.hidden = hidden
        self.ffn = ffn
        self.n_head = n_head
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.rot_dim = rot_dim
        self.rope_theta = rope_theta
        self.eps = eps
        self.full_attention_interval = full_attention_interval
        self.ssm_state = ssm_state
        self.ssm_k_heads = ssm_k_heads
        self.ssm_v_heads = ssm_v_heads
        self.ssm_v_head_dim = ssm_v_head_dim
        self.conv_kernel = conv_kernel

    @classmethod
    def from_metadata(cls, metadata):
        """Reads the qwen35 hyperparameters out of a GGUF metadata map."""
        nextn = metadata.get("qwen35.nextn_predict_layers", 0)
        if nextn > 0:
            raise ValueError(f"qwen35.nextn_predict_layers = {nextn}: this loader only implements the non-MTP trunk")
        ssm_state = int(metadata["qwen35.ssm.state_size"])
        ssm_k_heads = int(metadata["qwen35.ssm.group_count"])
        ssm_v_heads = int(metadata["qwen35.ssm.time_step_rank"])
        ssm_inner = int(metadata["qwen35.ssm.inner_size"])
        if ssm_inner % ssm_v_heads != 0:
            raise ValueError(
                f"qwen35.ssm.inner_size {ssm_inner} is not divisible by time_step_rank {ssm_v_heads}"
            )
        return cls(
            n_blocks=int(metadata["qwen35.block_count"]),
            hidden=int(metadata["qwen35.embedding_length"]),
            ffn=int(metadata["qwen35.feed_forward_length"]),
            n_head=int(metadata["qwen35.attention.head_count"]),
            n_kv=int(metadata["qwen35.attention.head_count_kv"]),
            head_dim=int(metadata["qwen35.attention.key_length"]),
            rot_dim=int(metadata["qwen35.rope.dimension_count"]),
            rope_theta=float(metadata["qwen35.rope.freq_base"]),
            eps=float(metadata["qwen35.attention.layer_norm_rms_epsilon"]),
            full_attention_interval=int(metadata["qwen35.full_attention_interval"]),
            ssm_state=ssm_state,
            ssm_k_heads=ssm_k_heads,
            ssm_v_heads=ssm_v_heads,
            ssm_v_head_dim=ssm_inner // ssm_v_heads,
            conv_kernel=int(metadata["qwen35.ssm.conv_kernel"]),
        )

    def ssm_key_dim(self):
        """Key width of one linear-attention block."""
        return self.ssm_state * self.ssm_k_heads

    def ssm_value_dim(self):
        """Value width of one linear-attention block."""
        return self.ssm_state * self.ssm_v_heads

    def conv_dim(self):
        """Concatenated q/k/v width fed to the depthwise conv."""
        return 2 * self.ssm_key_dim() + self.ssm_value_dim()

    def is_full(self, i):
        """`True` when block `i` is a full-attention block."""
        return (i + 1) % self.full_attention_interval == 0


def hadamard(block_size):
    """Normalized Sylvester-Walsh-Hadamard matrix."""
    scale = 1.0 / math.sqrt(block_size)
    data = []
    for row in range(block_size):
        for col in range(block_size):
            sign = 1.0 if bin(row & col).count("1") % 2 == 0 else -1.0
            data.append(sign * scale)
    return Tensor.from_vec(data, [block_size, block_size])


class Prism:
    """Prism's rotation-folded weights.

    The checkpoint stores most matmul weights multiplied by a blockwise
    normalized Walsh-Hadamard matrix (and a per-input-dimension sign vector).
    llama.cpp restores the product on the activation side: before a folded
    weight is used, the activation becomes `H (s * x)`; the embedding table is
    stored pre-rotated and is restored with `s * (H z)` right after the row
    lookup. This class implements exactly those two transforms with ordinary
    tensor ops.
    """

    def __init__(self, block_size, gdn_v_grouped, rotation, signs, folded, inverse):
        self.block_size = block_size
        self.gdn_v_grouped = gdn_v_grouped
        self.rotation = rotation
        self.signs = signs
        self.folded = folded
        # `inverse` is the method below, so the name set is kept apart.
        self.inverse_names = inverse

    @classmethod
    def from_metadata(cls, metadata):
        """Reads the `prism.hadamard.*` metadata.

        A checkpoint without the block loads with an empty `Prism` (no
        transforms, no inverse tables).
        """
        folded = set(metadata.get("prism.hadamard.weight_names", []))
        inverse = set(metadata.get("prism.hadamard.inverse_weight_names", []))
        block_size = int(metadata.get("prism.hadamard.block_size", 0))
        gdn_v_grouped = bool(metadata.get("prism.hadamard.gdn_v_grouped", False))

        if block_size == 0:
            if folded or inverse:
                raise ValueError(
                    "prism.hadamard.weight_names is set but prism.hadamard.block_size is missing"
                )
            return cls(0, gdn_v_grouped, Tensor.zeros([0, 0], dtype=DType.F32), {}, folded, inverse)
        if block_size < 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError(f"prism.hadamard.block_size {block_size} is not a positive power of two")

        widths = [int(w) for w in metadata.get("prism.hadamard.sign_widths", [])]
        values = [int(v) for v in metadata.get("prism.hadamard.sign_values", [])]
        total = sum(widths)
        if total != len(values):
            raise ValueError(
                f"prism.hadamard.sign_widths sum to {total} but sign_values has {len(values)} entries"
            )
        signs = {}
        offset = 0
        for width in widths:
            if width <= 0:
                raise ValueError(f"prism.hadamard.sign_widths has a non-positive width {width}")
            data = [float(v) for v in values[offset : offset + width]]
            if any(v != 1.0 and v != -1.0 for v in data):
                raise ValueError(
                    f"prism.hadamard.sign_values for width {width} contain values other than +/-1"
                )
            offset += width
            signs[width] = Tensor.from_vec(data, [width])

        return cls(block_size, gdn_v_grouped, hadamard(block_size), signs, folded, inverse)

    def to(self, dev):
        """Moves the rotation and sign tensors to `dev`."""
        self.rotation = self.rotation.to(dev)
        for width, sign in self.signs.items():
            self.signs[width] = sign.to(dev)

    def is_folded(self, name):
        """`True` when the weight name is folded and its activation is rotated."""
        return name in self.folded

    def is_inverse(self, name):
        """`True` when the table stores rotated rows."""
        return name in self.inverse_names

    def forward(self, x, width):
        """`H (s * x)` for an activation of last width `width`."""
        signs = self.signs.get(width)
        if signs is None:
            raise ValueError(f"prism.hadamard has no sign vector for width {width}")
        return self.rotate(x * signs, width)

    def inverse(self, x, width):
        """`s * (H z)` for a looked-up table row of width `width`."""
        signs = self.signs.get(width)
        if signs is None:
            raise ValueError(f"prism.hadamard has no sign vector for width {width}")
        return self.rotate(x, width) * signs

    def rotate(self, x, width):
        """Blockwise normalized Walsh-Hadamard rotation of the last dimension."""
        if width % self.block_size != 0:
            raise ValueError(f"prism.hadamard block {self.block_size} does not divide width {width}")
        shape = x.symbolic_shape()
        rows = x.numel() // self.block_size
        flat = x.reshape([rows, self.block_size])
        rotated = flat.matmul(self.rotation)
        return rotated.reshape(shape)


class QWeight:
    """One PQ2_0 projection on device: the raw `[num_blocks, 34]` U8 tensor plus
    the logical `[rows, cols]` shape resolved once at load."""

    def __init__(self, blocks, rows, cols, folded=False):
        if blocks.dtype != DType.U8:
            raise ValueError(f"PQ2_0 weight must be raw U8 blocks, got {blocks.dtype}")
        shape = blocks.resolve_shape()
        if len(shape) != 2 or shape[1] != PQ2_ROW:
            raise ValueError(f"PQ2_0 weight must be [num_blocks, 34], got {shape}")
        if shape[0] * PQ2_BLOCK != rows * cols:
            raise ValueError(
                f"PQ2_0 weight has {shape[0]} blocks ({shape[0] * PQ2_BLOCK} values) "
                f"but logical shape is [{rows}, {cols}]"
            )
        self.blocks = blocks
        self.rows = rows
        self.cols = cols
        self.folded = folded

    def dequantize(self, dtype):
        """Dequantizes to a dense `[rows, cols]` tensor of `dtype`."""
        return self.blocks.dequantize(zyx.QDType.PQ2_0, dtype).reshape([self.rows, self.cols])


class Ffn:
    """SwiGLU weights."""

    def __init__(self, gate, up, down, norm):
        self.gate = gate
        self.up = up
        self.down = down
        self.norm = norm


class FullBlock:
    """One full-attention block and its F16 key/value cache."""

    def __init__(self, norm, q, k, v, o, q_norm, k_norm, ffn, cache_k, cache_v):
        self.norm = norm
        self.q = q
        self.k = k
        self.v = v
        self.o = o
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.ffn = ffn
        self.cache_k = cache_k
        self.cache_v = cache_v

    def forward(self, cfg, prism, rope, mask, x, pos, cache_len, seq):
        h = rms_norm(x, self.norm, cfg.eps)
        qg = linear(prism, h, self.q)
        qg = qg.reshape([seq, cfg.n_head, 2, cfg.head_dim])
        q = qg.narrow(2, 0, 1).squeeze([2])
        gate = qg.narrow(2, 1, 1).squeeze([2])
        q = rms_norm(q, self.q_norm, cfg.eps)
        k = linear(prism, h, self.k).reshape([seq, cfg.n_kv, cfg.head_dim])
        k = rms_norm(k, self.k_norm, cfg.eps)
        v = linear(prism, h, self.v).reshape([seq, cfg.n_kv, cfg.head_dim])

        q = rope.apply(q, pos, seq, cfg.rot_dim, cfg.head_dim)
        k = rope.apply(k, pos, seq, cfg.rot_dim, cfg.head_dim)

        real = cache_len - pos
        self.cache_k.narrow(0, pos, real).assign(k.narrow(0, 0, real).cast(DType.F16))
        self.cache_v.narrow(0, pos, real).assign(v.narrow(0, 0, real).cast(DType.F16))

        kk = self.cache_k.narrow(0, 0, cache_len).transpose(0, 1)
        vv = self.cache_v.narrow(0, 0, cache_len).transpose(0, 1)
        rep = cfg.n_head // cfg.n_kv
        kk = kk.unsqueeze(1).expand([-1, rep, -1, -1]).reshape([Tensor(cfg.n_head), cache_len, Tensor(cfg.head_dim)])
        vv = vv.unsqueeze(1).expand([-1, rep, -1, -1]).reshape([Tensor(cfg.n_head), cache_len, Tensor(cfg.head_dim)])

        q = q.transpose(0, 1)
        scale = Tensor(1.0 / math.sqrt(cfg.head_dim))
        attn = q.cast(DType.F16).matmul(kk.transpose(1, 2).cast(DType.F16), acc_dtype=DType.F32, out_dtype=DType.F32) * scale
        if seq != 1:
            attn = attn + mask.unsqueeze(0)
        probs = attn.softmax([-1])
        out = (
            probs.cast(DType.F16)
            .matmul(vv, acc_dtype=DType.F32, out_dtype=DType.F32)
            .transpose(0, 1)
            .reshape([seq, cfg.n_head * cfg.head_dim])
        )
        gate = gate.reshape([seq, cfg.n_head * cfg.head_dim])
        out = out * gate.sigmoid()
        o = linear(prism, out, self.o)
        return x + o


class LinearBlock:
    """One gated-delta-net block plus its conv history and recurrent state."""

    def __init__(
        self,
        norm,
        qkv,
        gate,
        beta,
        alpha,
        conv_w,
        ssm_a,
        ssm_dt,
        ssm_norm,
        ssm_out,
        ffn,
        conv_state,
        state,
    ):
        self.norm = norm
        self.qkv = qkv
        self.gate = gate
        self.beta = beta
        self.alpha = alpha
        self.conv_w = conv_w
        self.ssm_a = ssm_a
        self.ssm_dt = ssm_dt
        self.ssm_norm = ssm_norm
        self.ssm_out = ssm_out
        self.ffn = ffn
        self.conv_state = conv_state
        self.state = state

    def forward(self, cfg, prism, x, pos, real, valid, seq):
        h = rms_norm(x, self.norm, cfg.eps)
        # A fresh sequence (position 0) starts from the zero recurrent state
        # and zero conv history, like llama.cpp's reset. Multiplying by
        # `1 - (pos == 0)` keeps the reset a runtime scalar, so one frozen
        # plan serves both a fresh prompt and a continuing decode.
        keep = Tensor(1.0) - pos.equal(0).cast(DType.F32)
        history = self.conv_state * keep
        state = self.state * keep

        qkv = linear(prism, h, self.qkv)
        z = linear(prism, h, self.gate)
        beta = linear_input(h, self.beta)
        alpha = linear_input(h, self.alpha)

        # Depthwise causal conv over history + this chunk, then SiLU.
        cat = Tensor.cat([history, qkv], 0)
        mixed = Tensor.zeros([seq, cfg.conv_dim()], dtype=DType.F32)
        for k in range(cfg.conv_kernel):
            window = cat.narrow(0, k, seq)
            weight = self.conv_w.narrow(1, k, 1).squeeze([1])
            mixed = mixed + window * weight.unsqueeze(0)
        mixed = mixed.swish()
        # Keep the last three *real* rows; the padded tail is not history.
        conv_state = cat.narrow(0, real, cfg.conv_kernel - 1).contiguous()
        self.conv_state.assign(conv_state)

        q, k, v = split_qkv(cfg, mixed, seq)
        eps = cfg.eps
        q = q * (q.square().sum([-1], keepdim=True) + eps).rsqrt()
        k = k * (k.square().sum([-1], keepdim=True) + eps).rsqrt()
        scale = Tensor(1.0 / math.sqrt(cfg.ssm_state))
        q = q * scale

        rep = cfg.ssm_v_heads // cfg.ssm_k_heads
        q = repeat_tiled(q, rep)
        k = repeat_tiled(k, rep)

        b = beta.sigmoid() * valid.unsqueeze(1)
        g = (alpha + self.ssm_dt).softplus(1.0, 20.0) * self.ssm_a * valid.unsqueeze(1)
        if seq == 1:
            core, new_state = delta_step(q, k, v, g, b, state)
        else:
            core, new_state = delta_chunked(q, k, v, g, b, state)
        self.state.assign(new_state)

        core = core.reshape([seq, cfg.ssm_v_heads, cfg.ssm_v_head_dim])
        z = z.reshape([seq, cfg.ssm_v_heads, cfg.ssm_v_head_dim])
        normed = rms_norm(core, self.ssm_norm, cfg.eps)
        normed = normed * z.swish()
        normed = normed.reshape([seq, cfg.ssm_value_dim()])

        # The folded out-projection is stored in grouped V order; the
        # activation leaves `delta_*` in llama.cpp's tiled order.
        if prism.gdn_v_grouped:
            vd = cfg.ssm_v_head_dim
            nk = cfg.ssm_k_heads
            rep = cfg.ssm_v_heads // nk
            normed = normed.reshape([seq, rep, nk, vd]).transpose(1, 2).reshape([seq, cfg.ssm_value_dim()])

        o = linear(prism, normed, self.ssm_out)
        return x + o


class Rope:
    """RoPE tables for every position the cache can hold."""

    def __init__(self, cos, sin):
        self.cos = cos
        self.sin = sin

    @classmethod
    def new(cls, cfg, max_context):
        """Builds the F32 cos/sin tables for llama.cpp's interleaved MRoPE with
        text-only positions (all axes share `pos`)."""
        half = cfg.rot_dim // 2
        cos = [0.0] * (max_context * half)
        sin = [0.0] * (max_context * half)
        for pos in range(max_context):
            for pair in range(half):
                angle = pos * cfg.rope_theta ** (-2.0 * pair / cfg.rot_dim)
                cos[pos * half + pair] = math.cos(angle)
                sin[pos * half + pair] = math.sin(angle)
        return cls(Tensor.from_vec(cos, [max_context, half]), Tensor.from_vec(sin, [max_context, half]))

    def to(self, dev):
        """Moves both tables to `dev`."""
        self.cos = self.cos.to(dev)
        self.sin = self.sin.to(dev)

    def apply(self, x, pos, seq, rot_dim, head_dim):
        """Partial RoPE on `[seq, heads, head_dim]`.

        llama.cpp pairs dimension `j` with `j + rot_dim/2`; the untouched tail
        is concatenated back.
        """
        cos = self.cos.narrow(0, pos, seq)
        sin = self.sin.narrow(0, pos, seq)
        x = x.transpose(0, 1).unsqueeze(0)
        rotated = x.narrow(3, 0, rot_dim)
        rest = x.narrow(3, rot_dim, head_dim - rot_dim)
        rotated = rotated.rope(cos, sin)
        return Tensor.cat([rotated, rest], 3).squeeze([0]).transpose(0, 1)


class Plan:
    """A frozen step. The key is the padded sequence length; decode steps use
    `1` because a single query needs no causal mask and the position is a
    replay variable."""

    def __init__(self, frozen):
        self.frozen = frozen


class Bonsai:
    """Ternary-Bonsai-2-27B."""

    def __init__(self, cfg, prism, emb, output, output_norm, blocks, rope, max_context, max_prefill):
        self.cfg = cfg
        self.prism = prism
        self.emb = emb
        self.output = output
        self.output_norm = output_norm
        self.blocks = blocks
        self.rope = rope
        self.max_context = max_context
        # Padded prefill width; a multiple of the GDN chunk length.
        self.max_prefill = max_prefill
        self.plans = {}

    @classmethod
    def from_gguf(cls, path, max_context, max_prefill):
        """Loads the trunk from a GGUF file.

        `max_context` sizes the full-attention K/V caches. `max_prefill` is the
        fixed padded width every prompt chunk is traced at; it is rounded up to
        a multiple of the 64-token GDN chunk. Decode always uses one token.
        """
        max_prefill = (max_prefill + CHUNK - 1) // CHUNK * CHUNK
        if max_prefill <= 0:
            raise ValueError("max_prefill must be positive")
        metadata, host = Tensor.load_gguf(path)
        cfg = Config.from_metadata(metadata)
        prism = Prism.from_metadata(metadata)
        dev = compute_device()
        prism.to(dev)

        def take(name):
            if name not in host:
                raise ValueError(f"gguf is missing tensor `{name}`")
            return host.pop(name).to(dev)

        def take_f32(name):
            """A dense gguf tensor as F32 on `dev`.

            A BF16 tensor is widened on the host: PTX has no bf16 `cvt` below
            sm_80 and the device reports BF16 as unsupported there, so no BF16
            value may enter a kernel on such a card.
            """
            if name not in host:
                raise ValueError(f"gguf is missing tensor `{name}`")
            tensor = host.pop(name)
            if tensor.dtype == DType.BF16:
                return Tensor(tensor.numpy().astype("float32")).to(dev)
            return tensor.to(dev).cast(DType.F32)

        emb_blocks = take("token_embd.weight")
        vocab = emb_blocks.resolve_shape()[0] * PQ2_BLOCK // cfg.hidden
        emb = QWeight(emb_blocks, vocab, cfg.hidden)
        output = QWeight(take("output.weight"), vocab, cfg.hidden, prism.is_folded("output.weight"))
        output_norm = take("output_norm.weight")

        blocks = []
        for i in range(cfg.n_blocks):
            pre = f"blk.{i}."
            ffn = Ffn(
                quant(take, prism, pre + "ffn_gate.weight", cfg.ffn, cfg.hidden),
                quant(take, prism, pre + "ffn_up.weight", cfg.ffn, cfg.hidden),
                quant(take, prism, pre + "ffn_down.weight", cfg.hidden, cfg.ffn),
                take(pre + "post_attention_norm.weight"),
            )
            norm = take(pre + "attn_norm.weight")
            if cfg.is_full(i):
                q_dim = 2 * cfg.n_head * cfg.head_dim
                kv_dim = cfg.n_kv * cfg.head_dim
                blocks.append(
                    FullBlock(
                        norm,
                        quant(take, prism, pre + "attn_q.weight", q_dim, cfg.hidden),
                        quant(take, prism, pre + "attn_k.weight", kv_dim, cfg.hidden),
                        quant(take, prism, pre + "attn_v.weight", kv_dim, cfg.hidden),
                        quant(take, prism, pre + "attn_output.weight", cfg.hidden, cfg.n_head * cfg.head_dim),
                        take(pre + "attn_q_norm.weight"),
                        take(pre + "attn_k_norm.weight"),
                        ffn,
                        Tensor.zeros([max_context, cfg.n_kv, cfg.head_dim], dtype=DType.F16).contiguous(),
                        Tensor.zeros([max_context, cfg.n_kv, cfg.head_dim], dtype=DType.F16).contiguous(),
                    )
                )
            else:
                blocks.append(
                    LinearBlock(
                        norm,
                        quant(take, prism, pre + "attn_qkv.weight", cfg.conv_dim(), cfg.hidden),
                        quant(take, prism, pre + "attn_gate.weight", cfg.ssm_value_dim(), cfg.hidden),
                        take_f32(pre + "ssm_beta.weight"),
                        take_f32(pre + "ssm_alpha.weight"),
                        take(pre + "ssm_conv1d.weight"),
                        take(pre + "ssm_a"),
                        take(pre + "ssm_dt.bias"),
                        take(pre + "ssm_norm.weight"),
                        quant(take, prism, pre + "ssm_out.weight", cfg.hidden, cfg.ssm_value_dim()),
                        ffn,
                        Tensor.zeros([cfg.conv_kernel - 1, cfg.conv_dim()], dtype=DType.F32).contiguous(),
                        Tensor.zeros([cfg.ssm_v_heads, cfg.ssm_state, cfg.ssm_state], dtype=DType.F32).contiguous(),
                    )
                )

        rope = Rope.new(cfg, max_context + max_prefill)
        rope.to(dev)
        return cls(cfg, prism, emb, output, output_norm, blocks, rope, max_context, max_prefill)

    def forward(self, input_ids, start_pos):
        """Runs one step and returns logits `[1, vocab]` for the last real token.

        Prompt chunks are padded to `max_prefill`; decode is the one-token
        plan. Both plans keep the real length and start position as runtime
        variables, so one frozen graph serves every prompt and every decode
        position.
        """
        cfg = self.cfg
        real = int(input_ids.numel().item())
        if start_pos + real > self.max_context:
            raise ValueError(
                f"step needs positions {start_pos}..{start_pos + real - 1}, above max_context {self.max_context}"
            )
        if real == 1:
            padded = 1
            ids_host = [int(input_ids.cast(DType.I64).reshape([1]).numpy()[0])]
            valid_host = [1.0]
        else:
            if real > self.max_prefill:
                raise ValueError(f"prompt chunk has {real} tokens, above max_prefill {self.max_prefill}")
            ids = [int(v) for v in input_ids.cast(DType.I64).reshape([real]).numpy()]
            padded = self.max_prefill
            ids_host = ids + [0] * (padded - real)
            valid_host = [1.0] * real + [0.0] * (padded - real)
        ids = Tensor(ids_host)
        valid = Tensor.from_vec(valid_host, [padded])
        pos = Tensor.variable(start_pos)
        real_t = Tensor.variable(real)
        cache_len = pos + real_t
        if padded in self.plans:
            return self.plans[padded].frozen.replay([ids, valid, pos, real_t])[0]

        tape = zyx.Tape.new(self.params())
        tape.add(ids)
        tape.add(valid)

        # Causal mask built from runtime position variables: rows are the
        # padded queries, columns are the cache positions actually read.
        if padded == 1:
            mask = Tensor.zeros([1, 1], dtype=DType.F32)
        else:
            rows = Tensor.arange(0, padded, 1).reshape([padded, 1])
            cols = Tensor.arange(0, self.max_context, 1).reshape([1, self.max_context])
            limit = pos + rows + 1
            keep = cols.cmplt(limit)
            mask = (keep.cast(DType.F32) - 1.0) * 1e30
            mask = mask.narrow(1, 0, cache_len)

        # Embedding: row-gather the folded PQ2_0 table, then restore the
        # primal basis with `s * (H z)`. Padded rows are zeroed so they can
        # never leak through the conv or the state.
        blocks_per_row = cfg.hidden // PQ2_BLOCK
        row = ids.reshape([padded, 1]) * blocks_per_row
        col = Tensor.arange(0, blocks_per_row, 1).reshape([1, blocks_per_row])
        index = (row + col).reshape([padded * blocks_per_row]).cast(DType.I32)
        picked = self.emb.blocks.index_select(0, index)
        x = picked.dequantize(zyx.QDType.PQ2_0, DType.F32).reshape([padded, cfg.hidden])
        if self.prism.is_inverse("token_embd.weight"):
            x = self.prism.inverse(x, cfg.hidden)
        x = x * valid.unsqueeze(1)

        for block in self.blocks:
            if isinstance(block, FullBlock):
                h = block.forward(cfg, self.prism, self.rope, mask, x, pos, cache_len, padded)
            else:
                h = block.forward(cfg, self.prism, x, pos, real_t, valid, padded)
            x = ffn_forward(self.prism, block.ffn, h, cfg.eps)

        # Only the last real token needs logits.
        last = x.narrow(0, real_t - Tensor(1), 1)
        last = rms_norm(last, self.output_norm, cfg.eps)
        logits = linear(self.prism, last, self.output)

        outputs = [logits]
        for block in self.blocks:
            if isinstance(block, FullBlock):
                outputs.append(block.cache_k)
                outputs.append(block.cache_v)
            else:
                outputs.append(block.conv_state)
                outputs.append(block.state)
        frozen = tape.freeze([ids, valid, pos, real_t], outputs)
        res = frozen.replay([ids, valid, pos, real_t])
        self.plans[padded] = Plan(frozen)
        return res[0]

    def params(self):
        """Every tensor the tape promotes: weights and owned caches."""
        params = [self.emb.blocks, self.output.blocks, self.output_norm]
        for block in self.blocks:
            if isinstance(block, FullBlock):
                params.extend([block.norm, block.q_norm, block.k_norm, block.cache_k, block.cache_v])
                params.extend([block.q.blocks, block.k.blocks, block.v.blocks, block.o.blocks])
                params.extend([block.ffn.norm, block.ffn.gate.blocks, block.ffn.up.blocks, block.ffn.down.blocks])
            else:
                params.extend(
                    [
                        block.norm,
                        block.beta,
                        block.alpha,
                        block.conv_w,
                        block.ssm_a,
                        block.ssm_dt,
                        block.ssm_norm,
                        block.conv_state,
                        block.state,
                    ]
                )
                params.extend([block.qkv.blocks, block.gate.blocks, block.ssm_out.blocks])
                params.extend([block.ffn.norm, block.ffn.gate.blocks, block.ffn.up.blocks, block.ffn.down.blocks])
        return params


def rms_norm(x, weight, eps):
    """`RMSNorm::forward` of `zyx-nn`, built from a loaded weight tensor."""
    dtype = x.dtype
    x_normed = x * ((x * x).mean([-1], keepdim=True) + Tensor(eps).cast(dtype)).rsqrt()
    return x_normed * weight


def linear(prism, x, w):
    """A quantized projection: F16 operands on the tensor cores, F32 accumulator
    and F32 result (llama.cpp's `ggml_mul_mat` contract)."""
    if w.folded:
        x = prism.forward(x, w.cols)
    return x.cast(DType.F16).matmul(w.dequantize(DType.F16).t(), acc_dtype=DType.F32, out_dtype=DType.F32)


def linear_input(x, weight):
    """A dense BF16/F32 projection (the small `ssm_beta`/`ssm_alpha` weights),
    with the same F16-operand/F32-accumulator contract."""
    return x.cast(DType.F16).matmul(weight.cast(DType.F16).t(), acc_dtype=DType.F32, out_dtype=DType.F32)


def ffn_forward(prism, ffn, x, eps):
    """SwiGLU feed-forward with its own pre-norm and residual."""
    h = rms_norm(x, ffn.norm, eps)
    gate = linear(prism, h, ffn.gate)
    up = linear(prism, h, ffn.up)
    mid = gate.swish() * up
    down = linear(prism, mid, ffn.down)
    return x + down


def split_qkv(cfg, mixed, seq):
    """Splits the mixed conv output into q/k/v head tensors."""
    kd = cfg.ssm_key_dim()
    q = mixed.narrow(1, 0, kd).reshape([seq, cfg.ssm_k_heads, cfg.ssm_state])
    k = mixed.narrow(1, kd, kd).reshape([seq, cfg.ssm_k_heads, cfg.ssm_state])
    v = mixed.narrow(1, 2 * kd, cfg.ssm_value_dim()).reshape([seq, cfg.ssm_v_heads, cfg.ssm_v_head_dim])
    return q, k, v


def repeat_tiled(x, rep):
    """Repeats `H_k` key heads to `H_v` value heads in llama.cpp's tiled order
    (`v_head % H_k`), not the interleaved HF order."""
    seq, heads, dim = x.shape
    return x.unsqueeze(1).expand([-1, rep, -1, -1]).reshape([seq, heads * rep, dim])


def delta_step(q, k, v, g, b, state):
    """Llama.cpp's token-recurrent gated delta rule (one decode step)."""
    q = q.squeeze([0])
    k = k.squeeze([0])
    v = v.squeeze([0])
    g = g.squeeze([0]).exp()
    b = b.squeeze([0])

    kv = (state * k.unsqueeze(2)).sum([1])
    delta = (v - g.unsqueeze(1) * kv) * b.unsqueeze(1)
    state = state * g.unsqueeze(1).unsqueeze(1) + k.unsqueeze(2) * delta.unsqueeze(1)
    core = (state * q.unsqueeze(2)).sum([1])
    return core.unsqueeze(0), state


def delta_chunked(q, k, v, g, b, state):
    """Llama.cpp's chunked gated delta rule (prefill)."""
    seq, h, d = q.shape
    pad = (CHUNK - seq % CHUNK) % CHUNK
    chunks = (seq + pad) // CHUNK

    def pad_t(x):
        if pad == 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[1] = pad
        return Tensor.cat([x, Tensor.zeros(pad_shape, dtype=DType.F32)], 1)

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    g = g.transpose(0, 1)
    b = b.transpose(0, 1)
    q = pad_t(q).reshape([h, chunks, CHUNK, d])
    k = pad_t(k).reshape([h, chunks, CHUNK, d])
    v = pad_t(v).reshape([h, chunks, CHUNK, d])
    g = pad_t(g).reshape([h, chunks, CHUNK])
    b = pad_t(b).reshape([h, chunks, CHUNK])

    gcs = g.cumsum(2)
    gexp = gcs.exp()
    gi = gcs.unsqueeze(3)
    gj = gcs.unsqueeze(2)
    decay = (gi - gj).exp() * lower(CHUNK, 0)
    strict = lower(CHUNK, -1)

    b3 = b.unsqueeze(3)
    v_b = v * b3
    k_b = k * b3

    a = k_b.matmul(k.transpose(2, 3)) * decay * strict
    n = -a
    identity = Tensor.eye(CHUNK, DType.F32).unsqueeze(0).unsqueeze(0).expand([h, chunks, CHUNK, CHUNK])
    m = identity
    npow = n
    for _ in range(6):
        m = m + npow.matmul(m)
        npow = npow.matmul(npow)

    v_new = m.matmul(v_b)
    k_cd = m.matmul(k_b * gexp.unsqueeze(3))
    q_g = q * gexp.unsqueeze(3)

    cores = []
    for chunk in range(chunks):
        qc = q_g.narrow(1, chunk, 1).squeeze([1])
        kc = k.narrow(1, chunk, 1).squeeze([1])
        kcd = k_cd.narrow(1, chunk, 1).squeeze([1])
        vnc = v_new.narrow(1, chunk, 1).squeeze([1])
        gc = gcs.narrow(1, chunk, 1).squeeze([1])
        dc = decay.narrow(1, chunk, 1).squeeze([1])

        a2 = qc.matmul(kc.transpose(1, 2)) * dc * strict
        v_prime = kcd.matmul(state)
        v_new_c = vnc - v_prime
        core = qc.matmul(state) + a2.matmul(v_new_c)

        g_last = gc.narrow(1, CHUNK - 1, 1)
        kg = kc * (g_last - gc).exp().unsqueeze(2)
        state = state * g_last.exp().unsqueeze(2) + kg.transpose(1, 2).matmul(v_new_c)
        cores.append(core)

    core = Tensor.cat(cores, 1).transpose(0, 1).narrow(0, 0, seq)
    return core, state


def lower(n, diagonal):
    """A `[n, n]` F32 lower-triangular mask (`diagonal` = 0 keeps the diagonal)."""
    data = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            if j <= i + diagonal:
                data[i * n + j] = 1.0
    return Tensor.from_vec(data, [n, n])


def quant(take, prism, name, rows, cols):
    """Loads one quantized projection and records whether it is folded."""
    return QWeight(take(name), rows, cols, prism.is_folded(name))


def compute_device():
    """The device weights live on. `ZYX_BONSAI_DEVICE` overrides the pick.

    Without the override this is the first non-CPU device `Dev.all()` reports,
    else CPU: the Rust `compute_device` rule.
    """
    spec = os.environ.get("ZYX_BONSAI_DEVICE")
    if spec is not None:
        return parse_device(spec)
    devices = zyx.Dev.all()
    for dev in devices:
        if repr(dev) != "Dev::C":
            return dev
    return zyx.Dev.cpu()


def parse_device(spec):
    """`Dev::from_str` spelled out for the Python device classes."""
    kind, _, ordinal = spec.partition(":")
    if kind == "cpu":
        return zyx.Dev.cpu()
    if kind == "cuda":
        return zyx.Dev.cuda(int(ordinal) if ordinal else 0)
    if kind == "vulkan":
        return zyx.Dev.vulkan(int(ordinal) if ordinal else 0)
    if kind == "opencl":
        return zyx.Dev.opencl(int(ordinal) if ordinal else 0)
    raise ValueError(f"ZYX_BONSAI_DEVICE: unknown device `{spec}`")


class Sampler:
    """Top-k/top-p sampler; `temperature <= 0` is greedy."""

    def __init__(self, seed, temperature, top_p, top_k):
        self.rng = random.Random(seed)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def sample(self, logits):
        best = max(range(len(logits)), key=logits.__getitem__)
        if self.temperature <= 0.0:
            return best
        inv = 1.0 / self.temperature
        top = max(logits)
        probs = [(i, math.exp((x - top) * inv)) for i, x in enumerate(logits)]
        probs.sort(key=lambda p: p[1], reverse=True)
        probs = probs[: self.top_k]
        total = sum(p for _, p in probs)
        probs = [(i, p / total) for i, p in probs]
        if 0.0 < self.top_p < 1.0:
            cum = 0.0
            keep = len(probs)
            for i, (_, p) in enumerate(probs):
                cum += p
                if cum >= self.top_p:
                    keep = i + 1
                    break
            probs = probs[:keep]
            total = sum(p for _, p in probs)
            probs = [(i, p / total) for i, p in probs]
        weights = [p for _, p in probs]
        pick = self.rng.choices(range(len(probs)), weights=weights)[0]
        return probs[pick][0]


def main():
    parser = argparse.ArgumentParser(description="Ternary-Bonsai-2-27B")
    parser.add_argument("model", help="GGUF model path")
    parser.add_argument("--tokenizer-file", help="Tokenizer JSON path; without it the prompt is read as integer ids")
    parser.add_argument("--prompt", default="Hello", help="Raw prompt (no chat template)")
    parser.add_argument("-n", "--sample-len", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--max-context", type=int, default=8192, help="Context window; sizes the full-attention K/V cache")
    parser.add_argument("--batch-size", type=int, default=2048, help="Prompt tokens per forward call")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=299_792_458)
    parser.add_argument("--ids-only", action="store_true", help="Print the prompt token ids and nothing else")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer_file is not None:
        import tokenizers

        tokenizer = tokenizers.Tokenizer.from_file(args.tokenizer_file)
    if tokenizer is not None:
        ids = list(tokenizer.encode(args.prompt, add_special_tokens=True).ids)
    else:
        ids = [int(x) for x in args.prompt.split()]
    if not ids:
        raise ValueError("prompt produced no token ids")
    if args.ids_only:
        print(ids)
        return 0

    model = Bonsai.from_gguf(args.model, args.max_context, args.batch_size)
    sampler = Sampler(args.seed, args.temperature, args.top_p, args.top_k)
    started = time.monotonic()

    pos = 0
    logits = None
    for start in range(0, len(ids), args.batch_size):
        chunk = ids[start : start + args.batch_size]
        logits = model.forward(Tensor(chunk), pos)
        pos += len(chunk)
    row = logits.squeeze([0]).cast(DType.F32).numpy().tolist()
    next_id = sampler.sample(row)

    eos = tokenizer.token_to_id("<|im_end|>") if tokenizer is not None else None
    if eos is None:
        eos = 248_046
    generated = 0
    for _ in range(args.sample_len):
        if tokenizer is not None:
            sys.stdout.write(tokenizer.decode([next_id], skip_special_tokens=True))
        else:
            sys.stdout.write(f"{next_id} ")
        sys.stdout.flush()
        generated += 1
        if next_id == eos:
            break
        if generated == args.sample_len:
            break
        out = model.forward(Tensor([next_id]), pos)
        pos += 1
        row = out.squeeze([0]).cast(DType.F32).numpy().tolist()
        next_id = sampler.sample(row)
    print()
    elapsed = time.monotonic() - started
    print(f"{generated} tokens in {elapsed:.2f}s ({generated / elapsed:.2f} tok/s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
