// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! End-to-end Qwen3.8-27B inference: gguf in, text out.
//!
//! [`Qwen`] is the model struct. It owns the weights as typed fields
//! ([`QWeight`] for quantized projections, plain [`Tensor`] for F32)
//! and every kernel compiled, once, in [`Qwen::new`]: one instance per
//! kernel per phase (prefill at the prompt length, decode at one
//! token), plus one quantized-projection kernel per unique
//! (quant format, m, k, n) key shared across all blocks that match it.
//!
//! Each block struct ([`FullBlock`], [`LinearBlock`], [`Ffn`]) has two
//! methods, `prefill` and `step`, and each method body is nothing but
//! direct `forward` calls on the compiled kernels — no helpers inside.
//! One `.forward(` line is one kernel launch, so the launch sequence
//! and count read straight off the method. [`Qwen::prefill`] runs the
//! prompt once and returns logits plus the per-layer caches;
//! [`Qwen::step`] runs one cached token.
//!
//! Weight names (`blk.{n}.attn_qkv.weight`, ...) appear exactly once,
//! in [`Qwen::new`] — the single seam where gguf naming meets kernel
//! contracts. Past construction there are no strings: projections take
//! `&QWeight`, which already carries its format and logical dims, so a
//! missing or renamed tensor fails in `new`, never mid-generate.
//!
//! Every kernel builder named here is fresh — none of the old builders
//! in `lib.rs` are used. Builders that are not written yet are
//! `todo!()`: calling them panics loudly instead of producing a
//! silently wrong result.
//!
//! Facts this file is built on (all verified against
//! `examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf`, not from memory):
//! - 65 blocks (`blk.0` .. `blk.64`). Full attention at `n % 4 == 3`
//!   (3, 7, .., 63) plus `blk.64`; every other block is linear
//!   (gated-delta SSM). Read off the actual tensor list.
//! - Linear block weights: `attn_qkv`, `attn_gate` (= z), `ssm_beta`,
//!   `ssm_alpha`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, `ssm_norm`,
//!   `ssm_out`, `attn_norm`.
//! - Full block weights: `attn_q` ([2*H*D, HIDDEN], q+gate fused),
//!   `attn_k`, `attn_v`, `attn_output`, `attn_norm`, `attn_q_norm`,
//!   `attn_k_norm`.
//! - Every block also carries `ffn_gate`, `ffn_up`, `ffn_down` and
//!   `post_attention_norm`. The FFN wiring (norm -> gate/up -> swiglu ->
//!   down -> residual) and both block-level residuals are COMPOSED after
//!   the standard pre-norm layout — tensor names are verified, the
//!   composition needs an end-to-end golden before it is trusted.
//! - Top level: `token_embd` (Q4_K), `output_norm` (F32), `output`
//!   (Q6_K, [5120, 248320]).
//! - Quant types in this file, by gguf code: 8 = Q8_0, 11 = Q3_K,
//!   12 = Q4_K, 13 = Q5_K, 14 = Q6_K, 23 = IQ4_XS. `load_gguf` hands
//!   every one of them over raw as `[num_blocks, block_bytes]` U8; the
//!   byte counts per block (144/110/176/210/34/136) are documented in
//!   `zyx/src/module.rs`, which is where the format detection below
//!   reads them from. The actual quant math lives inside the kernels.
//!
//! Rules for this forward:
//! - Kernels own their layouts. Fused tensors (e.g. q+gate) are split
//!   inside kernel index math, never by host-side surgery on packed
//!   blocks. Between kernels only whole F32 tensors are passed; there
//!   are no compute ops and no movement ops between kernel calls.
//! - Tilized tensors are never reshaped. (This binary currently targets
//!   CUDA row-major; the TT port must keep every intermediate tilized
//!   end to end and split fused tensors with an explicit kernel.)
//! - Activations are F32 everywhere. Each kernel casts internally if it
//!   wants narrower math; that is the kernel's own business.
//!
//! Tokenizer: the `tokenizers` crate loads a `tokenizer.json` given on
//! the command line (no chat template is applied — raw prompt encode).
//! The file itself is not in this repo; pass its path explicitly.

use qwen3_8_27b::{
    CONV_DIM, DT_RANK, HEADS, HEAD_DIM, HIDDEN, INTERMEDIATE, KD, KV_HEADS, ROT_DIM, VAL_DIM, VD,
    VH,
};
use std::collections::HashMap;
use tokenizers::Tokenizer;
use zyx::kernel::{CompiledKernel, Dev, Kernel};
use zyx::{DType, Tensor, ZyxError};

const N_BLOCKS: i64 = 65;
const VOCAB: i64 = 248320;
const FREQ_BASE: f32 = 10_000_000.0;

fn h_total() -> i64 {
    HEADS * HEAD_DIM
}

fn kv_total() -> i64 {
    KV_HEADS * HEAD_DIM
}

fn qgate_total() -> i64 {
    2 * h_total()
}

/// Full attention at `n % 4 == 3` plus `blk.64` — read off the tensor
/// list in the gguf, not from the interval metadata alone.
fn is_full(n: i64) -> bool {
    n % 4 == 3 || n == 64
}

// ---------------------------------------------------------------------------
// Kernel builders. Every one of these is a name for work that is not
// written yet: the contract in the doc comment is the whole specification,
// the body is `todo!()`. Nothing here runs.
// ---------------------------------------------------------------------------

/// Gather embeddings from the Q4_K embedding table.
/// ids [S] I64 + `token_embd` raw blocks -> [S, HIDDEN] F32.
fn embed_q4k(_dev: Dev, _seq: i64) -> Kernel {
    todo!("embed_q4k: gather rows from Q4_K token_embd raw blocks")
}

/// Plain RMSNorm: x [S, H] F32 + w [H] F32 -> [S, H] F32.
fn rmsnorm(_dev: Dev, _seq: i64, _hidden: i64) -> Kernel {
    todo!("rmsnorm: x * rsqrt(mean(x^2) + eps) * w")
}

/// Quantized GEMM, one builder per quant format. Contract shared by all:
/// a [M, K] F32 + raw packed weight ([K*N/elems, block_bytes] U8, gguf
/// storage order) -> [M, N] F32. Dequantization is fused inside; the
/// caller never sees dequantized weights.
fn qgemm_q8_0(_dev: Dev, _m: i64, _k: i64, _n: i64) -> Kernel {
    todo!("qgemm_q8_0: F32 activations against Q8_0 raw blocks")
}

/// See [`qgemm_q8_0`] for the shared contract.
fn qgemm_q3k(_dev: Dev, _m: i64, _k: i64, _n: i64) -> Kernel {
    todo!("qgemm_q3k: F32 activations against Q3_K raw blocks")
}

/// See [`qgemm_q8_0`] for the shared contract.
fn qgemm_q4k(_dev: Dev, _m: i64, _k: i64, _n: i64) -> Kernel {
    todo!("qgemm_q4k: F32 activations against Q4_K raw blocks")
}

/// See [`qgemm_q8_0`] for the shared contract.
fn qgemm_q5k(_dev: Dev, _m: i64, _k: i64, _n: i64) -> Kernel {
    todo!("qgemm_q5k: F32 activations against Q5_K raw blocks")
}

/// See [`qgemm_q8_0`] for the shared contract.
fn qgemm_q6k(_dev: Dev, _m: i64, _k: i64, _n: i64) -> Kernel {
    todo!("qgemm_q6k: F32 activations against Q6_K raw blocks")
}

/// See [`qgemm_q8_0`] for the shared contract.
fn qgemm_iq4xs(_dev: Dev, _m: i64, _k: i64, _n: i64) -> Kernel {
    todo!("qgemm_iq4xs: F32 activations against IQ4_XS raw blocks")
}

/// Depthwise causal conv1d (kernel 4, left pad 3) + SiLU, prefill form:
/// qkv [S, CONV_DIM] F32 + convw [CONV_DIM, 4] F32 ->
/// (mixed [S, CONV_DIM], hist [3, CONV_DIM]). `hist` is the last 3 qkv
/// rows, seeding decode steps.
fn ssm_conv_silu(_dev: Dev, _seq: i64) -> Kernel {
    todo!("ssm_conv_silu: depthwise conv + silu over the qkv projection")
}

/// Single-token conv step: qkv for one token + convw + history of the 3
/// previous qkv rows [3, CONV_DIM] -> (mixed [1, CONV_DIM], history'
/// with the new row shifted in). Causal depthwise conv needs the 3
/// predecessors; the kernel owns the shift so the caller never slices.
fn ssm_conv_silu_step(_dev: Dev) -> Kernel {
    todo!("ssm_conv_silu_step: conv over 1 new row + 3 history rows")
}

/// Token-recurrent gated delta rule over the whole sequence:
/// mixed [S, CONV_DIM] + b/a [S, DT_RANK] + ssm_a [VH] + ssm_dt [VH] ->
/// (core [VH, S, VD] F32, state [VH, VD, KD] F32). The q/k/v packing
/// inside `mixed` (q at 0, k at KEY_DIM, v at 2*KEY_DIM) is resolved in
/// kernel index math. The state output seeds decode steps.
fn gated_delta_rule(_dev: Dev, _seq: i64) -> Kernel {
    todo!("gated_delta_rule: recurrent state loop, per (head, token)")
}

/// Single-token delta-rule step: mixed/b/a for one token + ssm_a/ssm_dt
/// + incoming state [VH, VD, KD] -> (core [VH, 1, VD], state').
fn gated_delta_rule_step(_dev: Dev) -> Kernel {
    todo!("gated_delta_rule_step: one recurrent update, state in/out")
}

/// Gated output norm: core [VH, S, VD] + z [S, VAL_DIM] + nw [VD] ->
/// [S, VAL_DIM] F32. RMSNorm per (head, token), scaled by silu(z).
fn gated_rmsnorm(_dev: Dev, _seq: i64) -> Kernel {
    todo!("gated_rmsnorm: per-head rmsnorm times silu gate")
}

/// Per-head Q/K norm with weights. qgate [S, 2*H*D] (q and gate halves
/// split inside) + k [S, KV*D] + qw/kw [HEAD_DIM] -> (q [S, H*D],
/// k [S, KV*D], gate [S, H*D]).
fn qk_norm(_dev: Dev, _seq: i64, _heads: i64, _kv_heads: i64, _head_dim: i64) -> Kernel {
    todo!("qk_norm: split fused q/gate, per-head rmsnorm with weights")
}

/// Partial RoPE (leading ROT_DIM dims, GPT-NeoX rotate_half semantics):
/// q [S, H*D] + k [S, KV*D] + cos/sin [S, ROT_DIM] -> (q, k).
fn rope_apply(
    _dev: Dev,
    _seq: i64,
    _heads: i64,
    _kv_heads: i64,
    _head_dim: i64,
    _rot_dim: i64,
) -> Kernel {
    todo!("rope_apply: partial rotary embedding with given tables")
}

/// Causal GQA attention with sigmoid output gate, prefill form:
/// q [S, H*D] + k/v [S, KV*D] + gate [S, H*D] ->
/// (out [S, H*D] F32, k_cache [S, KV*D], v_cache [S, KV*D]).
/// The caches hold every token's K/V row-major; decode steps thread
/// them through, never reading them on the host.
fn full_attention(_dev: Dev, _seq: i64, _heads: i64, _kv_heads: i64, _head_dim: i64) -> Kernel {
    todo!("full_attention: causal softmax attention times sigmoid gate")
}

/// Single-token attention step: q/k/v/gate for one token + incoming
/// caches [T, KV*D] -> (out [1, H*D], k_cache' [T+1, KV*D],
/// v_cache' [T+1, KV*D]) with the new K/V appended inside.
fn full_attention_step(_dev: Dev, _heads: i64, _kv_heads: i64, _head_dim: i64) -> Kernel {
    todo!("full_attention_step: append K/V, attend over full cache")
}

/// SwiGLU elementwise: gate/up [S, INTER] F32 -> mid [S, INTER].
fn swiglu(_dev: Dev, _seq: i64, _inter: i64) -> Kernel {
    todo!("swiglu: silu(gate) * up")
}

/// Elementwise add: a/b [S, H] F32 -> [S, H].
fn residual_add(_dev: Dev, _seq: i64, _hidden: i64) -> Kernel {
    todo!("residual_add: a + b")
}

// ---------------------------------------------------------------------------
// Real code below. Model struct, dispatch, tables, generate loop.
// ---------------------------------------------------------------------------

/// Key for the shared projection-kernel maps: gguf quant code plus the
/// logical dims (m rows in, k width, n rows out).
type ProjKey = (u8, i64, i64, i64);

/// gguf quant code of a raw tensor, read back out of its shape. Block
/// sizes per `zyx/src/module.rs`: Q8_0 34, Q3_K 110, Q4_K 144,
/// Q5_K 176, Q6_K 210, IQ4_XS 136.
fn qtype_of(w_raw: &Tensor) -> u8 {
    debug_assert_eq!(w_raw.dtype(), DType::U8);
    let sh: Vec<i64> = w_raw.shape().iter().map(|t| t.item::<i64>()).collect();
    debug_assert_eq!(sh.len(), 2);
    match sh[1] {
        34 => 8,
        110 => 11,
        144 => 12,
        176 => 13,
        210 => 14,
        136 => 23,
        x => panic!("qtype_of: unrecognized quant block size {x} bytes"),
    }
}

fn qgemm_builder(qtype: u8) -> fn(Dev, i64, i64, i64) -> Kernel {
    match qtype {
        8 => qgemm_q8_0,
        11 => qgemm_q3k,
        12 => qgemm_q4k,
        13 => qgemm_q5k,
        14 => qgemm_q6k,
        23 => qgemm_iq4xs,
        x => panic!("qgemm_builder: unknown quant code {x}"),
    }
}

/// One quantized projection weight on device: the raw packed tensor
/// plus the format and logical dims resolved once at load. `rows`/`cols`
/// are storage-order (dim0 contiguous); they cannot be read back out of
/// the tensor (which only knows `[num_blocks, block_bytes]`), and the
/// same packed bytes may serve kernels wanting different views — so the
/// interpretation lives here, next to the name it was resolved from.
struct QWeight {
    t: Tensor,
    qtype: u8,
    rows: i64,
    cols: i64,
}

/// FFN weights shared by both layer kinds.
struct Ffn {
    gate: QWeight,
    up: QWeight,
    down: QWeight,
    post_norm: Tensor,
}

/// Full-attention block weights.
struct FullBlock {
    norm: Tensor,
    q: QWeight,
    k: QWeight,
    v: QWeight,
    o: QWeight,
    q_norm_w: Tensor,
    k_norm_w: Tensor,
    ffn: Ffn,
}

/// Linear (SSM) block weights.
struct LinearBlock {
    norm: Tensor,
    qkv: QWeight,
    gate: QWeight,
    beta: QWeight,
    alpha: QWeight,
    convw: Tensor,
    ssm_a: Tensor,
    ssm_dt: Tensor,
    ssm_norm_w: Tensor,
    ssm_out: QWeight,
    ffn: Ffn,
}

enum Block {
    Full(FullBlock),
    Linear(LinearBlock),
}

/// Per-layer decode state threaded through generate steps.
enum LayerCache {
    Full { k: Tensor, v: Tensor },
    Linear { state: Tensor, hist: Tensor },
}

/// Take a tensor out of the host map by gguf name, or fail in `new`
/// with the missing name — never mid-generate.
fn take(host: &mut HashMap<String, Tensor>, name: String) -> Tensor {
    host.remove(&name)
        .unwrap_or_else(|| panic!("gguf is missing tensor {name}"))
}

/// Quantized weight: resolve format + logical dims once, move to device.
fn qw(
    host: &mut HashMap<String, Tensor>,
    dev: Dev,
    name: String,
    rows: i64,
    cols: i64,
) -> Result<QWeight, ZyxError> {
    let t = take(host, name);
    let qtype = qtype_of(&t);
    Ok(QWeight {
        t: t.to(dev)?,
        qtype,
        rows,
        cols,
    })
}

/// F32 weight: check the dtype, move to device.
fn f32w(host: &mut HashMap<String, Tensor>, dev: Dev, name: String) -> Result<Tensor, ZyxError> {
    let t = take(host, name);
    debug_assert_eq!(t.dtype(), DType::F32);
    t.to(dev)
}

/// The model: weights as typed fields plus every kernel compiled once.
struct Qwen {
    dev: Dev,
    seq: i64,
    token_embd: QWeight,
    output_norm: Tensor,
    output: QWeight,
    blocks: Vec<Block>,
    // Prefill kernels (seq = prompt length).
    embed_pf: CompiledKernel,
    norm_pf: CompiledKernel,
    swiglu_pf: CompiledKernel,
    qknorm_pf: CompiledKernel,
    rope_pf: CompiledKernel,
    attn_pf: CompiledKernel,
    conv_pf: CompiledKernel,
    delta_pf: CompiledKernel,
    gated_pf: CompiledKernel,
    resid_pf: CompiledKernel,
    // Decode kernels (seq = 1).
    embed_1: CompiledKernel,
    norm_1: CompiledKernel,
    swiglu_1: CompiledKernel,
    qknorm_1: CompiledKernel,
    rope_1: CompiledKernel,
    attn_1: CompiledKernel,
    conv_1: CompiledKernel,
    delta_1: CompiledKernel,
    gated_1: CompiledKernel,
    resid_1: CompiledKernel,
    // Shared projection kernels. Key layout, constructed inline at
    // every projection site: (weight format code, m rows in, k width,
    // n rows out) = (w.qtype, m, w.cols, w.rows). The layout is stated
    // once here; every site repeats it verbatim.
    projs_pf: HashMap<ProjKey, CompiledKernel>,
    projs_1: HashMap<ProjKey, CompiledKernel>,
}

impl FullBlock {
    /// Prefill: h [S, HIDDEN] -> (residual [S, HIDDEN], k/v caches).
    /// Op order mirrors the verified sequence; residual placement is
    /// COMPOSED. 8 kernel launches.
    fn prefill(&self, q: &Qwen, h: &Tensor) -> Result<(Tensor, Tensor, Tensor), ZyxError> {
        let m = q.seq;
        let ht = h_total();
        let kvt = kv_total();
        let xn = q
            .norm_pf
            .forward(&[h, &self.norm], vec![vec![m, HIDDEN]])?
            .remove(0);
        let qg = q.projs_pf[&(self.q.qtype, m, self.q.cols, self.q.rows)]
            .forward1(&[&xn, &self.q.t], vec![m, self.q.rows])?;
        let k = q.projs_pf[&(self.k.qtype, m, self.k.cols, self.k.rows)]
            .forward1(&[&xn, &self.k.t], vec![m, self.k.rows])?;
        let v = q.projs_pf[&(self.v.qtype, m, self.v.cols, self.v.rows)]
            .forward1(&[&xn, &self.v.t], vec![m, self.v.rows])?;
        let mut outs = q.qknorm_pf.forward(
            &[&qg, &k, &self.q_norm_w, &self.k_norm_w],
            vec![vec![m, ht], vec![m, kvt], vec![m, ht]],
        )?;
        let gate = outs.pop().expect("qk_norm third output");
        let kk = outs.pop().expect("qk_norm second output");
        let qq = outs.pop().expect("qk_norm first output");
        let (cos, sin) = rope_tables(m, 0)?;
        let (cos, sin) = (cos.to(q.dev)?, sin.to(q.dev)?);
        let mut rqs = q
            .rope_pf
            .forward(&[&qq, &kk, &cos, &sin], vec![vec![m, ht], vec![m, kvt]])?;
        let kr = rqs.pop().expect("rope second output");
        let qr = rqs.pop().expect("rope first output");
        let mut aouts = q.attn_pf.forward(
            &[&qr, &kr, &v, &gate],
            vec![vec![m, ht], vec![m, kvt], vec![m, kvt]],
        )?;
        let vc = aouts.pop().expect("full_attention third output");
        let kc = aouts.pop().expect("full_attention second output");
        let att = aouts.pop().expect("full_attention first output");
        let o = q.projs_pf[&(self.o.qtype, m, self.o.cols, self.o.rows)]
            .forward1(&[&att, &self.o.t], vec![m, self.o.rows])?;
        let h_out = q
            .resid_pf
            .forward(&[h, &o], vec![vec![m, HIDDEN]])?
            .remove(0);
        Ok((h_out, kc, vc))
    }

    /// Decode step: h [1, HIDDEN] + caches [T, KV*D] at RoPE offset
    /// `pos` -> (residual [1, HIDDEN], caches [T+1, KV*D]).
    /// 8 kernel launches.
    fn step(
        &self,
        q: &Qwen,
        h: &Tensor,
        k0: Tensor,
        v0: Tensor,
        pos: i64,
    ) -> Result<(Tensor, Tensor, Tensor), ZyxError> {
        let ht = h_total();
        let kvt = kv_total();
        let xn = q
            .norm_1
            .forward(&[h, &self.norm], vec![vec![1, HIDDEN]])?
            .remove(0);
        let qg = q.projs_1[&(self.q.qtype, 1, self.q.cols, self.q.rows)]
            .forward1(&[&xn, &self.q.t], vec![1, self.q.rows])?;
        let k1 = q.projs_1[&(self.k.qtype, 1, self.k.cols, self.k.rows)]
            .forward1(&[&xn, &self.k.t], vec![1, self.k.rows])?;
        let v1 = q.projs_1[&(self.v.qtype, 1, self.v.cols, self.v.rows)]
            .forward1(&[&xn, &self.v.t], vec![1, self.v.rows])?;
        let mut outs = q.qknorm_1.forward(
            &[&qg, &k1, &self.q_norm_w, &self.k_norm_w],
            vec![vec![1, ht], vec![1, kvt], vec![1, ht]],
        )?;
        let gate = outs.pop().expect("qk_norm third output");
        let kk = outs.pop().expect("qk_norm second output");
        let qq = outs.pop().expect("qk_norm first output");
        let (cos, sin) = rope_tables(1, pos)?;
        let (cos, sin) = (cos.to(q.dev)?, sin.to(q.dev)?);
        let mut rqs = q
            .rope_1
            .forward(&[&qq, &kk, &cos, &sin], vec![vec![1, ht], vec![1, kvt]])?;
        let kr = rqs.pop().expect("rope second output");
        let qr = rqs.pop().expect("rope first output");
        let mut aouts = q.attn_1.forward(
            &[&qr, &kr, &v1, &gate, &k0, &v0],
            vec![vec![1, ht], vec![pos + 1, kvt], vec![pos + 1, kvt]],
        )?;
        let vc = aouts.pop().expect("full_attention_step third output");
        let kc = aouts.pop().expect("full_attention_step second output");
        let att = aouts.pop().expect("full_attention_step first output");
        let o = q.projs_1[&(self.o.qtype, 1, self.o.cols, self.o.rows)]
            .forward1(&[&att, &self.o.t], vec![1, self.o.rows])?;
        let h_out = q
            .resid_1
            .forward(&[h, &o], vec![vec![1, HIDDEN]])?
            .remove(0);
        Ok((h_out, kc, vc))
    }
}

impl LinearBlock {
    /// Prefill: h [S, HIDDEN] -> (residual, state, conv history).
    /// Op order mirrors the verified sequence; residual placement is
    /// COMPOSED. 9 kernel launches.
    fn prefill(&self, q: &Qwen, h: &Tensor) -> Result<(Tensor, Tensor, Tensor), ZyxError> {
        let m = q.seq;
        let xn = q
            .norm_pf
            .forward(&[h, &self.norm], vec![vec![m, HIDDEN]])?
            .remove(0);
        let qkv = q.projs_pf[&(self.qkv.qtype, m, self.qkv.cols, self.qkv.rows)]
            .forward1(&[&xn, &self.qkv.t], vec![m, self.qkv.rows])?;
        let z = q.projs_pf[&(self.gate.qtype, m, self.gate.cols, self.gate.rows)]
            .forward1(&[&xn, &self.gate.t], vec![m, self.gate.rows])?;
        let bb = q.projs_pf[&(self.beta.qtype, m, self.beta.cols, self.beta.rows)]
            .forward1(&[&xn, &self.beta.t], vec![m, self.beta.rows])?;
        let aa = q.projs_pf[&(self.alpha.qtype, m, self.alpha.cols, self.alpha.rows)]
            .forward1(&[&xn, &self.alpha.t], vec![m, self.alpha.rows])?;
        let mut mouts = q.conv_pf.forward(
            &[&qkv, &self.convw],
            vec![vec![m, CONV_DIM], vec![3, CONV_DIM]],
        )?;
        let hist = mouts.pop().expect("ssm_conv_silu second output");
        let mixed = mouts.pop().expect("ssm_conv_silu first output");
        let mut couts = q.delta_pf.forward(
            &[&mixed, &bb, &aa, &self.ssm_a, &self.ssm_dt],
            vec![vec![VH, m, VD], vec![VH, VD, KD]],
        )?;
        let state = couts.pop().expect("gated_delta_rule second output");
        let core = couts.pop().expect("gated_delta_rule first output");
        let n = q
            .gated_pf
            .forward(&[&core, &z, &self.ssm_norm_w], vec![vec![m, VAL_DIM]])?
            .remove(0);
        let o = q.projs_pf[&(self.ssm_out.qtype, m, self.ssm_out.cols, self.ssm_out.rows)]
            .forward1(&[&n, &self.ssm_out.t], vec![m, self.ssm_out.rows])?;
        let h_out = q
            .resid_pf
            .forward(&[h, &o], vec![vec![m, HIDDEN]])?
            .remove(0);
        Ok((h_out, state, hist))
    }

    /// Decode step: h [1, HIDDEN] + (state, hist) ->
    /// (residual, state', hist'). 9 kernel launches.
    fn step(
        &self,
        q: &Qwen,
        h: &Tensor,
        state: Tensor,
        hist: Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), ZyxError> {
        let xn = q
            .norm_1
            .forward(&[h, &self.norm], vec![vec![1, HIDDEN]])?
            .remove(0);
        let qkv = q.projs_1[&(self.qkv.qtype, 1, self.qkv.cols, self.qkv.rows)]
            .forward1(&[&xn, &self.qkv.t], vec![1, self.qkv.rows])?;
        let z = q.projs_1[&(self.gate.qtype, 1, self.gate.cols, self.gate.rows)]
            .forward1(&[&xn, &self.gate.t], vec![1, self.gate.rows])?;
        let bb = q.projs_1[&(self.beta.qtype, 1, self.beta.cols, self.beta.rows)]
            .forward1(&[&xn, &self.beta.t], vec![1, self.beta.rows])?;
        let aa = q.projs_1[&(self.alpha.qtype, 1, self.alpha.cols, self.alpha.rows)]
            .forward1(&[&xn, &self.alpha.t], vec![1, self.alpha.rows])?;
        let mut mouts = q.conv_1.forward(
            &[&qkv, &self.convw, &hist],
            vec![vec![1, CONV_DIM], vec![3, CONV_DIM]],
        )?;
        let hist = mouts.pop().expect("ssm_conv_silu_step second output");
        let mixed = mouts.pop().expect("ssm_conv_silu_step first output");
        let mut couts = q.delta_1.forward(
            &[&mixed, &bb, &aa, &self.ssm_a, &self.ssm_dt, &state],
            vec![vec![VH, 1, VD], vec![VH, VD, KD]],
        )?;
        let state = couts.pop().expect("gated_delta_rule_step second output");
        let core = couts.pop().expect("gated_delta_rule_step first output");
        let n = q
            .gated_1
            .forward1(&[&core, &z, &self.ssm_norm_w], vec![vec![1, VAL_DIM]])?;
        let o = q.projs_1[&(self.ssm_out.qtype, 1, self.ssm_out.cols, self.ssm_out.rows)]
            .forward1(&[&n, &self.ssm_out.t], vec![1, self.ssm_out.rows])?;
        let h_out = q.resid_1.forward1(&[h, &o], vec![vec![1, HIDDEN]])?;
        Ok((h_out, state, hist))
    }
}

impl Ffn {
    /// Prefill: h [S, HIDDEN] -> residual [S, HIDDEN]. COMPOSED after
    /// the standard pre-norm FFN. 5 kernel launches.
    fn prefill(&self, q: &Qwen, h: &Tensor) -> Result<Tensor, ZyxError> {
        let m = q.seq;
        let xn = q
            .norm_pf
            .forward(&[h, &self.post_norm], vec![vec![m, HIDDEN]])?
            .remove(0);
        let g = q.projs_pf[&(self.gate.qtype, m, self.gate.cols, self.gate.rows)]
            .forward1(&[&xn, &self.gate.t], vec![m, self.gate.rows])?;
        let u = q.projs_pf[&(self.up.qtype, m, self.up.cols, self.up.rows)]
            .forward1(&[&xn, &self.up.t], vec![m, self.up.rows])?;
        let mid = q
            .swiglu_pf
            .forward(&[&g, &u], vec![vec![m, INTERMEDIATE]])?
            .remove(0);
        let d = q.projs_pf[&(self.down.qtype, m, self.down.cols, self.down.rows)]
            .forward1(&[&mid, &self.down.t], vec![m, self.down.rows])?;
        Ok(q.resid_pf
            .forward(&[h, &d], vec![vec![m, HIDDEN]])?
            .remove(0))
    }

    /// Decode step: h [1, HIDDEN] -> residual [1, HIDDEN].
    /// 5 kernel launches.
    fn step(&self, q: &Qwen, h: &Tensor) -> Result<Tensor, ZyxError> {
        let xn = q
            .norm_1
            .forward(&[h, &self.post_norm], vec![vec![1, HIDDEN]])?
            .remove(0);
        let g = q.projs_1[&(self.gate.qtype, 1, self.gate.cols, self.gate.rows)]
            .forward1(&[&xn, &self.gate.t], vec![1, self.gate.rows])?;
        let u = q.projs_1[&(self.up.qtype, 1, self.up.cols, self.up.rows)]
            .forward1(&[&xn, &self.up.t], vec![1, self.up.rows])?;
        let mid = q
            .swiglu_1
            .forward(&[&g, &u], vec![vec![1, INTERMEDIATE]])?
            .remove(0);
        let d = q.projs_1[&(self.down.qtype, 1, self.down.cols, self.down.rows)]
            .forward1(&[&mid, &self.down.t], vec![1, self.down.rows])?;
        Ok(q.resid_1
            .forward(&[h, &d], vec![vec![1, HIDDEN]])?
            .remove(0))
    }
}

impl Qwen {
    /// Load weights into typed fields, move them to device, compile the
    /// whole kernel set once for this prompt length. The gguf names
    /// appear only here. Startup cost is one compile per unique kernel
    /// (~50 total); afterwards prefill and every decode step only
    /// launch.
    fn new(dev: Dev, gguf: &str, seq: i64) -> Result<Self, ZyxError> {
        let (_meta, mut host) = Tensor::load_gguf(gguf)?;
        let token_embd = qw(
            &mut host,
            dev,
            "token_embd.weight".to_string(),
            VOCAB,
            HIDDEN,
        )?;
        let output_norm = f32w(&mut host, dev, "output_norm.weight".to_string())?;
        let output = qw(&mut host, dev, "output.weight".to_string(), VOCAB, HIDDEN)?;
        let mut blocks = Vec::with_capacity(N_BLOCKS as usize);
        for b in 0..N_BLOCKS {
            let pre = format!("blk.{b}.");
            let ffn = Ffn {
                gate: qw(
                    &mut host,
                    dev,
                    pre.clone() + "ffn_gate.weight",
                    INTERMEDIATE,
                    HIDDEN,
                )?,
                up: qw(
                    &mut host,
                    dev,
                    pre.clone() + "ffn_up.weight",
                    INTERMEDIATE,
                    HIDDEN,
                )?,
                down: qw(
                    &mut host,
                    dev,
                    pre.clone() + "ffn_down.weight",
                    HIDDEN,
                    INTERMEDIATE,
                )?,
                post_norm: f32w(&mut host, dev, pre.clone() + "post_attention_norm.weight")?,
            };
            if is_full(b) {
                blocks.push(Block::Full(FullBlock {
                    norm: f32w(&mut host, dev, pre.clone() + "attn_norm.weight")?,
                    q: qw(
                        &mut host,
                        dev,
                        pre.clone() + "attn_q.weight",
                        qgate_total(),
                        HIDDEN,
                    )?,
                    k: qw(
                        &mut host,
                        dev,
                        pre.clone() + "attn_k.weight",
                        kv_total(),
                        HIDDEN,
                    )?,
                    v: qw(
                        &mut host,
                        dev,
                        pre.clone() + "attn_v.weight",
                        kv_total(),
                        HIDDEN,
                    )?,
                    o: qw(
                        &mut host,
                        dev,
                        pre.clone() + "attn_output.weight",
                        HIDDEN,
                        h_total(),
                    )?,
                    q_norm_w: f32w(&mut host, dev, pre.clone() + "attn_q_norm.weight")?,
                    k_norm_w: f32w(&mut host, dev, pre.clone() + "attn_k_norm.weight")?,
                    ffn,
                }));
            } else {
                blocks.push(Block::Linear(LinearBlock {
                    norm: f32w(&mut host, dev, pre.clone() + "attn_norm.weight")?,
                    qkv: qw(
                        &mut host,
                        dev,
                        pre.clone() + "attn_qkv.weight",
                        CONV_DIM,
                        HIDDEN,
                    )?,
                    gate: qw(
                        &mut host,
                        dev,
                        pre.clone() + "attn_gate.weight",
                        VAL_DIM,
                        HIDDEN,
                    )?,
                    beta: qw(
                        &mut host,
                        dev,
                        pre.clone() + "ssm_beta.weight",
                        DT_RANK,
                        HIDDEN,
                    )?,
                    alpha: qw(
                        &mut host,
                        dev,
                        pre.clone() + "ssm_alpha.weight",
                        DT_RANK,
                        HIDDEN,
                    )?,
                    convw: f32w(&mut host, dev, pre.clone() + "ssm_conv1d.weight")?,
                    ssm_a: f32w(&mut host, dev, pre.clone() + "ssm_a")?,
                    ssm_dt: f32w(&mut host, dev, pre.clone() + "ssm_dt.bias")?,
                    ssm_norm_w: f32w(&mut host, dev, pre.clone() + "ssm_norm.weight")?,
                    ssm_out: qw(
                        &mut host,
                        dev,
                        pre.clone() + "ssm_out.weight",
                        HIDDEN,
                        VAL_DIM,
                    )?,
                    ffn,
                }));
            }
        }
        let mut projs_pf = HashMap::new();
        let mut projs_1 = HashMap::new();
        let mut all: Vec<&QWeight> = vec![&output];
        for block in &blocks {
            match block {
                Block::Full(fw) => {
                    all.extend([
                        &fw.q,
                        &fw.k,
                        &fw.v,
                        &fw.o,
                        &fw.ffn.gate,
                        &fw.ffn.up,
                        &fw.ffn.down,
                    ]);
                }
                Block::Linear(lw) => {
                    all.extend([
                        &lw.qkv,
                        &lw.gate,
                        &lw.beta,
                        &lw.alpha,
                        &lw.ssm_out,
                        &lw.ffn.gate,
                        &lw.ffn.up,
                        &lw.ffn.down,
                    ]);
                }
            }
        }
        for w in all {
            let build = qgemm_builder(w.qtype);
            projs_pf
                .entry((w.qtype, seq, w.cols, w.rows))
                .or_insert(build(dev, seq, w.cols, w.rows).compile()?);
            projs_1
                .entry((w.qtype, 1, w.cols, w.rows))
                .or_insert(build(dev, 1, w.cols, w.rows).compile()?);
        }
        Ok(Self {
            dev,
            seq,
            token_embd,
            output_norm,
            output,
            blocks,
            embed_pf: embed_q4k(dev, seq).compile()?,
            norm_pf: rmsnorm(dev, seq, HIDDEN).compile()?,
            swiglu_pf: swiglu(dev, seq, INTERMEDIATE).compile()?,
            qknorm_pf: qk_norm(dev, seq, HEADS, KV_HEADS, HEAD_DIM).compile()?,
            rope_pf: rope_apply(dev, seq, HEADS, KV_HEADS, HEAD_DIM, ROT_DIM).compile()?,
            attn_pf: full_attention(dev, seq, HEADS, KV_HEADS, HEAD_DIM).compile()?,
            conv_pf: ssm_conv_silu(dev, seq).compile()?,
            delta_pf: gated_delta_rule(dev, seq).compile()?,
            gated_pf: gated_rmsnorm(dev, seq).compile()?,
            resid_pf: residual_add(dev, seq, HIDDEN).compile()?,
            embed_1: embed_q4k(dev, 1).compile()?,
            norm_1: rmsnorm(dev, 1, HIDDEN).compile()?,
            swiglu_1: swiglu(dev, 1, INTERMEDIATE).compile()?,
            qknorm_1: qk_norm(dev, 1, HEADS, KV_HEADS, HEAD_DIM).compile()?,
            rope_1: rope_apply(dev, 1, HEADS, KV_HEADS, HEAD_DIM, ROT_DIM).compile()?,
            attn_1: full_attention_step(dev, HEADS, KV_HEADS, HEAD_DIM).compile()?,
            conv_1: ssm_conv_silu_step(dev).compile()?,
            delta_1: gated_delta_rule_step(dev).compile()?,
            gated_1: gated_rmsnorm(dev, 1).compile()?,
            resid_1: residual_add(dev, 1, HIDDEN).compile()?,
            projs_pf,
            projs_1,
        })
    }

    /// Prefill: ids -> (logits [S, VOCAB], per-layer caches).
    fn prefill(&self, ids: &[i64]) -> Result<(Tensor, Vec<LayerCache>), ZyxError> {
        debug_assert_eq!(ids.len() as i64, self.seq);
        let idt = Tensor::from_vec(ids.to_vec(), [self.seq])?.to(self.dev)?;
        let mut h = self
            .embed_pf
            .forward1(&[&idt, &self.token_embd.t], vec![self.seq, HIDDEN])?;
        let mut caches = Vec::with_capacity(N_BLOCKS as usize);
        for block in &self.blocks {
            match block {
                Block::Full(fw) => {
                    let (hh, k, v) = fw.prefill(self, &h)?;
                    h = fw.ffn.prefill(self, &hh)?;
                    caches.push(LayerCache::Full { k, v });
                }
                Block::Linear(lw) => {
                    let (hh, state, hist) = lw.prefill(self, &h)?;
                    h = lw.ffn.prefill(self, &hh)?;
                    caches.push(LayerCache::Linear { state, hist });
                }
            }
        }
        let hn = self
            .norm_pf
            .forward1(&[&h, &self.output_norm], vec![self.seq, HIDDEN])?;
        let logits = self.projs_pf[&(
            self.output.qtype,
            self.seq,
            self.output.cols,
            self.output.rows,
        )]
            .forward1(&[&hn, &self.output.t], vec![self.seq, self.output.rows])?;
        Ok((logits, caches))
    }

    /// Single decode step: last token id + caches at length `pos`
    /// (tokens processed so far) -> (logits [1, VOCAB], new caches).
    fn step(
        &self,
        id: i64,
        pos: i64,
        caches: Vec<LayerCache>,
    ) -> Result<(Tensor, Vec<LayerCache>), ZyxError> {
        debug_assert!(pos > 0);
        let idt = Tensor::from_vec(vec![id], [1])?.to(self.dev)?;
        let mut h = self
            .embed_1
            .forward1(&[&idt, &self.token_embd.t], vec![1, HIDDEN])?;
        let mut out = Vec::with_capacity(N_BLOCKS as usize);
        for (block, cache) in self.blocks.iter().zip(caches) {
            match (block, cache) {
                (Block::Full(fw), LayerCache::Full { k, v }) => {
                    let (hh, k2, v2) = fw.step(self, &h, k, v, pos)?;
                    h = fw.ffn.step(self, &hh)?;
                    out.push(LayerCache::Full { k: k2, v: v2 });
                }
                (Block::Linear(lw), LayerCache::Linear { state, hist }) => {
                    let (hh, s2, h2) = lw.step(self, &h, state, hist)?;
                    h = lw.ffn.step(self, &hh)?;
                    out.push(LayerCache::Linear {
                        state: s2,
                        hist: h2,
                    });
                }
                _ => panic!("step: cache kind does not match block kind"),
            }
        }
        let hn = self
            .norm_1
            .forward1(&[&h, &self.output_norm], vec![1, HIDDEN])?;
        let logits = self.projs_1[&(self.output.qtype, 1, self.output.cols, self.output.rows)]
            .forward1(&[&hn, &self.output.t], vec![1, self.output.rows])?;
        Ok((logits, out))
    }
}

/// RoPE cos/sin tables [S, ROT_DIM] F32 for positions
/// `pos_offset..pos_offset + seq`.
///
/// UNVERIFIED: standard GPT-NeoX tables from `freq_base`
/// (pair j shares angle `pos * base^(-2j/ROT_DIM)` at dims j, j+32).
/// The gguf carries `rope.dimension_sections = [11, 11, 10, 0]`
/// (mRoPE sections), so the real model may map frequencies to dims
/// differently. Check the tables against the reference attention before
/// trusting any text this produces.
fn rope_tables(seq: i64, pos_offset: i64) -> Result<(Tensor, Tensor), ZyxError> {
    let r = ROT_DIM as usize;
    let s = seq as usize;
    let mut cos = vec![0.0f32; s * r];
    let mut sin = vec![0.0f32; s * r];
    for i in 0..s {
        let p = pos_offset + i as i64;
        for j in 0..r / 2 {
            let angle = p as f32 * FREQ_BASE.powf(-2.0 * j as f32 / ROT_DIM as f32);
            let (sn, cs) = angle.sin_cos();
            cos[i * r + j] = cs;
            cos[i * r + j + r / 2] = cs;
            sin[i * r + j] = sn;
            sin[i * r + j + r / 2] = sn;
        }
    }
    let c = Tensor::from_vec(cos, [seq, ROT_DIM])?;
    let sn = Tensor::from_vec(sin, [seq, ROT_DIM])?;
    Ok((c, sn))
}

/// Greedy next-token id from the last logit row.
fn greedy_next(logits: &Tensor, seq: i64) -> Result<i64, ZyxError> {
    let v: Vec<f32> = logits.to_vec()?;
    debug_assert_eq!(v.len(), (seq * VOCAB) as usize);
    let row = &v[((seq - 1) * VOCAB) as usize..];
    let mut best = 0i64;
    for (i, &x) in row.iter().enumerate() {
        if x > row[best as usize] {
            best = i as i64;
        }
    }
    Ok(best)
}

fn tok_err(e: impl std::fmt::Display) -> ZyxError {
    ZyxError::parse_error(e.to_string().into())
}

fn main() -> Result<(), ZyxError> {
    // CLI: <gguf> [--tokenizer tok.json] [--max-new N] <prompt ...>.
    // With --tokenizer the prompt is one string; without it, integer ids.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut gguf: Option<String> = None;
    let mut tok_path: Option<String> = None;
    let mut max_new = 8i64;
    let mut prompt: Vec<String> = Vec::new();
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        if a == "--tokenizer" {
            tok_path = Some(it.next().expect("--tokenizer needs a path"));
        } else if a == "--max-new" {
            max_new = it
                .next()
                .expect("--max-new needs a number")
                .parse()
                .expect("max-new must be an integer");
        } else if gguf.is_none() {
            gguf = Some(a);
        } else {
            prompt.push(a);
        }
    }
    let gguf = gguf
        .expect("usage: qwen3-8-27b <model.gguf> [--tokenizer tok.json] [--max-new N] <prompt>");
    if prompt.is_empty() {
        panic!("no prompt given");
    }

    // Tokenize (string mode) or parse ids (numbers mode).
    let tok = tok_path
        .map(|p| Tokenizer::from_file(p).map_err(tok_err))
        .transpose()?;
    let mut ids: Vec<i64> = if let Some(t) = &tok {
        // No chat template applied: raw prompt encode, special tokens on.
        t.encode(prompt.join(" "), true)
            .map_err(tok_err)?
            .get_ids()
            .iter()
            .map(|&x| x as i64)
            .collect()
    } else {
        prompt
            .iter()
            .map(|a| a.parse().expect("prompt args must be integer token ids"))
            .collect()
    };
    if ids.is_empty() {
        panic!("tokenizer produced no ids");
    }

    let dev = Dev::Cuda(0);
    let model = Qwen::new(dev, &gguf, ids.len() as i64)?;

    // Prefill once, then one cached step per token.
    let (logits, mut caches) = model.prefill(&ids)?;
    let mut next = greedy_next(&logits, ids.len() as i64)?;
    ids.push(next);
    for _ in 1..max_new {
        let pos = ids.len() as i64 - 1;
        let (l, c2) = model.step(next, pos, caches)?;
        caches = c2;
        next = greedy_next(&l, 1)?;
        ids.push(next);
    }

    // Print ids always; decoded text when a tokenizer is loaded.
    print!("ids:");
    for id in &ids {
        print!(" {id}");
    }
    println!();
    if let Some(t) = &tok {
        let u: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        println!("{}", t.decode(&u, true).map_err(tok_err)?);
    }
    Ok(())
}
