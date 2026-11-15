// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Ternary-Bonsai-2-27B (`qwen35` trunk) shapes, shared by every kernel in
//! this group.
//!
//! The numbers are the GGUF metadata's: hidden 5120, ffn 17408, 24 heads / 4 kv
//! heads, head_dim 256, vocab 248320, 64 blocks of which every fourth is a full
//! attention block (the other 48 run the gated delta rule), ssm state 128 with
//! 16 key / 48 value heads, conv kernel 4, prism rotation block 1024. Decode is
//! batch 1 seq 1; prefill is batch 1 at the port's padded prefill width.

pub const HIDDEN: i64 = 5120;
pub const FFN: i64 = 17408;
pub const HEADS: i64 = 24;
pub const KV_HEADS: i64 = 4;
pub const HEAD_DIM: i64 = 256;
pub const VOCAB: i64 = 248_320;
/// Padded prefill width (`max_prefill`), a multiple of [`CHUNK`].
pub const SEQ: i64 = 128;

pub const SSM_STATE: i64 = 128;
pub const SSM_K_HEADS: i64 = 16;
pub const SSM_V_HEADS: i64 = 48;
pub const SSM_V_HEAD_DIM: i64 = SSM_STATE;
/// `qwen35.ssm.group_count * ssm.state_size`: the linear block's key width.
pub const KEY_DIM: i64 = SSM_STATE * SSM_K_HEADS;
/// `qwen35.ssm.inner_size`: the linear block's value width.
pub const VALUE_DIM: i64 = SSM_STATE * SSM_V_HEADS;
/// `2 * key_dim + value_dim`: the depthwise conv's channel count.
pub const CONV_DIM: i64 = 2 * KEY_DIM + VALUE_DIM;
pub const CONV_KERNEL: i64 = 4;
/// Gated-delta-rule chunk length used by the chunked prefill.
pub const CHUNK: i64 = 64;

/// Elements covered by one PQ2_0 super-block.
pub const PQ2_BLOCK: i64 = 128;
/// Bytes of one PQ2_0 super-block: F16 scale + 32 bytes of 2-bit codes.
pub const PQ2_ROW: i64 = 34;
