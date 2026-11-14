// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Ternary-Bonsai-2-27B (`qwen35` trunk, PQ2_0 weights) inference.

pub mod config;
pub mod model;
pub mod prism;
pub mod weights;

pub use model::Bonsai;
