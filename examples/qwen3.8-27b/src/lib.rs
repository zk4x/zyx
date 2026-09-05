// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Qwen3.8-27B inference example (UD-Q4_K_XL).
//!
//! Per-op verification against torch goldens: each op in `tests/` has a
//! `<op>.rs` test and a `<op>_ref.py` golden-dump script. Reference side
//! runs on CUDA, tiled kernels run on Tenstorrent.
