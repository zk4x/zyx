// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
//! Host-side access to a Blackhole device.
//!
//! Ported from `metalium-rs` `crates/tt-device` (Apache-2.0), minus its
//! `ethernet` module, so that zyx owns the device layer. Written entirely
//! against [`Transport`] so that the simulator and real silicon are the same
//! code path. Nothing here may use `#[cfg]` to distinguish them.

pub mod arc_msg;
pub mod core_control;
pub mod device;
pub mod dram;
pub mod telemetry;
pub mod tlb;
pub mod trace;
pub mod transport;

pub use device::{Device, FencedWrite, PowerPolicy, Tile, Traffic, Window};
pub use transport::{
    Bar, ConfigOffset, HostMemory, Result, Transport, TransportError, DEVICE_ID_BLACKHOLE, VENDOR_ID_TENSTORRENT,
};
