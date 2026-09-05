// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use std::fmt::{Display, Write};

use crate::tensor::TensorId;

/// Enumeration representing the various errors that can occur within the Zyx library.
#[derive(Debug)]
pub enum ZyxError {
    /// Invalid shapes for operation
    ShapeError(Box<str>),
    /// Wrong dtype for given operation
    DTypeError(Box<str>),
    /// Error from file operations
    IOError(std::io::Error),
    /// Error parsing some data
    ParseError(Box<str>),
    /// Memory allocation error
    AllocationError(Box<str>),
    /// There are no available backends
    NoBackendAvailable,
    /// Error returned by backends
    BackendError(BackendError),
    /// Failed to launch kernel due to unknown reasons
    KernelLaunchFailure,
    /// Tried to load a graph tensor without calling realize first
    GraphTensorNotRealized(Box<str>),
    /// A frozen tape's leaf bindings (buffer pools) changed since `freeze` —
    /// the compiled plan no longer matches the inputs. The tape must be
    /// re-frozen.
    FrozenPlanStale(Box<str>),
    /// Invalid kernel structure (e.g. an op of the wrong kind for a transform)
    KernelError(Box<str>),
}

impl ZyxError {
    /// Shape error
    #[track_caller]
    #[must_use]
    pub fn shape_error(e: Box<str>) -> Self {
        let location = std::panic::Location::caller();
        let mut e: String = e.into();
        write!(e, ", {}:{}:{}", location.file(), location.line(), location.column()).unwrap();
        Self::ShapeError(e.into())
    }

    /// `DType` error
    #[track_caller]
    #[must_use]
    pub fn dtype_error(e: Box<str>) -> Self {
        let location = std::panic::Location::caller();
        let mut e: String = e.into();
        write!(e, ", {}:{}:{}", location.file(), location.line(), location.column()).unwrap();
        Self::DTypeError(e.into())
    }

    /// Parse error
    #[track_caller]
    #[must_use]
    pub fn parse_error(e: Box<str>) -> Self {
        let location = std::panic::Location::caller();
        let mut e: String = e.into();
        write!(e, ", {}:{}:{}", location.file(), location.line(), location.column()).unwrap();
        Self::ParseError(e.into())
    }

    /// Error when trying to load a graph tensor that has not been realized yet.
    /// Graph tensors must be materialized (via `Tape::realize`) before their
    /// buffers can be accessed. This error indicates the graph was never compiled.
    #[track_caller]
    pub fn graph_tensor_not_realized(tid: TensorId) -> Self {
        let location = std::panic::Location::caller();
        Self::GraphTensorNotRealized(
            format!(
                "graph tensor {tid} must be realized before loading, at {}:{}:{}",
                location.file(),
                location.line(),
                location.column()
            )
            .into(),
        )
    }

    /// A frozen tape was replayed with inputs whose buffer bindings changed
    /// since `freeze` compiled the plan. The frozen contract is that bindings
    /// are fixed; the tape must be re-frozen for the new inputs.
    #[track_caller]
    pub fn frozen_plan_stale(e: Box<str>) -> Self {
        let location = std::panic::Location::caller();
        let mut e: String = e.into();
        write!(e, ", {}:{}:{}", location.file(), location.line(), location.column()).unwrap();
        Self::FrozenPlanStale(e.into())
    }

    /// Kernel error
    #[track_caller]
    #[must_use]
    pub fn kernel_error(e: Box<str>) -> Self {
        let location = std::panic::Location::caller();
        let mut e: String = e.into();
        write!(e, ", {}:{}:{}", location.file(), location.line(), location.column()).unwrap();
        Self::KernelError(e.into())
    }
}

impl std::fmt::Display for ZyxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZyxError::ShapeError(e) => f.write_str(e),
            ZyxError::DTypeError(e) => f.write_fmt(format_args!("Wrong dtype {e:?}")),
            ZyxError::IOError(e) => f.write_fmt(format_args!("IO {e}")),
            ZyxError::ParseError(e) => f.write_fmt(format_args!("IO {e}")),
            ZyxError::NoBackendAvailable => f.write_fmt(format_args!("No available backend")),
            ZyxError::AllocationError(e) => f.write_fmt(format_args!("Allocation error {e}")),
            ZyxError::BackendError(e) => f.write_fmt(format_args!("Backend {e}")),
            ZyxError::KernelLaunchFailure => f.write_fmt(format_args!(
                "Multiple kernel variations failed to launch. Could be autotuner issue, but more likely faulty hardware driver."
            )),
            ZyxError::GraphTensorNotRealized(e) => f.write_str(e),
            ZyxError::FrozenPlanStale(e) => f.write_str(e),
            ZyxError::KernelError(e) => f.write_str(e),
        }
    }
}

impl std::error::Error for ZyxError {}

impl From<std::io::Error> for ZyxError {
    #[track_caller]
    fn from(value: std::io::Error) -> Self {
        Self::IOError(value)
    }
}

#[derive(Debug)]
pub struct BackendError {
    pub status: ErrorStatus,
    pub context: Box<str>,
}

impl From<BackendError> for ZyxError {
    fn from(value: BackendError) -> Self {
        ZyxError::BackendError(value)
    }
}

impl Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}: {}", self.status, self.context))
    }
}

#[derive(Debug)]
pub enum ErrorStatus {
    /// Dynamic library was not found on the disk
    DyLibNotFound,
    /// Backend initialization failure
    Initialization,
    /// Backend deinitialization failure
    Deinitialization,
    /// Failed to enumerate devices
    DeviceEnumeration,
    /// Failed to query device for information
    DeviceQuery,
    /// Failed to allocate memory
    MemoryAllocation,
    /// Failed to deallocate memory
    MemoryDeallocation,
    /// Failed to copy memory to pool
    MemoryCopyH2P,
    /// Failed to copy memory to host
    MemoryCopyP2H,
    /// Kernel argument was not correct
    IncorrectKernelArg,
    /// Failed to compile kernel
    KernelCompilation,
    /// Failed to launch kernel
    KernelLaunch,
    /// Failed to synchronize kernel
    KernelSync,
}
