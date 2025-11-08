use std::fmt::Display;

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
}

impl ZyxError {
    /// Shape error
    #[track_caller]
    pub fn shape_error(e: Box<str>) -> Self {
        let location = std::panic::Location::caller();
        use std::fmt::Write;
        let mut e: String = e.into();
        write!(e, ", {}:{}:{}", location.file(), location.line(), location.column()).unwrap();
        Self::ShapeError(e.into())
    }

    /// DType error
    #[track_caller]
    pub fn dtype_error(e: Box<str>) -> Self {
        let location = std::panic::Location::caller();
        use std::fmt::Write;
        let mut e: String = e.into();
        write!(e, ", {}:{}:{}", location.file(), location.line(), location.column()).unwrap();
        Self::DTypeError(e.into())
    }

    /// Parse error
    #[track_caller]
    pub fn parse_error(e: Box<str>) -> Self {
        let location = std::panic::Location::caller();
        use std::fmt::Write;
        let mut e: String = e.into();
        write!(e, ", {}:{}:{}", location.file(), location.line(), location.column()).unwrap();
        Self::ParseError(e.into())
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
