use super::DeviceInfo;

#[derive(Debug)]
pub(crate) struct X86_64Error {
    info: String,
}

#[derive(Debug)]
pub(crate) struct X86_64Program {}

#[derive(Debug)]
pub(crate) struct X86_64Buffer {}

#[derive(Debug)]
pub(crate) struct X86_64Device {}

impl X86_64Device {
    pub(crate) fn new() -> Self {
        // TODO also generate device info and store it in the device
        X86_64Device {}
    }
}