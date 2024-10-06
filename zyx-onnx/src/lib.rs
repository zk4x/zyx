use zyx::ZyxError;
use prost::Message;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod eval;
pub use eval::{dtype, simple_eval};

pub fn read_file<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto, ZyxError> {
    let buf = std::fs::read(p)?;
    onnx::ModelProto::decode(buf.as_slice()).map_err(ZyxError::ParseError("Failed to decode onnx model".into()))
}

