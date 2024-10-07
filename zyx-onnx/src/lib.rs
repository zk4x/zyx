use zyx::ZyxError;
use prost::Message;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod eval;
pub use eval::{dtype, simple_eval};

pub fn read_file<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto, ZyxError> {
    let buf = std::fs::read(p)?;
    onnx::ModelProto::decode(buf.as_slice()).map_err(|e| ZyxError::ParseError(format!("Failed to decode onnx model {e}")))
}

#[test]
fn t0() -> Result<(), ZyxError> {
    use zyx::{Tensor, DType};

    let mp = read_file("model.onnx")?;

    //let x = Tensor::rand([8, 128], DType::F32)?;
    let mut m = std::collections::HashMap::new();
    //m.insert("x".into(), x);

    let y = simple_eval(&mp, m);

    println!("{y:?}");

    Ok(())
}

