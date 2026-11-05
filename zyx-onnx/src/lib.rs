// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use zyx::ZyxError;
use prost::Message;

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub mod eval;
pub use eval::{dtype, simple_eval, OnnxModel};

pub fn read_file<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto, ZyxError> {
    let path = p.as_ref();
    let buf = std::fs::read(path)?;
    let mut model = onnx::ModelProto::decode(buf.as_slice())
        .map_err(|e| ZyxError::ParseError(format!("Failed to decode onnx model {e}").into()))?;

    // ONNX may keep initializer bytes outside the protobuf
    // (`data_location = EXTERNAL`). Resolve them here, relative to the model
    // file, so that `get_tensor` only ever sees inline `raw_data`. The shape
    // and dtype live in the protobuf; the external file only supplies the
    // element bytes described by `location`/`offset`/`length`.
    let base = path.parent().unwrap_or_else(|| std::path::Path::new(""));
    let mut external_files: std::collections::HashMap<String, Vec<u8>> =
        std::collections::HashMap::new();
    let mut external_checksums: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let resolve_external = |t: &mut onnx::TensorProto,
                            files: &mut std::collections::HashMap<String, Vec<u8>>,
                            checksums: &mut std::collections::HashMap<String, String>|
     -> Result<(), ZyxError> {
        if t.data_location != onnx::tensor_proto::DataLocation::External as i32 {
            return Ok(());
        }
        let mut location: Option<String> = None;
        let mut offset: usize = 0;
        let mut length: Option<usize> = None;
        let mut checksum: Option<String> = None;
        for entry in t.external_data.iter() {
            let value = entry.value.as_str();
            match entry.key.as_str() {
                "location" => location = Some(entry.value.clone()),
                "offset" => {
                    offset = value.parse().map_err(|_| {
                        ZyxError::parse_error(
                            format!("invalid external offset {value:?} for tensor {}", t.name).into(),
                        )
                    })?
                }
                "length" => {
                    length = Some(value.parse().map_err(|_| {
                        ZyxError::parse_error(
                            format!("invalid external length {value:?} for tensor {}", t.name).into(),
                        )
                    })?)
                }
                "checksum" => checksum = Some(entry.value.clone()),
                key => {
                    return Err(ZyxError::parse_error(
                        format!("unsupported external_data key {key:?} for tensor {}", t.name).into(),
                    ))
                }
            }
        }
        let location = location.ok_or_else(|| {
            ZyxError::parse_error(format!("external tensor {} has no location", t.name).into())
        })?;
        let file = match files.entry(location.clone()) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(std::fs::read(base.join(&location))?)
            }
        };
        // The checksum (when present) is the SHA1 of the whole external file.
        if let Some(expected) = checksum.as_deref() {
            if let Some(known) = checksums.get(&location) {
                if !known.eq_ignore_ascii_case(expected) {
                    return Err(ZyxError::parse_error(
                        format!(
                            "conflicting checksums for {location}: {known} vs {expected} for tensor {}",
                            t.name
                        )
                        .into(),
                    ));
                }
            } else {
                use sha1::Digest;
                let digest = sha1::Sha1::digest(&file[..]);
                let got: String = digest.iter().map(|b| format!("{b:02x}")).collect();
                if !got.eq_ignore_ascii_case(expected) {
                    return Err(ZyxError::parse_error(
                        format!(
                            "checksum mismatch for {location}: expected {expected}, got {got} for tensor {}",
                            t.name
                        )
                        .into(),
                    ));
                }
                checksums.insert(location.clone(), expected.to_string());
            }
        }
        let length = length.unwrap_or_else(|| file.len().saturating_sub(offset));
        let end = offset.checked_add(length).ok_or_else(|| {
            ZyxError::parse_error(
                format!("external slice overflows for tensor {}", t.name).into(),
            )
        })?;
        if end > file.len() {
            return Err(ZyxError::parse_error(
                format!(
                    "external slice {offset}..{end} for tensor {} is beyond {} ({} bytes)",
                    t.name,
                    location,
                    file.len()
                )
                .into(),
            ));
        }
        let elem_size = match t.data_type {
            1 | 6 | 12 => 4,                     // FLOAT, INT32, UINT32
            2 | 3 | 9 | 17 | 18 | 19 | 20 => 1,  // UINT8, INT8, BOOL, FLOAT8*
            4 | 5 | 10 | 16 => 2,                // UINT16, INT16, FLOAT16, BFLOAT16
            7 | 11 | 13 | 14 => 8,               // INT64, DOUBLE, UINT64, COMPLEX64
            15 => 16,                            // COMPLEX128
            dt => {
                return Err(ZyxError::parse_error(
                    format!("unsupported external data-type {dt} for tensor {}", t.name).into(),
                ))
            }
        };
        let numel = t.dims.iter().try_fold(1usize, |acc, &d| {
            let d = usize::try_from(d).map_err(|_| {
                ZyxError::parse_error(format!("invalid dim {d} for tensor {}", t.name).into())
            })?;
            acc.checked_mul(d).ok_or_else(|| {
                ZyxError::parse_error(format!("dim product overflow for tensor {}", t.name).into())
            })
        })?;
        let expected = numel.checked_mul(elem_size).ok_or_else(|| {
            ZyxError::parse_error(format!("expected size overflow for tensor {}", t.name).into())
        })?;
        if length != expected {
            return Err(ZyxError::parse_error(
                format!(
                    "external tensor {} has {length} bytes, dims {:?} x {elem_size} need {expected}",
                    t.name, t.dims
                )
                .into(),
            ));
        }
        t.raw_data = file[offset..end].to_vec();
        t.data_location = onnx::tensor_proto::DataLocation::Default as i32;
        Ok(())
    };

    if let Some(graph) = model.graph.as_mut() {
        for t in graph.initializer.iter_mut() {
            resolve_external(t, &mut external_files, &mut external_checksums)?;
        }
        for sparse in graph.sparse_initializer.iter_mut() {
            if let Some(t) = sparse.values.as_mut() {
                resolve_external(t, &mut external_files, &mut external_checksums)?;
            }
            if let Some(t) = sparse.indices.as_mut() {
                resolve_external(t, &mut external_files, &mut external_checksums)?;
            }
        }
        for node in graph.node.iter_mut() {
            for attr in node.attribute.iter_mut() {
                if let Some(t) = attr.t.as_mut() {
                    resolve_external(t, &mut external_files, &mut external_checksums)?;
                }
                for t in attr.tensors.iter_mut() {
                    resolve_external(t, &mut external_files, &mut external_checksums)?;
                }
                if let Some(sparse) = attr.sparse_tensor.as_mut() {
                    if let Some(t) = sparse.values.as_mut() {
                        resolve_external(t, &mut external_files, &mut external_checksums)?;
                    }
                    if let Some(t) = sparse.indices.as_mut() {
                        resolve_external(t, &mut external_files, &mut external_checksums)?;
                    }
                }
            }
        }
    }

    Ok(model)
}

pub fn load_model<P: AsRef<std::path::Path>>(p: P) -> Result<OnnxModel, ZyxError> {
    OnnxModel::load(p)
}

#[test]
fn t0() -> Result<(), ZyxError> {
    use zyx::{Tensor, DType};

    let mp = read_file("model.onnx")?;

    let x = Tensor::rand([8, 128], DType::F32)?;
    let mut m = std::collections::HashMap::new();
    m.insert("x".into(), x);

    let outputs = simple_eval(&mp, m)?;

    println!("{:?}", outputs["4"]);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{self, tensor_proto::DataType, type_proto, TensorProto, ValueInfoProto};
    use std::collections::HashMap;
    use zyx::{DType, Tensor, ZyxError};

    fn make_value_info(name: &str, dtype: DataType, shape: &[i64]) -> ValueInfoProto {
        let dims = shape
            .iter()
            .map(|&d| onnx::tensor_shape_proto::Dimension {
                value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(d)),
                denotation: "".to_string(),
            })
            .collect();
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx::TypeProto {
                value: Some(type_proto::Value::TensorType(onnx::type_proto::Tensor {
                    elem_type: dtype as i32,
                    shape: Some(onnx::TensorShapeProto { dim: dims }),
                })),
                denotation: "".to_string(),
            }),
            doc_string: "".to_string(),
        }
    }

    fn make_model(
        nodes: Vec<onnx::NodeProto>,
        inputs: Vec<ValueInfoProto>,
        outputs: Vec<ValueInfoProto>,
        initializers: Vec<TensorProto>,
    ) -> onnx::ModelProto {
        let graph = onnx::GraphProto {
            node: nodes,
            name: "test_graph".to_string(),
            initializer: initializers,
            input: inputs,
            output: outputs,
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        };
        onnx::ModelProto {
            ir_version: 7,
            opset_import: vec![onnx::OperatorSetIdProto {
                domain: "".to_string(),
                version: 14,
            }],
            producer_name: "test".to_string(),
            producer_version: "1".to_string(),
            domain: "".to_string(),
            model_version: 1,
            doc_string: "".to_string(),
            graph: Some(graph),
            metadata_props: vec![],
            training_info: vec![],
            functions: vec![],
        }
    }

    fn assert_close(a: &Tensor, b: &Tensor, tol: f32) -> Result<(), ZyxError> {
        let av: Vec<f32> = a.clone().cast(DType::F32).try_into()?;
        let bv: Vec<f32> = b.clone().cast(DType::F32).try_into()?;
        assert_eq!(av.len(), bv.len(), "len mismatch {} vs {}", av.len(), bv.len());
        for (i, (x, y)) in av.iter().zip(bv.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at {i}: {x} vs {y} diff {}",
                (x - y).abs()
            );
        }
        Ok(())
    }

    fn run_onnx(
        model: &onnx::ModelProto,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, ZyxError> {
        // test both simple_eval and OnnxModel give same
        let mut inputs_clone = inputs.clone();
        let out_simple = simple_eval(model, inputs_clone)?;
        let onnx_model = OnnxModel::from_model(model)?;
        let out_frozen = onnx_model.run(inputs)?;
        // compare
        for (k, v) in &out_simple {
            let fv = out_frozen.get(k).expect("missing frozen output");
            assert_close(v, fv, 1e-4)?;
        }
        Ok(out_frozen)
    }

    #[test]
    fn test_tape_add_direct() -> Result<(), ZyxError> {
        let a = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[5.0f32, 6.0], [7.0, 8.0]]);
        let expected = &a + &b;
        let tape = zyx::Tape::empty();
        tape.add(&a)?;
        tape.add(&b)?;
        let c = &a + &b;
        let frozen = tape.freeze([&a, &b], [&c])?;
        let outs = frozen.replay([&a, &b])?;
        let ov: Vec<f32> = outs[0].clone().try_into()?;
        eprintln!("direct tape add out: {:?}", ov);
        let ev: Vec<f32> = expected.clone().try_into()?;
        eprintln!("expected: {:?}", ev);
        assert_close(&outs[0], &expected, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_tape_add_with_zeros_placeholder() -> Result<(), ZyxError> {
        // Mimic OnnxModel's placeholder zeros as Leaf (host-backed)
        let a_ph = Tensor::from(vec![0.0f32; 4]).reshape([2, 2])?;
        let b_ph = Tensor::from(vec![0.0f32; 4]).reshape([2, 2])?;
        let tape = zyx::Tape::empty();
        tape.add(&a_ph)?;
        tape.add(&b_ph)?;
        let c_ph = &a_ph + &b_ph;
        let frozen = tape.freeze([&a_ph, &b_ph], [&c_ph])?;
        let a = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[5.0f32, 6.0], [7.0, 8.0]]);
        let expected = &a + &b;
        let outs = frozen.replay([&a, &b])?;
        let ov: Vec<f32> = outs[0].clone().try_into()?;
        eprintln!("zeros placeholder tape out: {:?}", ov);
        assert_close(&outs[0], &expected, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_add() -> Result<(), ZyxError> {
        let a = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Tensor::from([[5.0f32, 6.0], [7.0, 8.0]]);
        let expected = &a + &b;
        let ev: Vec<f32> = expected.clone().try_into()?;
        eprintln!("expected: {:?}", ev);
        let av: Vec<f32> = a.clone().try_into()?;
        eprintln!("a: {:?}", av);
        let bv: Vec<f32> = b.clone().try_into()?;
        eprintln!("b: {:?}", bv);
        let model = make_model(
            vec![onnx::NodeProto {
                input: vec!["a".to_string(), "b".to_string()],
                output: vec!["c".to_string()],
                name: "add".to_string(),
                op_type: "Add".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                doc_string: "".to_string(),
            }],
            vec![
                make_value_info("a", DataType::Float, &[2, 2]),
                make_value_info("b", DataType::Float, &[2, 2]),
            ],
            vec![make_value_info("c", DataType::Float, &[2, 2])],
            vec![],
        );
        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), a.clone());
        inputs.insert("b".to_string(), b.clone());
        let outs = run_onnx(&model, inputs)?;
        let ov: Vec<f32> = outs["c"].clone().try_into()?;
        eprintln!("onnx out: {:?}", ov);
        let simple_out: Vec<f32> = simple_eval(&model, {
            let mut m = HashMap::new();
            m.insert("a".to_string(), a.clone());
            m.insert("b".to_string(), b.clone());
            m
        })?["c"].clone().try_into()?;
        eprintln!("simple_eval out: {:?}", simple_out);
        assert_close(&outs["c"], &expected, 1e-5)?;
        Ok(())
    }

    #[test]
    fn test_relu_gemm() -> Result<(), ZyxError> {
        // Test Gemm + Relu chain like model.onnx
        let x = Tensor::rand([2, 4], DType::F32)?;
        let w = Tensor::rand([3, 4], DType::F32)?;
        let b = Tensor::rand([3], DType::F32)?;
        // Gemm with transB=1: x * w^T + b
        let expected = (x.clone().matmul(&w.t())? + b.clone().reshape([1, 3])?.expand([2, 3])?).relu();
        // Build model: Gemm (x, w, b) -> y, Relu(y) -> z
        let w_init = TensorProto {
            dims: vec![3, 4],
            data_type: DataType::Float as i32,
            segment: None,
            float_data: w.clone().try_into()?,
            int32_data: vec![],
            string_data: vec![],
            int64_data: vec![],
            name: "w".to_string(),
            doc_string: "".to_string(),
            raw_data: vec![],
            external_data: vec![],
            data_location: 0,
            double_data: vec![],
            uint64_data: vec![],
        };
        let b_init = TensorProto {
            dims: vec![3],
            data_type: DataType::Float as i32,
            segment: None,
            float_data: b.clone().try_into()?,
            int32_data: vec![],
            string_data: vec![],
            int64_data: vec![],
            name: "b".to_string(),
            doc_string: "".to_string(),
            raw_data: vec![],
            external_data: vec![],
            data_location: 0,
            double_data: vec![],
            uint64_data: vec![],
        };
        let model = make_model(
            vec![
                onnx::NodeProto {
                    input: vec!["x".to_string(), "w".to_string(), "b".to_string()],
                    output: vec!["y".to_string()],
                    name: "gemm".to_string(),
                    op_type: "Gemm".to_string(),
                    domain: "".to_string(),
                    attribute: vec![
                        onnx::AttributeProto {
                            name: "transB".to_string(),
                            ref_attr_name: "".to_string(),
                            doc_string: "".to_string(),
                            r#type: 2,
                            f: 0.0,
                            i: 1,
                            s: vec![],
                            t: None,
                            g: None,
                            sparse_tensor: None,
                            tp: None,
                            floats: vec![],
                            ints: vec![],
                            strings: vec![],
                            tensors: vec![],
                            graphs: vec![],
                            sparse_tensors: vec![],
                            type_protos: vec![],
                        },
                    ],
                    doc_string: "".to_string(),
                },
                onnx::NodeProto {
                    input: vec!["y".to_string()],
                    output: vec!["z".to_string()],
                    name: "relu".to_string(),
                    op_type: "Relu".to_string(),
                    domain: "".to_string(),
                    attribute: vec![],
                    doc_string: "".to_string(),
                },
            ],
            vec![make_value_info("x", DataType::Float, &[2, 4])],
            vec![make_value_info("z", DataType::Float, &[2, 3])],
            vec![w_init, b_init],
        );
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x);
        let outs = run_onnx(&model, inputs)?;
        assert_close(&outs["z"], &expected, 1e-4)?;
        Ok(())
    }

    #[test]
    fn test_onnx_model_wrapper() -> Result<(), ZyxError> {
        let mp = read_file("model.onnx")?;
        let model = OnnxModel::from_model(&mp)?;
        assert_eq!(model.input_names(), &["x"]);
        assert_eq!(model.output_names(), &["4"]);
        let x = Tensor::rand([8, 128], DType::F32)?;
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x.clone());
        let out1 = model.run(inputs.clone())?;
        let out2 = simple_eval(&mp, inputs)?;
        assert_close(&out1["4"], &out2["4"], 1e-4)?;
        // test second run
        let mut inputs2 = HashMap::new();
        inputs2.insert("x".to_string(), Tensor::rand([8, 128], DType::F32)?);
        let _ = model.run(inputs2)?;
        Ok(())
    }

    #[test]
    fn test_from_file_convenience() -> Result<(), ZyxError> {
        let model = OnnxModel::from_file("model.onnx")?;
        assert_eq!(model.input_names(), &["x"]);
        Ok(())
    }

    #[test]
    fn test_mnist_linear_static() -> Result<(), ZyxError> {
        let mp = read_file("mnist_linear_static.onnx")?;
        let model = OnnxModel::from_model(&mp)?;
        let x = Tensor::rand([1, 784], DType::F32)?;
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x);
        let outs = model.run(inputs)?;
        assert!(outs.contains_key("y"));
        let y = &outs["y"];
        assert_eq!(y.resolve_shape(), vec![1, 10]);
        Ok(())
    }

    #[test]
    fn test_ops_coverage() -> Result<(), ZyxError> {
        // Test a bunch of unary/binary ops via OnnxModel
        let cases: Vec<(&str, Tensor, Tensor)> = vec![
            ("Abs", Tensor::from([-1.0f32, 2.0, -3.0]), Tensor::from([1.0, 2.0, 3.0])),
            ("Neg", Tensor::from([1.0f32, -2.0]), Tensor::from([-1.0, 2.0])),
            ("Sqrt", Tensor::from([1.0f32, 4.0, 9.0]), Tensor::from([1.0, 2.0, 3.0])),
            ("Exp", Tensor::from([0.0f32, 1.0]), Tensor::from([1.0, std::f32::consts::E])),
            ("Log", Tensor::from([1.0f32, std::f32::consts::E]), Tensor::from([0.0, 1.0])),
        ];
        for (op, input, expected) in cases {
            let model = make_model(
                vec![onnx::NodeProto {
                    input: vec!["x".to_string()],
                    output: vec!["y".to_string()],
                    name: op.to_string(),
                    op_type: op.to_string(),
                    domain: "".to_string(),
                    attribute: vec![],
                    doc_string: "".to_string(),
                }],
                vec![make_value_info("x", DataType::Float, &[input.resolve_shape()[0]])],
                vec![make_value_info("y", DataType::Float, &[expected.resolve_shape()[0]])],
                vec![],
            );
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), input.clone());
            let outs = run_onnx(&model, inputs)?;
            assert_close(&outs["y"], &expected, 1e-4)?;
        }
        Ok(())
    }

    #[test]
    fn test_gather_rank1_indices() -> Result<(), ZyxError> {
        let x = Tensor::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let idx = Tensor::from([2i64, 0]);
        let model = make_model(
            vec![onnx::NodeProto {
                input: vec!["x".to_string(), "idx".to_string()],
                output: vec!["y".to_string()],
                name: "gather".to_string(),
                op_type: "Gather".to_string(),
                domain: "".to_string(),
                attribute: vec![onnx::AttributeProto {
                    name: "axis".to_string(),
                    ref_attr_name: "".to_string(),
                    doc_string: "".to_string(),
                    r#type: 2,
                    f: 0.0,
                    i: 1,
                    s: vec![],
                    t: None,
                    g: None,
                    sparse_tensor: None,
                    tp: None,
                    floats: vec![],
                    ints: vec![],
                    strings: vec![],
                    tensors: vec![],
                    graphs: vec![],
                    sparse_tensors: vec![],
                    type_protos: vec![],
                }],
                doc_string: "".to_string(),
            }],
            vec![
                make_value_info("x", DataType::Float, &[2, 3]),
                make_value_info("idx", DataType::Int64, &[2]),
            ],
            vec![make_value_info("y", DataType::Float, &[2, 2])],
            vec![],
        );
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x.clone());
        inputs.insert("idx".to_string(), idx);
        let outs = run_onnx(&model, inputs)?;
        let expected = Tensor::from([[3.0f32, 1.0], [6.0, 4.0]]);
        assert_close(&outs["y"], &expected, 1e-6)?;
        Ok(())
    }

    #[test]
    fn test_reshape_transpose() -> Result<(), ZyxError> {
        let x = Tensor::from([[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
        let expected = x.clone().reshape([8])?;
        // Reshape via ONNX: second input is shape tensor initializer
        let shape_init = TensorProto {
            dims: vec![1],
            data_type: DataType::Int64 as i32,
            segment: None,
            float_data: vec![],
            int32_data: vec![],
            string_data: vec![],
            int64_data: vec![8],
            name: "shape".to_string(),
            doc_string: "".to_string(),
            raw_data: vec![],
            external_data: vec![],
            data_location: 0,
            double_data: vec![],
            uint64_data: vec![],
        };
        let model = make_model(
            vec![onnx::NodeProto {
                input: vec!["x".to_string(), "shape".to_string()],
                output: vec!["y".to_string()],
                name: "reshape".to_string(),
                op_type: "Reshape".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                doc_string: "".to_string(),
            }],
            vec![make_value_info("x", DataType::Float, &[2, 4])],
            vec![make_value_info("y", DataType::Float, &[8])],
            vec![shape_init],
        );
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x);
        let outs = run_onnx(&model, inputs)?;
        assert_close(&outs["y"], &expected, 1e-5)?;

        // Transpose test
        let x2 = Tensor::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let expected2 = x2.t();
        let model2 = make_model(
            vec![onnx::NodeProto {
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
                name: "transpose".to_string(),
                op_type: "Transpose".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                doc_string: "".to_string(),
            }],
            vec![make_value_info("x", DataType::Float, &[2, 2])],
            vec![make_value_info("y", DataType::Float, &[2, 2])],
            vec![],
        );
        let mut inputs2 = HashMap::new();
        inputs2.insert("x".to_string(), x2);
        let outs2 = run_onnx(&model2, inputs2)?;
        assert_close(&outs2["y"], &expected2, 1e-5)?;
        Ok(())
    }
}
