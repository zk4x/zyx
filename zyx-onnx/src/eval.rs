// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use crate::onnx::attribute_proto::AttributeType;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::{self, GraphProto};
use std::collections::{HashMap, HashSet};
use zyx::{DType, FrozenTape, Tape, Tensor, ZyxError, bf16, f8e4m3, f8e5m2, f16};

pub type Value = Tensor;

pub fn dtype(dt: DataType) -> Option<DType> {
    match dt {
        DataType::Float => Some(DType::F32),
        DataType::Double => Some(DType::F64),
        DataType::Float16 => Some(DType::F16),
        DataType::Bfloat16 => Some(DType::BF16),
        DataType::Float8e4m3fn => Some(DType::F8E4M3),
        DataType::Float8e5m2 => Some(DType::F8E5M2),
        DataType::Uint8 => Some(DType::U8),
        DataType::Uint16 => Some(DType::U16),
        DataType::Uint32 => Some(DType::U32),
        DataType::Uint64 => Some(DType::U64),
        DataType::Int8 => Some(DType::I8),
        DataType::Int16 => Some(DType::I16),
        DataType::Int32 => Some(DType::I32),
        DataType::Int64 => Some(DType::I64),
        DataType::Bool => Some(DType::Bool),
        // STRING, UNDEFINED, COMPLEX*, and the fnuz float8 variants have no
        // zyx representation.
        _ => None,
    }
}

fn data_type_from_i32(v: i32) -> Option<DataType> {
    match v {
        0 => Some(DataType::Undefined),
        1 => Some(DataType::Float),
        2 => Some(DataType::Uint8),
        3 => Some(DataType::Int8),
        4 => Some(DataType::Uint16),
        5 => Some(DataType::Int16),
        6 => Some(DataType::Int32),
        7 => Some(DataType::Int64),
        8 => Some(DataType::String),
        9 => Some(DataType::Bool),
        10 => Some(DataType::Float16),
        11 => Some(DataType::Double),
        12 => Some(DataType::Uint32),
        13 => Some(DataType::Uint64),
        14 => Some(DataType::Complex64),
        15 => Some(DataType::Complex128),
        16 => Some(DataType::Bfloat16),
        17 => Some(DataType::Float8e4m3fn),
        18 => Some(DataType::Float8e4m3fnuz),
        19 => Some(DataType::Float8e5m2),
        20 => Some(DataType::Float8e5m2fnuz),
        _ => None,
    }
}

trait Attr {
    const TYPE: AttributeType;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self, ZyxError>;
}

trait AttrOwned: Sized {
    const TYPE: AttributeType;
    fn get(attr: &onnx::AttributeProto) -> Result<Self, ZyxError>;
}

impl Attr for i64 {
    const TYPE: AttributeType = AttributeType::Int;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self, ZyxError> {
        Ok(&attr.i)
    }
}

impl Attr for f32 {
    const TYPE: AttributeType = AttributeType::Float;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self, ZyxError> {
        Ok(&attr.f)
    }
}

impl Attr for [i64] {
    const TYPE: AttributeType = AttributeType::Ints;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self, ZyxError> {
        Ok(attr.ints.as_slice())
    }
}

impl Attr for str {
    const TYPE: AttributeType = AttributeType::String;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self, ZyxError> {
        std::str::from_utf8(&attr.s).map_err(|e| ZyxError::ParseError(format!("Failed to parse {e}").into()))
    }
}

impl Attr for GraphProto {
    const TYPE: AttributeType = AttributeType::Graph;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self, ZyxError> {
        attr.g
            .as_ref()
            .ok_or_else(|| ZyxError::ParseError("attribute does not contain graph".to_string().into()))
    }
}

impl AttrOwned for Vec<String> {
    const TYPE: AttributeType = AttributeType::Strings;
    fn get(attr: &onnx::AttributeProto) -> Result<Self, ZyxError> {
        let mut ret = vec![];
        for bytes in attr.strings.iter() {
            let s = String::from_utf8(bytes.clone()).map_err(|e| ZyxError::ParseError(format!("{e}").into()))?;
            ret.push(s);
        }
        Ok(ret)
    }
}

impl AttrOwned for Tensor {
    const TYPE: AttributeType = AttributeType::Tensor;
    fn get(attr: &onnx::AttributeProto) -> Result<Self, ZyxError> {
        let tensor_proto = match &attr.t {
            Some(value) => value,
            None => panic!(
                "attribute {} was of type TENSOR, but no tensor was found",
                attr.name
            ),
        };

        // Reuse get_tensor so raw_data is interpreted correctly (typed, not cast).
        return get_tensor(tensor_proto, &attr.name);
    }
}

fn get_attr_<'a>(node: &'a onnx::NodeProto, name: &str) -> Result<&'a onnx::AttributeProto, ZyxError> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => {
            panic!(
                "cannot find the '{name}' attribute in '{}' for {}",
                node.op_type,
                node.name
            )
        }
        Some(dt) => Ok(dt),
    }
}

fn get_attr<'a, T: Attr + ?Sized>(node: &'a onnx::NodeProto, name: &str) -> Result<&'a T, ZyxError> {
    let attr = get_attr_(node, name)?;
    if attr.r#type != T::TYPE as i32 {
        panic!(
            "unsupported type {:?} for '{name}' attribute in '{}' for {}",
            attr.r#type,
            node.op_type,
            node.name
        )
    }
    T::get(attr)
}

fn get_attr_opt<'a, T: Attr + ?Sized>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> Result<Option<&'a T>, ZyxError> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => Ok(None),
        Some(attr) => {
            if attr.r#type != T::TYPE as i32 {
                panic!(
                    "unsupported type {:?} for '{name}' attribute in '{}' for {}",
                    attr.r#type,
                    node.op_type,
                    node.name
                )
            }
            let val = T::get(attr)?;
            Ok(Some(val))
        }
    }
}

fn get_attr_opt_owned<T: AttrOwned>(node: &onnx::NodeProto, name: &str) -> Result<Option<T>, ZyxError> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => Ok(None),
        Some(attr) => {
            if attr.r#type != T::TYPE as i32 {
                panic!(
                    "unsupported type {:?} for '{name}' attribute in '{}' for {}",
                    attr.r#type,
                    node.op_type,
                    node.name
                )
            }
            let val = T::get(attr)?;
            Ok(Some(val))
        }
    }
}

pub fn get_tensor(t: &onnx::TensorProto, name: &str) -> Result<Tensor, ZyxError> {
    let dims: Vec<Tensor> = t.dims.iter().map(|&x| Tensor::from(x)).collect();
    let dt = data_type_from_i32(t.data_type)
        .unwrap_or_else(|| panic!("unsupported 'value' data-type {} for {name}", t.data_type));
    let dtype = dtype(dt).unwrap_or_else(|| panic!("unsupported 'value' data-type {dt:?} for {name}"));

    // Typed fields take precedence; the narrow types use int32_data (holding
    // their bit patterns for the float types), while F32/F64/I64 have
    // dedicated fields.
    let base = match dtype {
        DType::F32 if !t.float_data.is_empty() => Tensor::from(t.float_data.clone()),
        DType::F64 if !t.double_data.is_empty() => Tensor::from(t.double_data.clone()),
        DType::I64 if !t.int64_data.is_empty() => Tensor::from(t.int64_data.clone()),
        DType::I32 if !t.int32_data.is_empty() => Tensor::from(t.int32_data.clone()),
        DType::Bool if !t.int32_data.is_empty() => {
            Tensor::from(t.int32_data.iter().map(|&v| v != 0).collect::<Vec<_>>())
        }
        DType::U8 if !t.int32_data.is_empty() => {
            Tensor::from(t.int32_data.iter().map(|&v| v as u8).collect::<Vec<_>>())
        }
        DType::I8 if !t.int32_data.is_empty() => {
            Tensor::from(t.int32_data.iter().map(|&v| v as i8).collect::<Vec<_>>())
        }
        DType::U16 if !t.int32_data.is_empty() => {
            Tensor::from(t.int32_data.iter().map(|&v| v as u16).collect::<Vec<_>>())
        }
        DType::I16 if !t.int32_data.is_empty() => {
            Tensor::from(t.int32_data.iter().map(|&v| v as i16).collect::<Vec<_>>())
        }
        DType::F16 if !t.int32_data.is_empty() => {
            Tensor::from(t.int32_data.iter().map(|&v| f16(v as u16)).collect::<Vec<_>>())
        }
        DType::BF16 if !t.int32_data.is_empty() => {
            Tensor::from(t.int32_data.iter().map(|&v| bf16(v as u16)).collect::<Vec<_>>())
        }
        DType::F8E4M3 if !t.int32_data.is_empty() => {
            Tensor::from(t.int32_data.iter().map(|&v| f8e4m3(v as u8)).collect::<Vec<_>>())
        }
        DType::F8E5M2 if !t.int32_data.is_empty() => {
            Tensor::from(t.int32_data.iter().map(|&v| f8e5m2(v as u8)).collect::<Vec<_>>())
        }
        _ => {
            // raw_data is little-endian bytes for `dtype`; interpret directly
            // rather than bitcasting.
            match dtype {
                DType::F32 => Tensor::from(
                    t.raw_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                ),
                DType::F64 => Tensor::from(
                    t.raw_data
                        .chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                ),
                DType::I64 => Tensor::from(
                    t.raw_data
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                ),
                DType::I32 => Tensor::from(
                    t.raw_data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                ),
                DType::U64 => Tensor::from(
                    t.raw_data
                        .chunks_exact(8)
                        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                ),
                DType::U32 => Tensor::from(
                    t.raw_data
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                ),
                DType::U16 => Tensor::from(
                    t.raw_data
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                ),
                DType::I16 => Tensor::from(
                    t.raw_data
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes(c.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                ),
                DType::U8 => Tensor::from(t.raw_data.clone()),
                DType::I8 => Tensor::from(t.raw_data.iter().map(|&b| b as i8).collect::<Vec<_>>()),
                DType::Bool => Tensor::from(t.raw_data.iter().map(|&b| b != 0).collect::<Vec<_>>()),
                DType::F16 => Tensor::from(
                    t.raw_data
                        .chunks_exact(2)
                        .map(|c| f16(u16::from_le_bytes(c.try_into().unwrap())))
                        .collect::<Vec<_>>(),
                ),
                DType::BF16 => Tensor::from(
                    t.raw_data
                        .chunks_exact(2)
                        .map(|c| bf16(u16::from_le_bytes(c.try_into().unwrap())))
                        .collect::<Vec<_>>(),
                ),
                DType::F8E4M3 => {
                    Tensor::from(t.raw_data.iter().map(|&b| f8e4m3(b)).collect::<Vec<_>>())
                }
                DType::F8E5M2 => {
                    Tensor::from(t.raw_data.iter().map(|&b| f8e5m2(b)).collect::<Vec<_>>())
                }
            }
        }
    };
    if dims.is_empty() { Ok(base) } else { base.reshape(dims) }
}

// This function provides a direct evaluation of the proto.
pub fn simple_eval(
    model: &onnx::ModelProto,
    mut inputs: HashMap<String, Value>,
) -> Result<HashMap<String, Value>, ZyxError> {
    let graph = match &model.graph {
        None => panic!("no graph defined in proto"),
        Some(graph) => graph,
    };
    simple_eval_(graph, &mut inputs)
}

fn simple_eval_(
    graph: &onnx::GraphProto,
    values: &mut HashMap<String, Value>,
) -> Result<HashMap<String, Value>, ZyxError> {
    for t in graph.initializer.iter() {
        let tensor = get_tensor(t, t.name.as_str())?;
        values.insert(t.name.to_string(), tensor);
    }
    for input in graph.input.iter() {
        let input_type = match &input.r#type {
            Some(input_type) => input_type,
            None => continue,
        };
        let input_type = match &input_type.value {
            Some(input_type) => input_type,
            None => continue,
        };
        let tensor_type = match input_type {
            onnx::type_proto::Value::TensorType(tt) => tt,
            _ => continue,
        };

        let tensor = match values.get(&input.name) {
            None => panic!("missing input {}", input.name),
            Some(tensor) => tensor,
        };
        let dt = match data_type_from_i32(tensor_type.elem_type) {
            Some(dt) => match dtype(dt) {
                Some(dt) => dt,
                None => {
                    panic!("unsupported 'value' data-type {dt:?} for {}", input.name)
                }
            },
            None => panic!("unsupported input type {:?}", tensor_type.elem_type),
        };
        match &tensor_type.shape {
            None => continue,
            Some(shape) => {
                if shape.dim.len() != tensor.rank() as usize {
                    panic!(
                        "unexpected rank for {}, got {:?}, expected {:?}",
                        input.name,
                        shape.dim,
                        tensor.resolve_shape()
                    )
                }
                for (idx, (d, &dim)) in shape.dim.iter().zip(tensor.resolve_shape().iter()).enumerate() {
                    match &d.value {
                        Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                            if *v as usize != dim as usize {
                                panic!(
                                    "unexpected dim {idx} for {}, got {:?}, expected {:?}",
                                    input.name,
                                    shape.dim,
                                    tensor.resolve_shape()
                                )
                            }
                        }
                        Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) | None => (),
                    }
                }
            }
        };
        if dt != tensor.dtype() {
            panic!(
                "unexpected dtype for {}, got {:?}, expected {dt:?}",
                input.name,
                tensor.dtype()
            )
        }
    }
    eval_nodes(graph, values)?;
    graph
        .output
        .iter()
        .map(|output| match values.remove(&output.name) {
            None => panic!("cannot find output {}", output.name),
            Some(value) => Ok((output.name.clone(), value)),
        })
        .collect()
}

pub(crate) fn eval_nodes(
    graph: &onnx::GraphProto,
    values: &mut HashMap<String, Value>,
) -> Result<(), ZyxError> {
    for node in graph.node.iter() {
        let get = |input_name: &str| match values.get(input_name) {
            Some(value) => value,
            None => panic!("cannot find {input_name} for op '{}'", node.name),
        };
        let get_opt = |i: usize| {
            node.input
                .get(i)
                .filter(|s: &&String| !s.is_empty())
                .map(|s| get(s))
        };

        match node.op_type.as_str() {
            "Add" => {
                let input0 = get(&node.input[0]);
                let input1 = get(&node.input[1]);
                let output = input0 + input1;
                values.insert(node.output[0].clone(), output);
            }
            "Sub" => {
                let input0 = get(&node.input[0]);
                let input1 = get(&node.input[1]);
                let output = input0 - input1;
                values.insert(node.output[0].clone(), output);
            }
            "Mul" => {
                let input0 = get(&node.input[0]);
                let input1 = get(&node.input[1]);
                let output = input0 * input1;
                values.insert(node.output[0].clone(), output);
            }
            "Div" => {
                let input0 = get(&node.input[0]);
                let input1 = get(&node.input[1]);
                let output = input0 / input1;
                values.insert(node.output[0].clone(), output);
            }
            "Pow" => {
                let input0 = get(&node.input[0]);
                let input1 = get(&node.input[1]);
                let output = input0.pow(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Exp" => {
                let xs = get(&node.input[0]);
                let output = xs.exp();
                values.insert(node.output[0].clone(), output);
            }
            "Equal" => {
                let input0 = get(&node.input[0]);
                let input1 = get(&node.input[1]);
                let output = input0.equal(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Not" => {
                let xs = get(&node.input[0]);
                let xs = !xs;
                values.insert(node.output[0].clone(), xs);
            }
            "MatMul" => {
                let input0 = get(&node.input[0]);
                let input1 = get(&node.input[1]);
                let output = input0.matmul(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Reshape" => {
                let input0 = get(&node.input[0]);
                let input1: Tensor = get(&node.input[1]).clone();
                let shape: Vec<i64> = input1.try_into()?;
                let mut other_than_minus1 = 1i64;
                for &v in shape.iter() {
                    if v != -1 && v != 0 {
                        other_than_minus1 *= v
                    }
                }
                let shape_tensors: Vec<Tensor> = shape
                    .iter()
                    .enumerate()
                    .map(|(idx, &v)| match v {
                        -1 => input0.numel() / Tensor::from(other_than_minus1),
                        0 => input0.shape()[idx].clone(),
                        _ => Tensor::from(v),
                    })
                    .collect();
                let output = input0.reshape(shape_tensors)?;
                values.insert(node.output[0].clone(), output);
            }
            "LogSoftmax" => {
                let input = get(&node.input[0]);
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(-1);
                let output = input.ln_softmax([axis as i32])?;
                values.insert(node.output[0].clone(), output);
            }
            "Softmax" => {
                let input = get(&node.input[0]);
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => input.softmax([-1])?,
                    Some(&axis) => input.softmax([axis as i32])?,
                };
                values.insert(node.output[0].clone(), output);
            }
            "Transpose" => {
                let input = get(&node.input[0]);
                let output = match get_attr_opt::<[i64]>(node, "perm")? {
                    None => input.t(),
                    Some(perm) => {
                        let perm = perm.iter().map(|&v| v as i32).collect::<Vec<_>>();
                        input.permute(perm)?
                    }
                };
                values.insert(node.output[0].clone(), output);
            }
            "Dropout" => {
                let input = get(&node.input[0]).clone();
                values.insert(node.output[0].clone(), input.clone());
                if node.output.len() > 1 {
                    values.insert(
                        node.output[1].clone(),
                        Tensor::ones(input.resolve_shape(), DType::Bool),
                    );
                }
            }
            "Sqrt" => {
                let xs = get(&node.input[0]);
                let output = xs.sqrt();
                values.insert(node.output[0].clone(), output);
            }
            "Greater" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);

                let output = a.cmpgt(b)?;
                values.insert(node.output[0].clone(), output);
            }
            "Less" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);

                let output = a.cmplt(b)?;
                values.insert(node.output[0].clone(), output);
            }
            "Log" => {
                let a = get(&node.input[0]);
                let output = a.ln();
                values.insert(node.output[0].clone(), output);
            }
            "Min" => {
                let mut output = get(&node.input[0]).clone();
                for input in node.input.iter() {
                    let input = get(input);
                    output = output.minimum(input)?;
                }
                values.insert(node.output[0].clone(), output);
            }
            "Max" => {
                let mut output = get(&node.input[0]).clone();
                for input in node.input.iter().skip(1) {
                    let input = get(input);
                    output = output.maximum(input)?;
                }
                values.insert(node.output[0].clone(), output);
            }
            "Where" => {
                let cond = get(&node.input[0]);
                let a = get(&node.input[1]);
                let b = get(&node.input[2]);
                let output = cond.select(a, b)?;
                values.insert(node.output[0].clone(), output);
            }
            "Concat" => {
                let inputs: Vec<Value> = node
                    .input
                    .iter()
                    .map(|n| get(n.as_str()).clone())
                    .collect();
                let axis: i64 = *get_attr(node, "axis")?;
                if inputs.is_empty() {
                    panic!("empty concat")
                };
                let output = Tensor::cat(&inputs, axis as i32)?;
                values.insert(node.output[0].clone(), output);
            }
            "Abs" => {
                let input = get(&node.input[0]);
                let output = input.abs();
                values.insert(node.output[0].clone(), output);
            }
            "Cos" => {
                let input = get(&node.input[0]);
                let output = input.cos();
                values.insert(node.output[0].clone(), output);
            }
            "Sin" => {
                let input = get(&node.input[0]);
                let output = input.sin();
                values.insert(node.output[0].clone(), output);
            }
            "Neg" => {
                let input = get(&node.input[0]);
                let output = -input;
                values.insert(node.output[0].clone(), output);
            }
            "Tanh" => {
                let input = get(&node.input[0]);
                let output = input.tanh();
                values.insert(node.output[0].clone(), output);
            }
            "Sigmoid" => {
                let input = get(&node.input[0]);
                let output = input.sigmoid();
                values.insert(node.output[0].clone(), output);
            }
            "Gelu" => {
                let input = get(&node.input[0]);
                let output = input.gelu();
                values.insert(node.output[0].clone(), output);
            }
            "Relu" => {
                let input = get(&node.input[0]);
                let output = input.relu();
                values.insert(node.output[0].clone(), output);
            }
            "Constant" => {
                let attr = |name: &str| node.attribute.iter().find(|attr| attr.name == name);
                let output = if let Some(value) = attr("value") {
                    let t = value
                        .t
                        .as_ref()
                        .unwrap_or_else(|| panic!("'Constant.value' has no tensor for {}", node.name));
                    get_tensor(t, &node.name)?
                } else if let Some(value) = attr("value_float") {
                    Tensor::from(value.f)
                } else if let Some(value) = attr("value_floats") {
                    Tensor::from(value.floats.clone())
                } else if let Some(value) = attr("value_int") {
                    Tensor::from(value.i)
                } else if let Some(value) = attr("value_ints") {
                    Tensor::from(value.ints.clone())
                } else {
                    panic!(
                        "unsupported Constant variant for {} (string/sparse constants are not supported)",
                        node.name
                    )
                };
                values.insert(node.output[0].clone(), output);
            }
            "Cast" => {
                let input = get(&node.input[0]);
                let dt: i64 = *get_attr(node, "to")?;
                let dtype = match data_type_from_i32(dt as i32) {
                    Some(dt) => match dtype(dt) {
                        Some(dt) => dt,
                        None => {
                            panic!("unsupported 'to' value {dt:?} for cast {}", node.name)
                        }
                    },
                    None => {
                        panic!("unsupported 'to' value {dt:?} for cast {}", node.name)
                    }
                };
                let output = input.cast(dtype);
                values.insert(node.output[0].clone(), output);
            }
            "CumSum" => {
                let exclusive = get_attr_opt::<i64>(node, "exclusive")?
                    .copied()
                    .unwrap_or(0);
                let reverse = get_attr_opt::<i64>(node, "reverse")?.copied().unwrap_or(0);
                if exclusive != 0 {
                    panic!("only exclusive == 0 is supported in CumSum")
                }
                if reverse != 0 {
                    panic!("only reverse == 0 is supported in CumSum")
                }
                let input = get(&node.input[0]);
                let axis: i64 = get(&node.input[1]).clone().try_into()?;
                let output = input.cumsum(axis as i32)?;
                values.insert(node.output[0].clone(), output);
            }
            "Flatten" => {
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(1) as usize;
                let input = get(&node.input[0]);
                let shape = input.resolve_shape();
                let first_part: i64 = shape.iter().take(axis).product::<i64>();
                let total: i64 = shape.iter().product();
                let second = total / first_part;
                let output = input.reshape([Tensor::from(first_part), Tensor::from(second)])?;
                values.insert(node.output[0].clone(), output);
            }
            "Identity" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.clone());
            }
            "ReduceMean" => {
                let input = get(&node.input[0]);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop = get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied().unwrap_or(0);

                let axes: Vec<i32> = match get_opt(1) {
                    Some(axes) => {
                        let axes: Vec<i64> = axes.clone().try_into()?;
                        axes.into_iter().map(|a| a as i32).collect()
                    }
                    None => match get_attr_opt::<[i64]>(node, "axes")? {
                        Some(axes) => axes.iter().map(|a| *a as i32).collect(),
                        None if noop == 1 => vec![],
                        None => (0..input.rank()).map(|a| a as i32).collect(),
                    },
                };
                let output = if axes.is_empty() && noop == 1 {
                    input.clone()
                } else if keepdims == 1 {
                    input.mean_keepdim(axes)?
                } else {
                    input.mean(axes)?
                };
                values.insert(node.output[0].clone(), output);
            }
            "ReduceSum" => {
                let input = get(&node.input[0]);
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop_with_empty_axes = get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                    .copied()
                    .unwrap_or(0);

                let axes: Vec<i32> = match axes {
                    Some(axes) => {
                        let axes: Vec<i64> = axes.clone().try_into()?;
                        axes.into_iter().map(|x| x as i32).collect()
                    }
                    None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input.rank()).map(|a| a as i32).collect()
                        }
                    }
                };

                let output = if axes.is_empty() && noop_with_empty_axes == 1 {
                    input.clone()
                } else if keepdims == 1 {
                    input.sum_keepdim(axes)?
                } else {
                    input.sum(axes)?
                };

                values.insert(node.output[0].clone(), output);
            }
            "ReduceMax" => {
                let input = get(&node.input[0]);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop = get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied().unwrap_or(0);
                let axes: Vec<i32> = match get_opt(1) {
                    Some(axes) => {
                        let axes: Vec<i64> = axes.clone().try_into()?;
                        axes.into_iter().map(|a| a as i32).collect()
                    }
                    None => match get_attr_opt::<[i64]>(node, "axes")? {
                        Some(axes) => axes.iter().map(|a| *a as i32).collect(),
                        None if noop == 1 => vec![],
                        None => (0..input.rank()).map(|a| a as i32).collect(),
                    },
                };
                let output = if axes.is_empty() && noop == 1 {
                    input.clone()
                } else if keepdims == 1 {
                    input.max_keepdim(axes)?
                } else {
                    input.max(axes)?
                };
                values.insert(node.output[0].clone(), output);
            }
            "ReduceMin" => {
                let input = get(&node.input[0]);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop = get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied().unwrap_or(0);
                let axes: Vec<i32> = match get_opt(1) {
                    Some(axes) => {
                        let axes: Vec<i64> = axes.clone().try_into()?;
                        axes.into_iter().map(|a| a as i32).collect()
                    }
                    None => match get_attr_opt::<[i64]>(node, "axes")? {
                        Some(axes) => axes.iter().map(|a| *a as i32).collect(),
                        None if noop == 1 => vec![],
                        None => (0..input.rank()).map(|a| a as i32).collect(),
                    },
                };
                let output = if axes.is_empty() && noop == 1 {
                    input.clone()
                } else if keepdims == 1 {
                    input.min_keepdim(axes)?
                } else {
                    input.min(axes)?
                };
                values.insert(node.output[0].clone(), output);
            }
            "LeakyRelu" => {
                let input = get(&node.input[0]);
                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(0.01);
                let output = input.leaky_relu(alpha);
                values.insert(node.output[0].clone(), output);
            }
            "Gemm" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);

                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(1.0);
                let beta = get_attr_opt::<f32>(node, "beta")?.copied().unwrap_or(1.0);

                let alpha_t = Tensor::from(alpha);
                let beta_t = Tensor::from(beta);

                let trans_a = get_attr_opt::<i64>(node, "transA")?.copied().unwrap_or(0);
                let trans_b = get_attr_opt::<i64>(node, "transB")?.copied().unwrap_or(0);

                let a = if trans_a == 0 { a.clone() } else { a.t() };
                let b = if trans_b == 0 { b.clone() } else { b.t() };

                let ab = (a * alpha_t).matmul(&b)?;
                // C is optional in ONNX; without it Gemm is just alpha*A*B.
                let output = match get_opt(2) {
                    Some(c) => ab + c.clone() * beta_t,
                    None => ab,
                };
                values.insert(node.output[0].clone(), output);
            }
            "Clip" => {
                let input = get(&node.input[0]);
                let min = get_opt(1);
                let max = get_opt(2);
                let mut out = input.clone();
                if let Some(min) = min {
                    out = out.maximum(min)?;
                }
                if let Some(max) = max {
                    out = out.minimum(max)?;
                }
                // also handle attribute version (clip v6)
                if let Some(min_attr) = get_attr_opt::<f32>(node, "min")? {
                    out = out.maximum(Tensor::from(*min_attr))?;
                }
                if let Some(max_attr) = get_attr_opt::<f32>(node, "max")? {
                    out = out.minimum(Tensor::from(*max_attr))?;
                }
                values.insert(node.output[0].clone(), out);
            }
            "Squeeze" => {
                let input = get(&node.input[0]);
                // ONNX Squeeze: axes as attribute or second input
                let axes: Option<Vec<i64>> = if let Some(axes) = get_attr_opt::<[i64]>(node, "axes")? {
                    Some(axes.to_vec())
                } else if let Some(ax_tensor) = get_opt(1) {
                    let v: Vec<i64> = ax_tensor.clone().try_into()?;
                    Some(v)
                } else {
                    None
                };
                let output = if let Some(axes) = axes {
                    let axes_i32: Vec<i32> = axes.into_iter().map(|a| a as i32).collect();
                    input.squeeze(axes_i32)
                } else {
                    // squeeze all dims of size 1
                    let shape = input.resolve_shape();
                    let axes: Vec<i32> = shape.iter().enumerate().filter_map(|(i, &d)| if d == 1 { Some(i as i32) } else { None }).collect();
                    input.squeeze(axes)
                };
                values.insert(node.output[0].clone(), output);
            }
            "Unsqueeze" => {
                let input = get(&node.input[0]);
                let axes: Vec<i64> = if let Some(axes) = get_attr_opt::<[i64]>(node, "axes")? {
                    axes.to_vec()
                } else {
                    let t = get(&node.input[1]).clone();
                    t.try_into()?
                };
                let mut out = input.clone();
                let mut axes_sorted = axes;
                axes_sorted.sort();
                for &axis in axes_sorted.iter() {
                    out = out.unsqueeze(axis as i32)?;
                }
                values.insert(node.output[0].clone(), out);
            }
            "Shape" => {
                let input = get(&node.input[0]);
                let rank = input.rank() as i64;
                let start = get_attr_opt::<i64>(node, "start")?.copied().unwrap_or(0);
                let end = get_attr_opt::<i64>(node, "end")?.copied().unwrap_or(rank);
                let start = if start < 0 { start + rank } else { start }.clamp(0, rank);
                let end = if end < 0 { end + rank } else { end }.clamp(start, rank);
                let shape = input.shape();
                let shape = &shape[start as usize..end as usize];
                // shape() returns Vec<Tensor> each is scalar dim
                let out = if shape.is_empty() {
                    Tensor::from(Vec::<i64>::new())
                } else {
                    Tensor::stack(shape)?
                };
                values.insert(node.output[0].clone(), out);
            }
            "Size" => {
                let input = get(&node.input[0]);
                let n = input.numel();
                // numel is Tensor scalar, but ONNX Size is scalar i64 tensor
                values.insert(node.output[0].clone(), n);
            }
            "Slice" => {
                let data = get(&node.input[0]);
                // opset >= 10: starts/ends/axes/steps are inputs; older opsets
                // carry the same fields as attributes.
                let starts: Vec<i64> = if node.input.get(1).is_some_and(|s| !s.is_empty()) {
                    get(&node.input[1]).clone().try_into()?
                } else {
                    get_attr::<[i64]>(node, "starts")?.to_vec()
                };
                let ends: Vec<i64> = if node.input.get(2).is_some_and(|s| !s.is_empty()) {
                    get(&node.input[2]).clone().try_into()?
                } else {
                    get_attr::<[i64]>(node, "ends")?.to_vec()
                };
                let axes: Vec<i64> = if node.input.get(3).is_some_and(|s| !s.is_empty()) {
                    get(&node.input[3]).clone().try_into()?
                } else if let Some(axes) = get_attr_opt::<[i64]>(node, "axes")? {
                    axes.to_vec()
                } else {
                    (0..starts.len() as i64).collect()
                };
                let steps: Vec<i64> = if node.input.get(4).is_some_and(|s| !s.is_empty()) {
                    get(&node.input[4]).clone().try_into()?
                } else if let Some(steps) = get_attr_opt::<[i64]>(node, "steps")? {
                    steps.to_vec()
                } else {
                    vec![1; starts.len()]
                };
                assert!(
                    starts.len() == ends.len() && axes.len() == starts.len() && steps.len() == starts.len(),
                    "Slice starts/ends/axes/steps length mismatch {}/{}/{}/{} for {}",
                    starts.len(),
                    ends.len(),
                    axes.len(),
                    steps.len(),
                    node.name
                );
                let rank = data.rank();
                let mut out = data.clone();
                for i in 0..starts.len() {
                    let axis = axes[i];
                    let step = steps[i];
                    if step == 0 {
                        panic!("Slice step cannot be 0 for {}", node.name);
                    }
                    let axis = if axis < 0 { axis + rank } else { axis };
                    if axis < 0 || axis >= rank {
                        panic!("Slice axis {axis} out of range for rank {rank} for {}", node.name);
                    }
                    let dim = out.resolve_shape()[axis as usize];
                    if dim == 0 {
                        continue;
                    }
                    // Negative starts/ends count from the end. Saturating add
                    // keeps i64::MIN (the ONNX "to the beginning" sentinel)
                    // negative through the clamp below.
                    let mut start = starts[i];
                    let mut end = ends[i];
                    if start < 0 {
                        start = start.saturating_add(dim);
                    }
                    if end < 0 {
                        end = end.saturating_add(dim);
                    }
                    if step > 0 {
                        let start = start.clamp(0, dim);
                        let end = end.clamp(0, dim);
                        let len = (end - start).max(0);
                        out = out.narrow(axis as i32, Tensor::from(start), Tensor::from(len))?;
                    } else {
                        let start = start.clamp(0, dim - 1);
                        let end = end.clamp(-1, dim - 1);
                        let mut indices = Vec::new();
                        let mut v = start;
                        while v > end {
                            indices.push(v);
                            v += step;
                        }
                        out = out.index_select(axis as i32, Tensor::from(indices))?;
                    }
                }
                values.insert(node.output[0].clone(), out);
            }
            "Gather" => {
                let data = get(&node.input[0]);
                let indices = get(&node.input[1]);
                let rank = data.rank();
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                let axis = if axis < 0 { axis + rank } else { axis };
                assert!(
                    axis >= 0 && axis < rank,
                    "Gather axis {axis} out of range for rank {rank} for {}",
                    node.name
                );
                let axis = axis as usize;
                // ONNX allows any index rank: gather into the flattened index
                // list, then reshape to `data.shape[..axis] + indices.shape +
                // data.shape[axis+1..]`. `index_select` also wraps negative
                // indices.
                let data_shape = data.shape();
                let index_shape = indices.shape();
                let mut target = data_shape[..axis].to_vec();
                target.extend(index_shape);
                target.extend(data_shape[axis + 1..].iter().cloned());
                let flat = indices.reshape([indices.numel()])?;
                let out = data.index_select(axis as i32, flat)?.reshape(target)?;
                values.insert(node.output[0].clone(), out);
            }
            "GatherElements" => {
                let data = get(&node.input[0]);
                let indices = get(&node.input[1]);
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0) as i32;
                let out = data.gather(axis, indices.clone())?;
                values.insert(node.output[0].clone(), out);
            }
            "Expand" => {
                let input = get(&node.input[0]);
                let shape: Vec<i64> = get(&node.input[1]).clone().try_into()?;
                let shape_tensors: Vec<Tensor> = shape.into_iter().map(Tensor::from).collect();
                let out = input.expand(shape_tensors)?;
                values.insert(node.output[0].clone(), out);
            }
            "Split" => {
                let input = get(&node.input[0]);
                let rank = input.rank();
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                let axis = if axis < 0 { axis + rank } else { axis };
                assert!(
                    axis >= 0 && axis < rank,
                    "Split axis {axis} out of range for rank {rank} for {}",
                    node.name
                );
                let axis = axis as i32;
                let num_outputs = match get_attr_opt::<i64>(node, "num_outputs")?.copied().unwrap_or(0) {
                    0 => node.output.len(),
                    n => n as usize,
                };
                let split: Option<Vec<i64>> = if node.input.get(1).is_some_and(|s| !s.is_empty()) {
                    Some(get(&node.input[1]).clone().try_into()?)
                } else {
                    get_attr_opt::<[i64]>(node, "split")?.map(|a| a.to_vec())
                };
                let dim = input.resolve_shape()[axis as usize];
                let sizes: Vec<i64> = match split {
                    Some(sizes) => sizes,
                    None => {
                        // ONNX distributes the remainder over the leading
                        // outputs: each = dim / n, and the first dim % n get 1
                        // extra element.
                        let each = dim / num_outputs as i64;
                        let rem = dim % num_outputs as i64;
                        (0..num_outputs as i64)
                            .map(|i| each + i64::from(i < rem))
                            .collect()
                    }
                };
                assert_eq!(
                    sizes.iter().sum::<i64>(),
                    dim,
                    "Split sizes {:?} do not add up to axis {axis} length {dim} for {}",
                    sizes,
                    node.name
                );
                let mut offset = 0i64;
                let mut outs = Vec::new();
                for &sz in &sizes {
                    outs.push(input.narrow(axis, Tensor::from(offset), Tensor::from(sz))?);
                    offset += sz;
                }
                for (name, t) in node.output.iter().zip(outs.into_iter()) {
                    values.insert(name.clone(), t);
                }
                continue;
            }
            "Pad" => {
                let data = get(&node.input[0]);
                // opset >= 11: pads/constant_value/axes are inputs; older
                // opsets use the `pads` and `value` attributes.
                let pads: Vec<i64> = if node.input.get(1).is_some_and(|s| !s.is_empty()) {
                    get(&node.input[1]).clone().try_into()?
                } else {
                    get_attr::<[i64]>(node, "pads")?.to_vec()
                };
                let mode = get_attr_opt::<str>(node, "mode")?.unwrap_or("constant");
                if mode != "constant" {
                    panic!("Pad only supports constant mode, got {mode}");
                }
                let rank = data.rank() as usize;
                // opset 18: `axes` selects which dimensions `pads` applies to.
                let axes: Vec<i64> = if node.input.get(3).is_some_and(|s| !s.is_empty()) {
                    get(&node.input[3]).clone().try_into()?
                } else {
                    (0..rank as i64).collect()
                };
                if pads.len() != axes.len() * 2 {
                    panic!(
                        "Pad pads len {} != axes len {} * 2 for {}",
                        pads.len(),
                        axes.len(),
                        node.name
                    );
                }
                let mut padding = vec![(0i64, 0i64); rank];
                for (i, &axis) in axes.iter().enumerate() {
                    let axis = if axis < 0 { axis + rank as i64 } else { axis };
                    if axis < 0 || axis >= rank as i64 {
                        panic!("Pad axis {axis} out of range for rank {rank} for {}", node.name);
                    }
                    padding[axis as usize] = (pads[i], pads[i + axes.len()]);
                }
                // `constant_value` defaults to zero; `Tensor::pad` handles
                // negative padding (cropping) and dtype-checks the value.
                let value = match get_opt(2) {
                    Some(value) => value.clone(),
                    None => match get_attr_opt::<f32>(node, "value")? {
                        Some(&value) => Tensor::from(value).cast(data.dtype()),
                        None => Tensor::zeros(Vec::<Tensor>::new(), data.dtype()),
                    },
                };
                let out = data.pad(padding, value)?;
                values.insert(node.output[0].clone(), out);
            }
            "Erf" => {
                let input = get(&node.input[0]);
                let output = input.erf();
                values.insert(node.output[0].clone(), output);
            }
            "Ceil" => {
                let input = get(&node.input[0]);
                let output = input.ceil();
                values.insert(node.output[0].clone(), output);
            }
            "Floor" => {
                let input = get(&node.input[0]);
                let output = input.floor();
                values.insert(node.output[0].clone(), output);
            }
            "Range" => {
                let start: i64 = get(&node.input[0]).clone().try_into()?;
                let limit: i64 = get(&node.input[1]).clone().try_into()?;
                let delta: i64 = get(&node.input[2]).clone().try_into()?;
                let out = Tensor::arange(start, limit, delta)?;
                values.insert(node.output[0].clone(), out);
            }
            "ConstantOfShape" => {
                let shape: Vec<i64> = get(&node.input[0]).clone().try_into()?;
                let shape_tensors: Vec<Tensor> = shape.into_iter().map(Tensor::from).collect();
                let out = match get_attr_opt_owned::<Tensor>(node, "value")? {
                    Some(value) => value.expand(shape_tensors)?,
                    None => Tensor::zeros(shape_tensors, DType::F32),
                };
                values.insert(node.output[0].clone(), out);
            }
            "Tile" => {
                let input = get(&node.input[0]);
                let repeats: Vec<i64> = get(&node.input[1]).clone().try_into()?;
                values.insert(node.output[0].clone(), input.repeat(repeats)?);
            }
            "Trilu" => {
                let input = get(&node.input[0]);
                let k: i64 = match get_opt(1) {
                    Some(k) => k.clone().try_into()?,
                    None => get_attr_opt::<i64>(node, "k")?.copied().unwrap_or(0),
                };
                let upper = get_attr_opt::<i64>(node, "upper")?.copied().unwrap_or(1);
                let out = if upper != 0 { input.triu(k)? } else { input.tril(k)? };
                values.insert(node.output[0].clone(), out);
            }
            "ArgMax" | "ArgMin" => {
                let input = get(&node.input[0]);
                let rank = input.rank();
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                let axis = if axis < 0 { axis + rank } else { axis };
                assert!(
                    axis >= 0 && axis < rank,
                    "{} axis {axis} out of range for rank {rank} for {}",
                    node.op_type,
                    node.name
                );
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let select_last = get_attr_opt::<i64>(node, "select_last_index")?.copied().unwrap_or(0);
                let base = if node.op_type == "ArgMin" { -input.clone() } else { input.clone() };
                let mut idx = if select_last != 0 {
                    let dim = base.resolve_shape()[axis as usize];
                    let flipped = base.flip([axis as i32])?;
                    dim - 1 - flipped.argmax_axis(axis as i32)?
                } else {
                    base.argmax_axis(axis as i32)?
                };
                if keepdims != 0 {
                    idx = idx.unsqueeze(axis as i32)?;
                }
                values.insert(node.output[0].clone(), idx.cast(DType::I64));
            }
            "ReduceProd" | "ReduceL1" | "ReduceL2" | "ReduceLogSum" | "ReduceLogSumExp" => {
                let input = get(&node.input[0]);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop = get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied().unwrap_or(0);
                let axes: Vec<i32> = match get_opt(1) {
                    Some(axes) => {
                        let axes: Vec<i64> = axes.clone().try_into()?;
                        axes.into_iter().map(|a| a as i32).collect()
                    }
                    None => match get_attr_opt::<[i64]>(node, "axes")? {
                        Some(axes) => axes.iter().map(|a| *a as i32).collect(),
                        None if noop == 1 => vec![],
                        None => (0..input.rank()).map(|a| a as i32).collect(),
                    },
                };
                let out = if axes.is_empty() && noop == 1 {
                    input.clone()
                } else {
                    let data = match node.op_type.as_str() {
                        "ReduceL1" => input.abs(),
                        "ReduceL2" => input.square(),
                        "ReduceLogSumExp" => input.exp(),
                        _ => input.clone(),
                    };
                    let summed = if keepdims == 1 {
                        data.sum_keepdim(axes)?
                    } else {
                        data.sum(axes)?
                    };
                    match node.op_type.as_str() {
                        "ReduceL2" => summed.sqrt(),
                        "ReduceLogSum" | "ReduceLogSumExp" => summed.ln(),
                        _ => summed,
                    }
                };
                values.insert(node.output[0].clone(), out);
            }
            "Elu" => {
                let input = get(&node.input[0]);
                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(1.0);
                values.insert(node.output[0].clone(), input.elu(alpha));
            }
            "Selu" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.selu());
            }
            "HardSigmoid" => {
                let input = get(&node.input[0]);
                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(0.2);
                let beta = get_attr_opt::<f32>(node, "beta")?.copied().unwrap_or(0.5);
                let out = input.hard_sigmoid();
                let out = if (alpha - 0.2).abs() < f32::EPSILON && (beta - 0.5).abs() < f32::EPSILON {
                    out
                } else {
                    let t = get(&node.input[0]).clone() * alpha + beta;
                    t.maximum(0.0f32)?.minimum(1.0f32)?
                };
                values.insert(node.output[0].clone(), out);
            }
            "HardSwish" => {
                let input = get(&node.input[0]);
                let gate = (input.clone() * (1.0f32 / 6.0f32) + 0.5f32)
                    .maximum(0.0f32)?
                    .minimum(1.0f32)?;
                values.insert(node.output[0].clone(), input.clone() * gate);
            }
            "Softplus" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.softplus(1.0f32, 20.0f32));
            }
            "Softsign" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.clone() / (input.abs() + 1.0f32));
            }
            "Mish" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.mish());
            }
            "Sign" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.sign());
            }
            "Round" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.round());
            }
            "Reciprocal" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.reciprocal());
            }
            "PRelu" => {
                let input = get(&node.input[0]);
                let slope = get(&node.input[1]);
                let out = input.cmpgt(0.0f32)?.select(input.clone(), input.clone() * slope.clone())?;
                values.insert(node.output[0].clone(), out);
            }
            "ThresholdedRelu" => {
                let input = get(&node.input[0]);
                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(1.0);
                let out = input.cmpgt(alpha)?.select(input.clone(), 0.0f32)?;
                values.insert(node.output[0].clone(), out);
            }
            "IsInf" => {
                let input = get(&node.input[0]);
                let pos = get_attr_opt::<i64>(node, "detect_positive")?.copied().unwrap_or(1);
                let neg = get_attr_opt::<i64>(node, "detect_negative")?.copied().unwrap_or(1);
                let out = match (pos != 0, neg != 0) {
                    (true, true) => input.isinf(),
                    (true, false) => input.equal(f32::INFINITY)?,
                    (false, true) => input.equal(f32::NEG_INFINITY)?,
                    (false, false) => input.isnan().select(true, false)?,
                };
                values.insert(node.output[0].clone(), out);
            }
            "IsNaN" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.isnan());
            }
            "GreaterOrEqual" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);
                values.insert(node.output[0].clone(), a.cmpge(b)?);
            }
            "LessOrEqual" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);
                values.insert(node.output[0].clone(), !a.cmpgt(b)?);
            }
            "And" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);
                values.insert(node.output[0].clone(), a.logical_and(b)?);
            }
            "Or" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);
                values.insert(node.output[0].clone(), a.logical_or(b)?);
            }
            "Scatter" | "ScatterElements" => {
                let data = get(&node.input[0]);
                let indices = get(&node.input[1]);
                let updates = get(&node.input[2]);
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0) as i32;
                let reduction = get_attr_opt::<str>(node, "reduction")?.unwrap_or("none");
                let out = match reduction {
                    "none" => data.scatter(axis, indices.clone(), updates.clone())?,
                    "add" => data.scatter_add(axis, indices.clone(), updates.clone())?,
                    "mul" => data.scatter_mul(axis, indices.clone(), updates.clone())?,
                    other => panic!("unsupported ScatterElements reduction {other} for {}", node.name),
                };
                values.insert(node.output[0].clone(), out);
            }
            "Conv" => {
                let x = get(&node.input[0]);
                let w = get(&node.input[1]);
                let bias = get_opt(2);
                let rank = x.rank() as usize;
                let spatial = rank - 2;
                let kernel = w.resolve_shape()[2..].to_vec();
                let strides: Vec<i64> = get_attr_opt::<[i64]>(node, "strides")?
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| vec![1; spatial]);
                let dilations: Vec<i64> = get_attr_opt::<[i64]>(node, "dilations")?
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| vec![1; spatial]);
                let pads: Vec<i64> = get_attr_opt::<[i64]>(node, "pads")?
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| vec![0; spatial * 2]);
                let groups = get_attr_opt::<i64>(node, "group")?.copied().unwrap_or(1) as u64;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?.unwrap_or("NOTSET");
                assert_eq!(strides.len(), spatial, "Conv strides length for {}", node.name);
                assert_eq!(dilations.len(), spatial, "Conv dilations length for {}", node.name);
                let in_shape = x.resolve_shape();
                // ONNX pads are [begin spatial..., end spatial...]; zyx conv
                // takes interleaved (left, right) pairs.
                let mut pad_pairs = Vec::with_capacity(spatial * 2);
                for i in 0..spatial {
                    let (left, right) = match auto_pad {
                        "VALID" => (0, 0),
                        "SAME_UPPER" | "SAME_LOWER" => {
                            let k = kernel[i] as i64;
                            let s = strides[i];
                            let d = dilations[i];
                            let input = in_shape[2 + i] as i64;
                            let out = (input + s - 1) / s;
                            let total = ((out - 1) * s + (k - 1) * d + 1 - input).max(0);
                            if auto_pad == "SAME_UPPER" {
                                (total / 2, total - total / 2)
                            } else {
                                (total - total / 2, total / 2)
                            }
                        }
                        _ => (pads[i], pads[i + spatial]),
                    };
                    pad_pairs.push(left);
                    pad_pairs.push(right);
                }
                let out = x.conv(w, bias, groups, strides, dilations, pad_pairs)?;
                values.insert(node.output[0].clone(), out);
            }
            "MaxPool" => {
                let x = get(&node.input[0]);
                if node.output.len() > 1 {
                    panic!("MaxPool indices output is not supported for {}", node.name);
                }
                let kernel: Vec<i64> = get_attr::<[i64]>(node, "kernel_shape")?.to_vec();
                let spatial = kernel.len();
                let strides: Vec<i64> = get_attr_opt::<[i64]>(node, "strides")?
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| vec![1; spatial]);
                let dilations: Vec<i64> = get_attr_opt::<[i64]>(node, "dilations")?
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| vec![1; spatial]);
                let pads: Vec<i64> = get_attr_opt::<[i64]>(node, "pads")?
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| vec![0; spatial * 2]);
                let ceil_mode = get_attr_opt::<i64>(node, "ceil_mode")?.copied().unwrap_or(0);
                if ceil_mode != 0 {
                    panic!("MaxPool ceil_mode is not supported for {}", node.name);
                }
                let padding: Vec<(i64, i64)> =
                    (0..spatial).map(|i| (pads[i], pads[i + spatial])).collect();
                let out = x.max_pool(kernel, strides, dilations, padding, false, false)?;
                values.insert(node.output[0].clone(), out);
            }
            "AveragePool" => {
                let x = get(&node.input[0]);
                let kernel: Vec<i64> = get_attr::<[i64]>(node, "kernel_shape")?.to_vec();
                let spatial = kernel.len();
                let strides: Vec<i64> = get_attr_opt::<[i64]>(node, "strides")?
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| vec![1; spatial]);
                let dilations: Vec<i64> = get_attr_opt::<[i64]>(node, "dilations")?
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| vec![1; spatial]);
                let pads: Vec<i64> = get_attr_opt::<[i64]>(node, "pads")?
                    .map(|a| a.to_vec())
                    .unwrap_or_else(|| vec![0; spatial * 2]);
                let ceil_mode = get_attr_opt::<i64>(node, "ceil_mode")?.copied().unwrap_or(0);
                let count_include_pad = get_attr_opt::<i64>(node, "count_include_pad")?.copied().unwrap_or(0);
                if ceil_mode != 0 {
                    panic!("AveragePool ceil_mode is not supported for {}", node.name);
                }
                if count_include_pad == 0 && pads.iter().any(|&p| p != 0) {
                    panic!(
                        "AveragePool with padding and count_include_pad=0 is not supported for {}",
                        node.name
                    );
                }
                let padding: Vec<(i64, i64)> =
                    (0..spatial).map(|i| (pads[i], pads[i + spatial])).collect();
                let padded = x.pad(padding, 0.0f32)?;
                let pooled = padded.pool(kernel.clone(), strides, dilations)?;
                let axes: Vec<i32> = (-(spatial as i32)..0).collect();
                let volume: f32 = kernel.iter().product::<i64>() as f32;
                let out = pooled.sum(axes)? / volume;
                values.insert(node.output[0].clone(), out);
            }
            "GlobalAveragePool" | "GlobalMaxPool" => {
                let x = get(&node.input[0]);
                let rank = x.rank() as i32;
                let axes: Vec<i32> = (2..rank).collect();
                let out = if node.op_type == "GlobalAveragePool" {
                    x.mean_keepdim(axes)?
                } else {
                    x.max_keepdim(axes)?
                };
                values.insert(node.output[0].clone(), out);
            }
            "BatchNormalization" => {
                let x = get(&node.input[0]);
                let scale = get(&node.input[1]);
                let bias = get(&node.input[2]);
                let mean = get(&node.input[3]);
                let var = get(&node.input[4]);
                let eps = get_attr_opt::<f32>(node, "epsilon")?.copied().unwrap_or(1e-5);
                let training = get_attr_opt::<i64>(node, "training_mode")?.copied().unwrap_or(0);
                if training != 0 {
                    panic!("BatchNormalization training mode is not supported for {}", node.name);
                }
                let rank = x.rank() as usize;
                let mut shape: Vec<Tensor> = (0..rank).map(|_| Tensor::from(1i64)).collect();
                shape[1] = scale.shape()[0].clone();
                let out = (x.clone() - mean.reshape(shape.clone())?)
                    / (var.reshape(shape.clone())? + eps).sqrt()
                    * scale.reshape(shape.clone())?
                    + bias.reshape(shape)?;
                values.insert(node.output[0].clone(), out);
            }
            "InstanceNormalization" => {
                let x = get(&node.input[0]);
                let scale = get(&node.input[1]);
                let bias = get(&node.input[2]);
                let eps = get_attr_opt::<f32>(node, "epsilon")?.copied().unwrap_or(1e-5);
                let rank = x.rank() as usize;
                let axes: Vec<i32> = (2..rank as i32).collect();
                let mean = x.mean_keepdim(axes.clone())?;
                let var = (x.clone() - mean.clone()).square().mean_keepdim(axes)?;
                let mut shape: Vec<Tensor> = (0..rank).map(|_| Tensor::from(1i64)).collect();
                shape[1] = scale.shape()[0].clone();
                let out = (x.clone() - mean) / (var + eps).sqrt()
                    * scale.reshape(shape.clone())?
                    + bias.reshape(shape)?;
                values.insert(node.output[0].clone(), out);
            }
            "LayerNormalization" => {
                let x = get(&node.input[0]);
                let scale = get(&node.input[1]);
                let bias = get_opt(2);
                let rank = x.rank() as i64;
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(-1);
                let axis = if axis < 0 { axis + rank } else { axis };
                assert!(
                    axis >= 0 && axis < rank,
                    "LayerNormalization axis {axis} out of range for rank {rank} for {}",
                    node.name
                );
                let eps = get_attr_opt::<f32>(node, "epsilon")?.copied().unwrap_or(1e-5);
                let axes: Vec<i32> = (axis as i32..rank as i32).collect();
                let mean = x.mean_keepdim(axes.clone())?;
                let var = (x.clone() - mean.clone()).square().mean_keepdim(axes)?;
                let invstd = (var.clone() + eps).sqrt().reciprocal();
                let mut out = (x.clone() - mean.clone()) * invstd.clone() * scale.clone();
                if let Some(bias) = bias {
                    out = out + bias.clone();
                }
                values.insert(node.output[0].clone(), out);
                if node.output.len() > 1 {
                    values.insert(node.output[1].clone(), mean);
                }
                if node.output.len() > 2 {
                    values.insert(node.output[2].clone(), invstd);
                }
            }
            "Sum" | "Mean" => {
                let mut output = get(&node.input[0]).clone();
                for name in node.input.iter().skip(1) {
                    output = output + get(name).clone();
                }
                if node.op_type == "Mean" {
                    output = output / (node.input.len() as f32);
                }
                values.insert(node.output[0].clone(), output);
            }
            "DepthToSpace" => {
                let input = get(&node.input[0]);
                let block = get_attr_opt::<i64>(node, "blocksize")?.copied().unwrap_or(1);
                let mode = get_attr_opt::<str>(node, "mode")?.unwrap_or("DCR");
                let shape = input.resolve_shape();
                let [n, c, h, w] = shape[..4] else {
                    panic!("DepthToSpace requires rank 4 for {}", node.name);
                };
                let out_c = c / (block * block);
                let out = if mode == "CRD" {
                    input
                        .reshape([n, out_c, block, block, h, w])?
                        .permute([0, 1, 4, 2, 5, 3])?
                        .reshape([n, out_c, h * block, w * block])?
                } else {
                    input
                        .reshape([n, block, block, out_c, h, w])?
                        .permute([0, 3, 4, 1, 5, 2])?
                        .reshape([n, out_c, h * block, w * block])?
                };
                values.insert(node.output[0].clone(), out);
            }
            "SpaceToDepth" => {
                let input = get(&node.input[0]);
                let block = get_attr_opt::<i64>(node, "blocksize")?.copied().unwrap_or(1);
                let shape = input.resolve_shape();
                let [n, c, h, w] = shape[..4] else {
                    panic!("SpaceToDepth requires rank 4 for {}", node.name);
                };
                let out = input
                    .reshape([n, c, h / block, block, w / block, block])?
                    .permute([0, 3, 5, 1, 2, 4])?
                    .reshape([n, c * block * block, h / block, w / block])?;
                values.insert(node.output[0].clone(), out);
            }
            "MeanVarianceNormalization" => {
                let input = get(&node.input[0]);
                let axes: Vec<i32> = get_attr_opt::<[i64]>(node, "axes")?
                    .map(|a| a.iter().map(|&x| x as i32).collect())
                    .unwrap_or_else(|| vec![0, 2, 3]);
                let mean = input.mean_keepdim(axes.clone())?;
                let var = input.var_correction(axes, 0)?;
                let out = (input.clone() - mean) / (var + 1e-9).sqrt();
                values.insert(node.output[0].clone(), out);
            }
            "LpNormalization" => {
                let input = get(&node.input[0]);
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(-1) as i32;
                let p = get_attr_opt::<i64>(node, "p")?.copied().unwrap_or(2);
                let norm = if p == 1 {
                    input.abs().sum_keepdim([axis])?
                } else {
                    input.square().sum_keepdim([axis])?.sqrt()
                };
                values.insert(node.output[0].clone(), input / norm);
            }
            "Xor" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);
                let out = a.equal(b)?.select(false, true)?;
                values.insert(node.output[0].clone(), out);
            }
            op_type => panic!("unsupported op_type {op_type} for op {node:?}"),
        }
    }
    Ok(())
}

fn broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>, ZyxError> {
    let (longest, shortest) = if shape_a.len() > shape_b.len() {
        (shape_a, shape_b)
    } else {
        (shape_b, shape_a)
    };
    let diff = longest.len() - shortest.len();
    let mut target_shape = longest[0..diff].to_vec();
    for (dim1, dim2) in longest[diff..].iter().zip(shortest.iter()) {
        if *dim1 == *dim2 || *dim2 == 1 || *dim1 == 1 {
            target_shape.push(usize::max(*dim1, *dim2));
        } else {
            panic!(
                "Expand: incompatible shapes for broadcast, {:?} and {:?}",
                shape_a,
                shape_b
            );
        }
    }
    Ok(target_shape)
}

fn broadcast_shape_from_many(shapes: &[&[usize]]) -> Result<Vec<usize>, ZyxError> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }
    let mut shape_out = shapes[0].to_vec();
    for shape in shapes[1..].iter() {
        shape_out = broadcast_shape(&shape_out, shape)?;
    }
    Ok(shape_out)
}

// ---------------------------------------------------------------------------
// OnnxModel wrapper using FrozenTape
// ---------------------------------------------------------------------------

/// ONNX model backed by a frozen zyx tape.
///
/// Load an ONNX file, trace it once into a `Tape`, freeze it, and then
/// replay it for every inference via `FrozenTape::replay`.  Inputs and
/// outputs are addressed by their ONNX names (as in `onnxruntime`).
pub struct OnnxModel {
    frozen: FrozenTape,
    input_names: Vec<String>,
    // (input index, dim index) for each symbolic DimParam, in freeze order.
    dim_vars: Vec<(usize, usize)>,
    output_names: Vec<String>,
}

impl OnnxModel {
    /// Load an ONNX model from a file and freeze it.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ZyxError> {
        let model = crate::read_file(path)?;
        Self::from_model(&model)
    }

    /// Convenience alias for `load` — `OnnxModel::from_file("my_model.onnx")`.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ZyxError> {
        Self::load(path)
    }

    /// Freeze an already-decoded `ModelProto`.
    pub fn from_model(model: &crate::onnx::ModelProto) -> Result<Self, ZyxError> {
        let graph = model
            .graph
            .as_ref()
            .ok_or_else(|| ZyxError::ParseError("model has no graph".to_string().into()))?;

        // Inputs that are not initializers are model inputs.
        let initializer_names: HashSet<String> =
            graph.initializer.iter().map(|t| t.name.clone()).collect();

        let mut input_names = Vec::new();
        let mut input_infos = Vec::new(); // (name, dtype, shape_tensors)
        // One variable per distinct DimParam name, shared by every input dim
        // that uses it; the concrete value is bound at replay.
        let mut dim_params: HashMap<String, Tensor> = HashMap::new();
        let mut dim_tensors: Vec<Tensor> = Vec::new();
        let mut dim_vars: Vec<(usize, usize)> = Vec::new();
        for inp in &graph.input {
            if initializer_names.contains(&inp.name) {
                continue;
            }
            // Must have a tensor type to create a placeholder.
            let Some(ty) = inp.r#type.as_ref() else {
                continue;
            };
            let Some(onnx::type_proto::Value::TensorType(tt)) = &ty.value else {
                continue;
            };
            let dtype = match data_type_from_i32(tt.elem_type).and_then(dtype) {
                Some(d) => d,
                None => continue,
            };
            let shape: Vec<Tensor> = match &tt.shape {
                None => vec![],
                Some(s) => s
                    .dim
                    .iter()
                    .enumerate()
                    .map(|(dim_idx, d)| match &d.value {
                        Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => Tensor::from(*v),
                        Some(onnx::tensor_shape_proto::dimension::Value::DimParam(p)) => {
                            let var = match dim_params.get(p) {
                                Some(var) => var.clone(),
                                None => {
                                    let var = Tensor::variable(1i64);
                                    dim_params.insert(p.clone(), var.clone());
                                    dim_tensors.push(var.clone());
                                    var
                                }
                            };
                            dim_vars.push((input_infos.len(), dim_idx));
                            var
                        }
                        None => Tensor::from(1i64),
                    })
                    .collect(),
            };
            input_infos.push((inp.name.clone(), dtype, shape));
            input_names.push(inp.name.clone());
        }

        let output_names: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();

        // Build tape with placeholders for each model input.
        // Use host-backed Leaf tensors (not `Tensor::zeros` which is an
        // Eager Expand) so the leaf is a buffer-backed Leaf that can be
        // rebound at replay time.  `Tensor::zeros` creates an Eager Expand
        // of a Constant, which has no valid producer path for the frozen
        // graph (see `test_tape_add_with_zeros_placeholder`).
        let tape = Tape::empty();
        let mut values: HashMap<String, Tensor> = HashMap::new();
        let mut input_tensors: Vec<Tensor> = Vec::new();
        for (name, dtype, shape) in &input_infos {
            let placeholder = if shape.is_empty() {
                match dtype {
                    DType::F32 => Tensor::from(0.0f32),
                    DType::F64 => Tensor::from(0.0f64),
                    DType::I64 => Tensor::from(0i64),
                    DType::I32 => Tensor::from(0i32),
                    DType::U8 => Tensor::from(0u8),
                    _ => Tensor::zeros(Vec::<Tensor>::new(), *dtype),
                }
            } else {
                let shape_concrete: Vec<i64> =
                    shape.iter().map(|t| t.item::<i64>()).collect();
                let n: usize = shape_concrete.iter().map(|&x| x as usize).product();
                let ph = match dtype {
                    DType::F32 => Tensor::from(vec![0.0f32; n]).reshape(shape.clone())?,
                    DType::F64 => Tensor::from(vec![0.0f64; n]).reshape(shape.clone())?,
                    DType::I64 => Tensor::from(vec![0i64; n]).reshape(shape.clone())?,
                    DType::I32 => Tensor::from(vec![0i32; n])
                        .reshape(shape.clone())?
                        .cast(DType::I64),
                    DType::U8 => Tensor::from(vec![0u8; n]).reshape(shape.clone())?,
                    _ => Tensor::zeros(shape.clone(), *dtype),
                };
                ph
            };
            tape.add(&placeholder)?;
            input_tensors.push(placeholder.clone());
            values.insert(name.clone(), placeholder);
        }
        // Symbolic dims are bound at replay together with the data inputs.
        for var in &dim_tensors {
            input_tensors.push(var.clone());
        }

        // Evaluate the graph symbolically (under the tape). This inserts
        // initializers, validates inputs, and creates graph nodes for every
        // ONNX op. Because at least one input is a graph tensor, all
        // downstream ops become graph nodes.
        let outputs_map = simple_eval_(graph, &mut values)?;

        // Collect output tensors (they are graph tensors) and freeze.
        let mut outputs = Vec::new();
        for name in &output_names {
            let t = outputs_map
                .get(name)
                .unwrap_or_else(|| panic!("output {} not found after eval", name))
                .clone();
            outputs.push(t);
        }

        let frozen = tape.freeze(input_tensors.iter(), outputs.iter())?;

        Ok(Self {
            frozen,
            input_names,
            dim_vars,
            output_names,
        })
    }

    /// Run inference.
    ///
    /// `inputs` maps ONNX input names (e.g. `"x"`) to `Tensor`s. Every model
    /// input must be present; extra entries are ignored.
    pub fn run(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>, ZyxError> {
        let mut ordered = Vec::new();
        let mut shapes = Vec::new();
        for name in &self.input_names {
            let t = inputs
                .get(name)
                .unwrap_or_else(|| panic!("missing input {}", name))
                .clone();
            shapes.push(t.resolve_shape());
            ordered.push(t);
        }
        for &(input_idx, dim_idx) in &self.dim_vars {
            ordered.push(Tensor::variable(shapes[input_idx][dim_idx]));
        }
        let outs = self.frozen.replay(ordered.iter())?;
        let mut map = HashMap::new();
        for (name, t) in self.output_names.iter().zip(outs.into_iter()) {
            map.insert(name.clone(), t);
        }
        Ok(map)
    }

    /// ONNX input names in tape order.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// ONNX output names.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}
