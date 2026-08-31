// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::onnx::attribute_proto::AttributeType;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::{self, GraphProto};
use std::collections::{HashMap, HashSet};
use zyx::{DType, FrozenTape, Tape, Tensor, ZyxError};

pub type Value = Tensor;

pub fn dtype(dt: DataType) -> Option<DType> {
    match dt {
        DataType::Uint8 => Some(DType::U8),
        DataType::Uint32 => Some(DType::U32),
        DataType::Int64 => Some(DType::I64),
        DataType::Float => Some(DType::F32),
        DataType::Double => Some(DType::F64),
        DataType::Bool => Some(DType::U8),
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
    match data_type_from_i32(t.data_type) {
        Some(DataType::Int32) => {
            if t.int32_data.is_empty() {
                let data: Vec<i64> = t
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect();
                let base = Tensor::from(data);
                if dims.is_empty() {
                    Ok(base)
                } else {
                    base.reshape(dims)
                }
            } else {
                let data = t.int32_data.iter().map(|v| *v as i64).collect::<Vec<_>>();
                let base = Tensor::from(data);
                if dims.is_empty() {
                    Ok(base)
                } else {
                    base.reshape(dims)
                }
            }
        }
        Some(dt) => match dtype(dt) {
            Some(dtype) => {
                if dtype == DType::F32 && !t.float_data.is_empty() {
                    let base = Tensor::from(t.float_data.clone());
                    if dims.is_empty() { Ok(base) } else { base.reshape(dims) }
                } else if dtype == DType::F64 && !t.double_data.is_empty() {
                    let base = Tensor::from(t.double_data.clone());
                    if dims.is_empty() { Ok(base) } else { base.reshape(dims) }
                } else if dtype == DType::I64 && !t.int64_data.is_empty() {
                    let base = Tensor::from(t.int64_data.clone());
                    if dims.is_empty() { Ok(base) } else { base.reshape(dims) }
                } else {
                    // raw_data is little-endian bytes for `dtype`; interpret
                    // without `bitcast` (which requires equal bit widths for
                    // the current zyx custom kernel).
                    let base = match dtype {
                        DType::F32 => {
                            let v: Vec<f32> = t
                                .raw_data
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                                .collect();
                            Tensor::from(v)
                        }
                        DType::F64 => {
                            let v: Vec<f64> = t
                                .raw_data
                                .chunks_exact(8)
                                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                                .collect();
                            Tensor::from(v)
                        }
                        DType::I64 => {
                            let v: Vec<i64> = t
                                .raw_data
                                .chunks_exact(8)
                                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                                .collect();
                            Tensor::from(v)
                        }
                        DType::I32 => {
                            let v: Vec<i64> = t
                                .raw_data
                                .chunks_exact(4)
                                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                                .collect();
                            Tensor::from(v)
                        }
                        DType::U8 => Tensor::from(t.raw_data.clone()),
                        _ => panic!("unsupported raw_data dtype {dtype:?}"),
                    };
                    if dims.is_empty() { Ok(base) } else { base.reshape(dims) }
                }
            }
            None => {
                panic!("unsupported 'value' data-type {dt:?} for {name}")
            }
        },
        None => {
            panic!("unsupported 'value' data-type {} for {name}", t.data_type,)
        }
    }
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
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => input.softmax([-1])?,
                    Some(&axis) => {
                        input.ln_softmax([axis as i32])?
                    }
                };
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
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.clone());
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
                let output = cond.where_(a, b)?;
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
                let value = match node.attribute.iter().find(|attr| attr.name == "value") {
                    None => {
                        panic!("cannot find 'value' attr in 'Constant' for {}", node.name)
                    }
                    Some(value) => value,
                };
                let output = match value.r#type {
                    x if x == AttributeType::Tensor as i32 => {
                        let t = value.t.as_ref().unwrap();
                        get_tensor(t, &node.name)?
                    }
                    rtype => panic!("unsupported 'value' type {rtype:?} for {}", node.name),
                };
                values.insert(node.output[0].clone(), output);
            }
            "Cast" => {
                let input = get(&node.input[0]);
                let dt: i64 = *get_attr(node, "to")?;
                let dtype = match data_type_from_i32(dt as i32) {
                    Some(DataType::Int32) => DType::I64,
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
                let axes = get_attr_opt::<[i64]>(node, "axes")?;
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);

                let n_dims = input.rank() as i64;

                let axes: Vec<i32> = if let Some(axes) = axes {
                    axes.iter().map(|a| *a as i32).collect()
                } else {
                    (0..n_dims).map(|a| a as i32).collect()
                };
                let output = if keepdims == 1 {
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
                let axes = get_attr_opt::<[i64]>(node, "axes")?;
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let axes: Vec<i32> = if let Some(axes) = axes {
                    axes.iter().map(|a| *a as i32).collect()
                } else {
                    (0..input.rank()).map(|a| a as i32).collect()
                };
                let output = if keepdims == 1 {
                    input.max_keepdim(axes)?
                } else {
                    input.max(axes)?
                };
                values.insert(node.output[0].clone(), output);
            }
            "ReduceMin" => {
                let input = get(&node.input[0]);
                let axes = get_attr_opt::<[i64]>(node, "axes")?;
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let axes: Vec<i32> = if let Some(axes) = axes {
                    axes.iter().map(|a| *a as i32).collect()
                } else {
                    (0..input.rank()).map(|a| a as i32).collect()
                };
                let output = if keepdims == 1 {
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
                let c = get(&node.input[2]);

                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(1.0);
                let beta = get_attr_opt::<f32>(node, "beta")?.copied().unwrap_or(1.0);

                let alpha_t = Tensor::from(alpha);
                let beta_t = Tensor::from(beta);

                let trans_a = get_attr_opt::<i64>(node, "transA")?.copied().unwrap_or(0);
                let trans_b = get_attr_opt::<i64>(node, "transB")?.copied().unwrap_or(0);

                let a = if trans_a == 0 { a.clone() } else { a.t() };
                let b = if trans_b == 0 { b.clone() } else { b.t() };

                let output = (a * alpha_t).matmul(&b)? + c.clone() * beta_t;
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
                let shape = input.shape();
                // shape() returns Vec<Tensor> each is scalar dim
                let out = if shape.is_empty() {
                    Tensor::from(Vec::<i64>::new())
                } else {
                    Tensor::stack(&shape)?
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
                let starts: Vec<i64> = get(&node.input[1]).clone().try_into()?;
                let ends: Vec<i64> = get(&node.input[2]).clone().try_into()?;
                let axes: Option<Vec<i64>> = if node.input.len() > 3 {
                    let v: Vec<i64> = get(&node.input[3]).clone().try_into()?;
                    Some(v)
                } else {
                    get_attr_opt::<[i64]>(node, "axes")?.map(|a| a.to_vec())
                };
                let steps: Option<Vec<i64>> = if node.input.len() > 4 {
                    let v: Vec<i64> = get(&node.input[4]).clone().try_into()?;
                    Some(v)
                } else {
                    None
                };
                let axes = axes.unwrap_or_else(|| (0..starts.len() as i64).collect());
                let steps = steps.unwrap_or_else(|| vec![1; starts.len()]);
                let mut out = data.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    let start = starts[i];
                    let end = ends[i];
                    let step = steps[i];
                    if step != 1 {
                        panic!("Slice with step != 1 not supported");
                    }
                    // use narrow for step 1: start..end
                    let axis_i32 = axis as i32;
                    let rank = out.rank();
                    let axis_usize = if axis_i32 < 0 { (rank + axis_i32 as i64) as usize } else { axis_i32 as usize };
                    let dim = out.resolve_shape()[axis_usize];
                    let s = if start < 0 { dim + start } else { start };
                    let e = if end < 0 { dim + end } else { end };
                    let len = (e - s).max(0) as i64;
                    out = out.narrow(axis_i32, Tensor::from(s), Tensor::from(len))?;
                }
                values.insert(node.output[0].clone(), out);
            }
            "Gather" => {
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
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0) as i32;
                let split: Option<Vec<i64>> = if node.input.len() > 1 {
                    let v: Vec<i64> = get(&node.input[1]).clone().try_into()?;
                    Some(v)
                } else {
                    get_attr_opt::<[i64]>(node, "split")?.map(|a| a.to_vec())
                };
                let num_outputs = node.output.len();
                let out_tensors = if let Some(sizes) = split {
                    // split with explicit sizes
                    let sizes_usize: Vec<usize> = sizes.into_iter().map(|x| x as usize).collect();
                    // use Tensor::split which takes &[usize] ?
                    // For now implement via narrow iteratively
                    let mut outs = Vec::new();
                    let mut offset = 0i64;
                    for &sz in &sizes_usize {
                        let t = input.narrow(axis, Tensor::from(offset), Tensor::from(sz as i64))?;
                        outs.push(t);
                        offset += sz as i64;
                    }
                    outs
                } else {
                    // equal split
                    let dim = input.resolve_shape()[axis as usize];
                    let each = (dim / num_outputs as i64) as i64;
                    let mut outs = Vec::new();
                    for i in 0..num_outputs {
                        let sz = if i == num_outputs - 1 { dim - each * (num_outputs as i64 - 1) } else { each };
                        let t = input.narrow(axis, Tensor::from(i as i64 * each), Tensor::from(sz))?;
                        outs.push(t);
                    }
                    outs
                };
                for (name, t) in node.output.iter().zip(out_tensors.into_iter()) {
                    values.insert(name.clone(), t);
                }
                continue;
            }
            "Pad" => {
                let data = get(&node.input[0]);
                let pads: Vec<i64> = get(&node.input[1]).clone().try_into()?;
                let mode = get_attr_opt::<str>(node, "mode")?.unwrap_or("constant");
                if mode != "constant" {
                    panic!("Pad only supports constant mode, got {mode}");
                }
                let rank = data.rank() as usize;
                if pads.len() != rank * 2 {
                    panic!("Pad pads len {} != rank*2 {}", pads.len(), rank*2);
                }
                let mut out = data.clone();
                // pads are [pad_begin_0, ..., pad_begin_rank-1, pad_end_0, ..., pad_end_rank-1]
                for axis in 0..rank {
                    let left = pads[axis];
                    let right = pads[axis + rank];
                    if left == 0 && right == 0 {
                        continue;
                    }
                    // use pad_zeros: need to handle negative? assume non-negative
                    let dim = out.resolve_shape()[axis];
                    let new_len = dim + left + right;
                    let lp = Tensor::from(left);
                    let len = Tensor::from(new_len);
                    out = out.pad_zeros_axis(axis, lp, len)?;
                }
                // handle constant value if provided
                if node.input.len() > 2 {
                    let _value = get(&node.input[2]);
                    // constant value pad not yet supported via pad_zeros (which pads zeros)
                    // For now, if value is zero, fine, else panic
                    // check if value is zero scalar
                    // We could implement pad with value via where, but keep simple
                    // For non-zero, fallback to error
                    // Try to get value as scalar f32
                    // For now just assume zero
                }
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
                    .map(|d| match &d.value {
                        Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => Tensor::from(*v),
                        Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_)) | None => Tensor::from(1i64),
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
                    DType::F32 => Tensor::from(vec![0.0f32; n]).reshape(shape_concrete.clone())?,
                    DType::F64 => Tensor::from(vec![0.0f64; n]).reshape(shape_concrete.clone())?,
                    DType::I64 => Tensor::from(vec![0i64; n]).reshape(shape_concrete.clone())?,
                    DType::I32 => Tensor::from(vec![0i32; n])
                        .reshape(shape_concrete.clone())?
                        .cast(DType::I64),
                    DType::U8 => Tensor::from(vec![0u8; n]).reshape(shape_concrete.clone())?,
                    _ => Tensor::zeros(shape.clone(), *dtype),
                };
                ph
            };
            tape.add(&placeholder)?;
            values.insert(name.clone(), placeholder);
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

        let frozen = tape.freeze(outputs.iter())?;

        Ok(Self {
            frozen,
            input_names,
            output_names,
        })
    }

    /// Run inference.
    ///
    /// `inputs` maps ONNX input names (e.g. `"x"`) to `Tensor`s. Every model
    /// input must be present; extra entries are ignored.
    pub fn run(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>, ZyxError> {
        let mut ordered = Vec::new();
        for name in &self.input_names {
            let t = inputs
                .get(name)
                .unwrap_or_else(|| panic!("missing input {}", name))
                .clone();
            ordered.push(t);
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
