use crate::onnx::attribute_proto::AttributeType;
use crate::onnx::tensor_proto::DataType;
use crate::onnx::{self, GraphProto};
use zyx::{DType, ZyxError, Tensor};
use std::collections::HashMap;

pub type Value = Tensor;

pub fn dtype(dt: DataType) -> Option<DType> {
    match dt {
        DataType::Uint8 => Some(DType::U8),
        DataType::Uint32 => Some(DType::U32),
        DataType::Int64 => Some(DType::I64),
        //DataType::Float16 => Some(DType::F16),
        DataType::Float => Some(DType::F32),
        DataType::Double => Some(DType::F64),
        DataType::Bool => Some(DType::U8),
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
        std::str::from_utf8(&attr.s).map_err(|e| ZyxError::ParseError(format!("Failed to parse {e}")))
    }
}

impl Attr for GraphProto {
    const TYPE: AttributeType = AttributeType::Graph;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self, ZyxError> {
        attr.g
            .as_ref()
            .ok_or_else(|| ZyxError::ParseError("attribute does not contain graph".to_string()))
    }
}

impl AttrOwned for Vec<String> {
    const TYPE: AttributeType = AttributeType::Strings;
    fn get(attr: &onnx::AttributeProto) -> Result<Self, ZyxError> {
        let mut ret = vec![];
        for bytes in attr.strings.iter() {
            let s = String::from_utf8(bytes.clone()).map_err(|e| ZyxError::ParseError(format!("{e}")))?;
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

        let data_type = match DataType::try_from(tensor_proto.data_type) {
            Ok(value) => value,
            Err(_) => panic!(
                "attribute {} of type TENSOR was an invalid data_type number {}",
                attr.name,
                tensor_proto.data_type
            ),
        };

        let dtype = match dtype(data_type) {
            Some(value) => value,
            None => panic!(
                "attribute {} of type TENSOR has an unsupported data_type {}",
                attr.name,
                data_type.as_str_name()
            ),
        };

        let mut dims = Vec::with_capacity(tensor_proto.dims.len());
        for dim in &tensor_proto.dims {
            if dim < &0 {
                panic!(
                    "attribute {} of type TENSOR has a negative dimension, which is unsupported",
                    attr.name
                )
            }
            dims.push(*dim as usize)
        }

        Tensor::from(&tensor_proto.raw_data).cast(dtype).reshape(&dims)
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
    if attr.r#type() != T::TYPE {
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
            if attr.r#type() != T::TYPE {
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
            if attr.r#type() != T::TYPE {
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
    let dims: Vec<usize> = t.dims.iter().map(|&x| x as usize).collect();
    match DataType::try_from(t.data_type) {
        Ok(DataType::Int32) => {
            if t.int32_data.is_empty() {
                let len = t.raw_data.len() / 4;
                let data: &[i32] =
                    unsafe { std::slice::from_raw_parts(t.raw_data.as_ptr() as *const i32, len) };
                let data = data.iter().map(|v| *v as i64).collect::<Vec<_>>();
                Ok(Tensor::from(data))
            } else {
                let data = t.int32_data.iter().map(|v| *v as i64).collect::<Vec<_>>();
                Ok(Tensor::from(data))
            }
        }
        Ok(dt) => match dtype(dt) {
            Some(dt) => {
                if dt == DType::F32 && !t.float_data.is_empty() {
                    Tensor::from(&t.float_data).reshape(dims.as_slice())
                } else if dt == DType::F64 && !t.double_data.is_empty() {
                    Tensor::from(&t.double_data).reshape(dims.as_slice())
                } else if dt == DType::I64 && !t.int64_data.is_empty() {
                    Tensor::from(&t.int64_data).reshape(dims.as_slice())
                } else {
                    let x = Tensor::from(t.raw_data.as_slice());
                    unsafe { x.bitcast(dt) }?.reshape(dims.as_slice())
                }
            }
            None => {
                panic!("unsupported 'value' data-type {dt:?} for {name}")
            }
        },
        Err(_) => {
            panic!("unsupported 'value' data-type {} for {name}", t.data_type,)
        }
    }
}

// This function provides a direct evaluation of the proto.
// Longer-term, we should first convert the proto to an intermediate representation of the compute
// graph so as to make multiple evaluations more efficient.
// An example upside of this would be to remove intermediary values when they are not needed
// anymore.
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
        let dt = match DataType::try_from(tensor_type.elem_type) {
            Ok(dt) => match dtype(dt) {
                Some(dt) => dt,
                None => {
                    panic!("unsupported 'value' data-type {dt:?} for {}", input.name)
                }
            },
            type_ => panic!("unsupported input type {type_:?}"),
        };
        match &tensor_type.shape {
            None => continue,
            Some(shape) => {
                if shape.dim.len() != tensor.rank() {
                    panic!(
                        "unexpected rank for {}, got {:?}, expected {:?}",
                        input.name,
                        shape.dim,
                        tensor.shape()
                    )
                }
                for (idx, (d, &dim)) in shape.dim.iter().zip(tensor.shape().iter()).enumerate() {
                    match &d.value {
                        Some(onnx::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                            if *v as usize != dim {
                                panic!(
                                    "unexpected dim {idx} for {}, got {:?}, expected {:?}",
                                    input.name,
                                    shape.dim,
                                    tensor.shape()
                                )
                            }
                        }
                        // We do not check equality constraints for the DimParam dimensions for now.
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
    // The nodes are topologically sorted so we can just process them in order.
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

        // TODO: Validate node.input for each operator.
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
                // TODO: Check that there is at most a single -1 or 0, handle other neg values.
                let mut other_than_minus1 = 1usize;
                for &v in shape.iter() {
                    if v != -1 && v != 0 {
                        other_than_minus1 *= v as usize
                    }
                }
                let shape = shape
                    .iter()
                    .enumerate()
                    .map(|(idx, &v)| match v {
                        -1 => input0.numel() / other_than_minus1,
                        0 => input0.shape()[idx],
                        _ => v as usize,
                    })
                    .collect::<Vec<usize>>();
                let output = input0.reshape(shape)?;
                values.insert(node.output[0].clone(), output);
            }
            "LogSoftmax" => {
                let input = get(&node.input[0]);
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => input.softmax([-1])?,
                    Some(&axis) => {
                        input.ln_softmax([axis as isize])?
                    }
                };
                values.insert(node.output[0].clone(), output);
            }
            "Softmax" => {
                let input = get(&node.input[0]);
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => input.softmax([-1])?,
                    Some(&axis) => input.softmax([axis as isize])?,
                };
                values.insert(node.output[0].clone(), output);
            }
            "Transpose" => {
                let input = get(&node.input[0]);
                let output = match get_attr_opt::<[i64]>(node, "perm")? {
                    None => input.t(),
                    Some(perm) => {
                        let perm = perm.iter().map(|&v| v as isize).collect::<Vec<_>>();
                        input.permute(perm)?
                    }
                };
                values.insert(node.output[0].clone(), output);
            }
            "Dropout" => {
                let input = get(&node.input[0]);
                // Do not apply dropout at the moment, consider that we're only doing inference.
                values.insert(node.output[0].clone(), input.clone());
            }
            /*"MaxPool" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let kernel_shape = get_attr::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                match auto_pad {
                    None | Some("NOTSET") => (),
                    Some(s) => panic!("unsupported auto_pad {s}"),
                };
                if let Some(d) = dilations {
                    if d.iter().any(|&v| v != 1) {
                        panic!("MaxPool with dilation != 1, {dilations:?}")
                    }
                }
                if let Some(d) = pads {
                    if d.iter().any(|&v| v != 0) {
                        panic!("MaxPool with pads != 0, {pads:?}")
                    }
                }
                let xs = get(&node.input[0])?;
                let (k1, k2) = match kernel_shape {
                    [k1, k2] => (*k1 as usize, *k2 as usize),
                    _ => panic!("only 2d MaxPool is supported, kernel shape {kernel_shape:?}"),
                };
                let ys = match strides {
                    None => xs.max_pool2d((k1, k2))?,
                    Some([s1, s2]) => {
                        xs.max_pool2d_with_stride((k1, k2), (*s1 as usize, *s2 as usize))?
                    }
                    Some(strides) => panic!("only 2d MaxPool is supported, strides {strides:?}"),
                };
                values.insert(node.output[0].clone(), ys);
            }*/
            /*"AveragePool" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let kernel_shape = get_attr::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                match auto_pad {
                    None | Some("NOTSET") => (),
                    Some(s) => panic!("unsupported auto_pad {s}"),
                };
                if let Some(d) = dilations {
                    if d.iter().any(|&v| v != 1) {
                        panic!("AvgPool with dilation != 1, {dilations:?}")
                    }
                }
                if let Some(d) = pads {
                    if d.iter().any(|&v| v != 0) {
                        panic!("AvgPool with pads != 0, {pads:?}")
                    }
                }
                let xs = get(&node.input[0])?;
                let (k1, k2) = match kernel_shape {
                    [k1, k2] => (*k1 as usize, *k2 as usize),
                    _ => panic!("only 2d AvgPool is supported, kernel shape {kernel_shape:?}"),
                };
                let ys = match strides {
                    None => xs.avg_pool2d((k1, k2))?,
                    Some([s1, s2]) => {
                        xs.avg_pool2d_with_stride((k1, k2), (*s1 as usize, *s2 as usize))?
                    }
                    Some(strides) => panic!("only 2d AvgPool is supported, strides {strides:?}"),
                };
                values.insert(node.output[0].clone(), ys);
            }*/
            /*"BatchNormalization" => {
                let training_mode = get_attr_opt::<i64>(node, "training_mode")?;
                if training_mode.copied().unwrap_or(0) != 0 {
                    panic!("training mode is not supported for BatchNorm")
                }
                let eps = get_attr_opt::<f32>(node, "epsilon")?
                    .copied()
                    .unwrap_or(1e-5);
                let xs = get(&node.input[0])?;
                let weight = get(&node.input[1])?;
                let bias = get(&node.input[2])?;
                let running_mean = get(&node.input[3])?;
                let running_var = get(&node.input[4])?;
                let target_shape: Vec<usize> = xs
                    .shape()
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| if idx == 1 { *v } else { 1 })
                    .collect();
                let target_shape = target_shape.as_slice();
                let xs = xs
                    .broadcast_sub(&running_mean.reshape(target_shape)?)?
                    .broadcast_div(&(running_var.reshape(target_shape)? + eps as f64)?.sqrt()?)?;
                let weight = weight.reshape(target_shape)?;
                let bias = bias.reshape(target_shape)?;
                let xs = xs.broadcast_mul(&weight)?.broadcast_add(&bias)?;
                values.insert(node.output[0].clone(), xs);
            }*/
            /*"Squeeze" => {
                let xs = get(&node.input[0])?;
                let mut axes = if node.input.len() <= 1 {
                    // contract all the dimensions with size 1 except the batch dim.
                    xs.shape()
                        .iter()
                        .enumerate()
                        .flat_map(|(idx, &s)| if s == 1 && idx > 0 { Some(idx) } else { None })
                        .collect()
                } else {
                    let axes = get(&node.input[1])?.clone();
                    let axes: Vec<i64> = x.try_into()?;
                    axes
                };
                axes.sort();
                let mut xs = xs.clone();
                for &axis in axes.iter().rev() {
                    xs = xs.squeeze(axis)?
                }
                values.insert(node.output[0].clone(), xs);
            }*/
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape
            /*"ConstantOfShape" => {
                let input = get(&node.input[0])?;
                /*let value = get_attr_opt_owned::<Tensor>(node, "value")?.unwrap_or(Tensor::zeros(
                    (),
                    DType::F32,
                    &Device::Cpu,
                )?);

                let xs = Tensor::ones(input.shape(), value.dtype(), input.device())?
                    .broadcast_mul(&value)?;*/
                values.insert(node.output[0].clone(), xs);
            }*/
            /*"Unsqueeze" => {
                let xs = get(&node.input[0])?;
                let axes = match get_attr_opt::<[i64]>(node, "axes")? {
                    Some(axis) => axis.to_vec(),
                    None => get(&node.input[1])?.to_vec1::<i64>()?,
                };
                let mut axes = axes
                    .iter()
                    .map(|&i| {
                        if i == xs.rank() as i64 {
                            Ok(xs.rank())
                        } else if i < 0 {
                            // normalize_axis doesn't work correctly here
                            // because we actually want normalized with respect
                            // to the final size, not the current (off by one)
                            Ok(xs.rank() - (-i as usize) + 1)
                        } else {
                            xs.normalize_axis(i)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                axes.sort();
                let mut xs = xs.clone();
                for &axis in axes.iter().rev() {
                    xs = xs.unsqueeze(axis)?
                }
                values.insert(node.output[0].clone(), xs);
            }*/
            /*"Clip" => {
                let xs = get(&node.input[0])?;
                let xs = if let Some(mins) = get_opt(1) {
                    xs.broadcast_maximum(mins?)?
                } else {
                    xs.clone()
                };
                let xs = if let Some(maxs) = get_opt(2) {
                    xs.broadcast_minimum(maxs?)?
                } else {
                    xs.clone()
                };
                values.insert(node.output[0].clone(), xs);
            }
            "Gather" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather
                let xs = get(&node.input[0])?;
                let indices = get(&node.input[1])?;
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                let axis = xs.normalize_axis(axis)?;

                // index_select does not support negative indices, so normalize them
                // to positive indices.
                let indices = &{
                    let zeros = Tensor::zeros(indices.shape(), indices.dtype(), indices.device())?;
                    let max = Tensor::new(xs.dims()[axis] as i64, indices.device())?
                        .to_dtype(indices.dtype())?;
                    let mask = indices.lt(&zeros)?;
                    mask.to_dtype(indices.dtype())?
                        .broadcast_mul(&max)?
                        .add(indices)?
                };

                // In Pytorch or Numpy this can be done by indexing the xs tensor using the indices
                // tensor directly, but candle does not support tensor indexing at the moment, so
                // some workarounds must be done.
                let xs = match indices.dims() {
                    [] => {
                        let index = indices.to_vec0::<i64>()? as usize;
                        xs.narrow(axis, index, 1)?.squeeze(axis)?
                    }
                    [_] => xs.index_select(indices, axis)?,
                    [first, _] => {
                        let mut v = Vec::with_capacity(*first);
                        for i in 0..*first {
                            v.push(xs.index_select(&indices.get(i)?, axis)?)
                        }
                        Tensor::stack(&v, axis)?
                    }
                    _ => {
                        // TODO: Provide an op to handle the ONNX generalized gather op ideally in a
                        // differentiable way.
                        todo!("implement gather for {xs:?} {indices:?} axis {axis}")
                    }
                };
                values.insert(node.output[0].clone(), xs);
            }*/
            /*"Shape" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape
                let xs = get(&node.input[0])?;
                let start = get_attr_opt::<i64>(node, "start")?.copied().unwrap_or(0);
                let end = get_attr_opt::<i64>(node, "end")?.copied().unwrap_or(-1);
                let start = xs.normalize_axis(start)?;
                let end = xs.normalize_axis(end)?;
                let mut dims = vec![];
                for idx in start..=end {
                    dims.push(xs.dim(idx)? as i64)
                }
                let dims = Tensor::from_vec(dims, xs.rank(), xs.device())?;
                values.insert(node.output[0].clone(), dims);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size
            "Size" => {
                let data = get(&node.input[0])?;
                let size: usize = data.dims().iter().product();
                let output = Tensor::from_slice(&[size as i64], (), data.device())?;
                values.insert(node.output[0].clone(), output);
            }*/
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt
            "Sqrt" => {
                let xs = get(&node.input[0]);
                let output = xs.sqrt();
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range
            /*"Range" => {
                let start = get(&node.input[0])?;
                let limit = get(&node.input[1])?;
                let delta = get(&node.input[2])?;

                macro_rules! arange_step {
                    ($t: ty) => {
                        Tensor::arange_step(
                            start.to_vec0::<$t>()?,
                            limit.to_vec0::<$t>()?,
                            delta.to_vec0::<$t>()?,
                            &Device::Cpu,
                        )?
                    };
                }

                let output = match start.dtype() {
                    DType::U8 => arange_step!(u8),
                    DType::U32 => arange_step!(u32),
                    DType::I64 => arange_step!(i64),
                    DType::BF16 => arange_step!(f32),
                    DType::F16 => arange_step!(f32),
                    DType::F32 => arange_step!(f32),
                    DType::F64 => arange_step!(f64),
                };

                values.insert(node.output[0].clone(), output);
            }*/
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater
            "Greater" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);

                let output = a.cmpgt(b)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less
            "Less" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);

                let output = a.cmplt(b)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log
            "Log" => {
                let a = get(&node.input[0]);
                let output = a.ln();
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min
            "Min" => {
                let mut output = get(&node.input[0]).clone();
                for input in node.input.iter() {
                    let input = get(input);
                    output = output.minimum(input)?;
                }
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where
            "Where" => {
                let cond = get(&node.input[0]);
                let a = get(&node.input[1]);
                let b = get(&node.input[2]);
                let output = cond.where_(a, b)?;
                values.insert(node.output[0].clone(), output);
            }
            /*"Conv" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let groups = get_attr_opt::<i64>(node, "group")?.copied().unwrap_or(1);
                let _kernel_shape = get_attr_opt::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                match auto_pad {
                    None | Some("NOTSET") => (),
                    Some(s) => panic!("unsupported auto_pad {s}"),
                };
                let xs = get(&node.input[0])?;
                let ws = get(&node.input[1])?;
                let ys = match ws.rank() {
                    3 => {
                        let (pads, xs) = match pads {
                            None => (0, xs.clone()),
                            Some([p]) => (*p as usize, xs.clone()),
                            Some([p1, p2]) => {
                                if p1 != p2 {
                                    (0usize, xs.pad_zeros(2, *p1 as usize, *p2 as usize)?)
                                } else {
                                    (*p1 as usize, xs.clone())
                                }
                            }
                            Some(pads) => {
                                panic!("more pads than expected in conv1d {pads:?} {}", node.name)
                            }
                        };
                        let strides = match strides {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some(s) => {
                                panic!("more strides than expected in conv1d {s:?} {}", node.name)
                            }
                        };
                        let dilations = match dilations {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some(s) => {
                                panic!("more dilations than expected in conv1d {s:?} {}", node.name)
                            }
                        };
                        xs.conv1d(ws, pads, strides, dilations, groups as usize)?
                    }
                    4 => {
                        let (pads, xs) = match pads {
                            None => (0, xs.clone()),
                            Some([p]) => (*p as usize, xs.clone()),
                            Some(&[p1, p2, p3, p4]) => {
                                let p1 = p1 as usize;
                                let p2 = p2 as usize;
                                let p3 = p3 as usize;
                                let p4 = p4 as usize;
                                if p1 != p2 || p1 != p3 || p1 != p4 {
                                    (0, xs.pad_with_zeros(2, p1, p3)?.pad_with_zeros(3, p2, p4)?)
                                } else {
                                    (p1, xs.clone())
                                }
                            }
                            Some(pads) => {
                                panic!("more pads than expected in conv2d {pads:?} {}", node.name)
                            }
                        };
                        let strides = match strides {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some([p1, p2]) => {
                                if p1 != p2 {
                                    panic!(
                                        "strides have to be the same on both axis {pads:?} {}",
                                        node.name
                                    )
                                }
                                *p1 as usize
                            }
                            Some(s) => {
                                panic!("more strides than expected in conv2d {s:?} {}", node.name)
                            }
                        };
                        let dilations = match dilations {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some([p1, p2]) => {
                                if p1 != p2 {
                                    panic!(
                                        "dilations have to be the same on both axis {pads:?} {}",
                                        node.name
                                    )
                                }
                                *p1 as usize
                            }
                            Some(s) => {
                                panic!("more dilations than expected in conv2d {s:?} {}", node.name)
                            }
                        };
                        xs.conv2d(ws, pads, strides, dilations, groups as usize)?
                    }
                    rank => panic!(
                        "unsupported rank for weight matrix {rank} in conv {}",
                        node.name
                    ),
                };
                let ys = if node.input.len() > 2 {
                    let bs = get(&node.input[2])?;
                    let mut bs_shape = vec![1; ys.rank()];
                    bs_shape[1] = bs.elem_count();
                    ys.broadcast_add(&bs.reshape(bs_shape)?)?
                } else {
                    ys
                };
                values.insert(node.output[0].clone(), ys);
            }*/
            "Concat" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
                let inputs: Vec<Value> = node
                    .input
                    .iter()
                    .map(|n| get(n.as_str()).clone())
                    .collect();
                let axis: i64 = *get_attr(node, "axis")?;
                if inputs.is_empty() {
                    panic!("empty concat")
                };
                let output = Tensor::cat(&inputs, axis as isize)?;
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
            /*"Erf" => {
                let input = get(&node.input[0])?;
                let output = input.erf();
                values.insert(node.output[0].clone(), output);
            }*/
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
            /*"Ceil" => {
                let input = get(&node.input[0])?;
                let output = input.ceil();
                values.insert(node.output[0].clone(), output);
            }
            "Floor" => {
                let input = get(&node.input[0])?;
                let output = input.floor();
                values.insert(node.output[0].clone(), output);
            }*/
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant
            "Constant" => {
                let value = match node.attribute.iter().find(|attr| attr.name == "value") {
                    None => {
                        // TODO: support sparse_value etc.
                        panic!("cannot find 'value' attr in 'Constant' for {}", node.name)
                    }
                    Some(value) => value,
                };
                let output = match value.r#type() {
                    AttributeType::Tensor => {
                        let t = value.t.as_ref().unwrap();
                        get_tensor(t, &node.name)?
                    }
                    rtype => panic!("unsupported 'value' type {rtype:?} for {}", node.name),
                };
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
            "Cast" => {
                let input = get(&node.input[0]);
                let dt: i64 = *get_attr(node, "to")?;
                let dtype = match DataType::try_from(dt as i32) {
                    Ok(DataType::Int32) => DType::I64,
                    Ok(dt) => match dtype(dt) {
                        Some(dt) => dt,
                        None => {
                            panic!("unsupported 'to' value {dt:?} for cast {}", node.name)
                        }
                    },
                    Err(_) => {
                        panic!("unsupported 'to' value {dt:?} for cast {}", node.name)
                    }
                };
                let output = input.cast(dtype);
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
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
                let axis: u32 = get(&node.input[1]).cast(DType::U32).try_into()?;
                let output = input.cumsum(axis as isize)?;
                values.insert(node.output[0].clone(), output);
            }
            //  https://github.com/onnx/onnx/blob/main/docs/Operators.md#flatten
            "Flatten" => {
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(1) as usize;
                let input = get(&node.input[0]);
                let first_part: usize = input.shape().iter().take(axis).product();
                let end_index = input.shape().iter().product::<usize>();
                let new_shape = (first_part, end_index / first_part);
                let output = input.reshape(new_shape)?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#identity
            "Identity" => {
                let input = get(&node.input[0]);
                values.insert(node.output[0].clone(), input.clone());
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#if
            /*"If" => {
                // protobuf encodes boolean false as 0 and true as 1
                let cond = get(&node.input[0])?.get(0)?.to_scalar::<u8>()?;
                let attr_name = if cond != 0 {
                    "then_branch"
                } else {
                    "else_branch"
                };
                let sub_graph = get_attr::<GraphProto>(node, attr_name)?;
                if sub_graph.output.len() != node.output.len() {
                    panic!(
                        "If node {:?} is malformed: branch outputs ({}) don't match node outputs ({})",
                        node.name,
                        sub_graph.output.len(),
                        node.output.len()
                    );
                }
                let branch_out = simple_eval_(sub_graph, values)?;
                for (i, out) in node.output.iter().enumerate() {
                    values.insert(
                        out.clone(),
                        branch_out.get(&sub_graph.output[i].name).unwrap().clone(),
                    );
                }
            }*/
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#pad
            /*"Pad" => {
                let mode = get_attr_opt(node, "mode")?.unwrap_or("constant");
                let data = get(&node.input[0])?;
                let pads = get(&node.input[1])?;
                if node.input.len() > 2 {
                    panic!(
                        "unsupported number of inputs {} for Pad node {:?}, expected 2",
                        node.input.len(),
                        node.name
                    );
                }
                if pads.rank() != 1 {
                    panic!("Pad expects 'pads' input to be 1D vector: {pads:?}");
                }
                if pads.dim(0).unwrap() != 2 * data.rank() {
                    panic!("Pad expects 'pads' input len to be 2 * rank of 'data' input: pads: {}, data rank: {}", pads, data.rank());
                }

                let pads = pads.to_vec1::<i64>()?;
                let (pads_pre, pads_post) = pads.split_at(pads.len() / 2);

                match mode {
                    "reflect" => {
                        let mut out = data.clone();
                        for (i, &dim) in data.dims().iter().enumerate().rev() {
                            if pads_pre[i] == 0 && pads_post[i] == 0 {
                                continue;
                            }
                            fn zigzag(min: i64, max: i64) -> impl Iterator<Item = i64> {
                                std::iter::repeat((min..max).chain((min + 1..=max).rev())).flatten()
                            }
                            let idx = if dim > 1 {
                                let cycle_len = dim * 2 - 2;
                                let skip = cycle_len - ((pads_pre[i] as usize) % cycle_len);
                                let idx = zigzag(0, (dim - 1) as i64)
                                    .skip(skip)
                                    .take((pads_pre[i] as usize) + dim + (pads_post[i] as usize));
                                Tensor::from_iter(idx, out.device())?
                            } else {
                                Tensor::full(0i64, (dim,), out.device())?
                            };

                            out = out.index_select(&idx, i)?;
                        }

                        values.insert(node.output[0].clone(), out);
                    }
                    _ => panic!(
                        "unsupported 'mode' value {mode:?} for Pad node {:?}",
                        node.name
                    ),
                }
            }*/
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#slice
            /*"Slice" => {
                let data = get(&node.input[0])?;
                let starts = get(&node.input[1])?;
                let ends = get(&node.input[2])?;
                let default_axes;
                let default_steps;
                let axes: &Tensor;
                let steps: &Tensor;
                // If axes are omitted, they are set to [0, ..., r-1]. If steps are omitted,
                // they are set to [1, ..., 1] of length len(starts)
                match node.input.len() {
                    3 => {
                        let len = starts.dims()[0];
                        default_axes = Some(Tensor::arange(0, len as i64, starts.device())?);
                        axes = default_axes.as_ref().unwrap();
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    }
                    4 => {
                        let len = starts.dims()[0];
                        axes = get(&node.input[3])?;
                        default_steps = Some(Tensor::ones((len,), DType::I64, starts.device())?);
                        steps = default_steps.as_ref().unwrap();
                    }
                    5 => {
                        steps = get(&node.input[4])?;
                        axes = get(&node.input[3])?;
                    }
                    _ => panic!(
                        "Slice node is invalid, expected 3-5 inputs, got {}: {:?}",
                        node.input.len(),
                        node
                    ),
                }

                let mut out = data.clone();
                for (i, axis) in axes.to_vec1::<i64>()?.into_iter().enumerate() {
                    // All negative elements of axes are made non-negative by
                    // adding r to them, where r = rank(input).
                    let axis = if axis < 0 {
                        axis + data.rank() as i64
                    } else {
                        axis
                    } as usize;

                    let data_dim = data.shape()[axis] as i64;
                    let mut s: i64 = starts.get(i)?.try_into()?;
                    let mut e: i64 = ends.get(i)?.try_into()?;
                    // All negative values in starts[i] and ends[i] have
                    // dims[axes[i]] added to them, where dims are the
                    // dimensions of input.
                    if s < 0 {
                        s += data_dim;
                    }
                    if e < 0 {
                        e += data_dim;
                    }

                    let p: i64 = steps.get(i)?.try_into()?;
                    // starts[i] is clamped into the range [0, dims[axes[i]]]
                    // for positive stepping and [0, dims[axes[i]]-1] for
                    // negative stepping.
                    // for positive stepping ends[axes[i]] is clamped to
                    // [0, dims[axes[i]]], while for negative stepping it is
                    // clamped to [-1, dims[axes[i]]-1].
                    if p >= 0 {
                        s = s.clamp(0, data_dim);
                        e = e.clamp(0, data_dim);
                    } else {
                        s = s.clamp(0, data_dim - 1);
                        e = e.clamp(-1, data_dim - 1);
                    }

                    let indexes = Tensor::arange(s, e, p)?;
                    out = out.index_select(&indexes, axis)?
                }
                values.insert(node.output[0].clone(), out);
            }*/
            // https://onnx.ai/onnx/operators/onnx__ReduceMean.html#reducemean-13
            // TODO: This version is only compatible with ReduceMean V13 and below.
            "ReduceMean" => {
                let input = get(&node.input[0]);
                let axes = get_attr_opt::<[i64]>(node, "axes")?;
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);

                let n_dims = input.shape().len() as isize;

                let axes: Vec<isize> = if let Some(axes) = axes {
                    axes.iter().map(|a| *a as isize).collect()
                } else {
                    (0..n_dims).collect()
                };
                let output = if keepdims == 1 {
                    input.mean_kd(axes)?
                } else {
                    input.mean(axes)?
                };
                values.insert(node.output[0].clone(), output);
            }
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
            // Version 18 impl
            /*"Split" => {
                let input_tensor = get(&node.input[0])?;
                let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
                let axis = input_tensor.normalize_axis(axis)?;

                // Determine split sizes
                let splits = if node.input.len() > 1 {
                    // If the split tensor is provided, use it to determine sizes
                    let split_tensor = get(&node.input[1])?.to_vec1::<i64>()?;
                    split_tensor.iter().map(|&x| x as usize).collect::<Vec<_>>()
                } else {
                    let num_outputs = if let Some(&num_outputs_attrib) =
                        get_attr_opt::<i64>(node, "num_outputs")?
                    {
                        num_outputs_attrib as usize
                    } else {
                        node.output.len()
                    };

                    let input_dim = input_tensor.dim(axis)?;

                    let mut split_sizes =
                        vec![input_dim / num_outputs as usize; num_outputs as usize];
                    let remainder = input_dim % num_outputs as usize;
                    if remainder > 0 {
                        // If there's a remainder, add it to the last split size
                        split_sizes[num_outputs as usize - 1] += remainder;
                    }

                    split_sizes
                };

                // Perform the split operation
                let mut outputs = vec![];
                let mut start = 0;
                for &size in &splits {
                    let end = start + size;
                    let slice = input_tensor.narrow(axis, start, size)?;
                    outputs.push(slice);
                    start = end;
                }

                // Insert the split outputs into the values map
                for (output, slice) in node.output.iter().zip(outputs.into_iter()) {
                    values.insert(output.clone(), slice);
                }
            }*/
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand
            // Version 13 impl
            /*"Expand" => {
                // unlike broadcast_to, expand allows for the output shape to
                // be different from the specified shape.
                let input_tensor = get(&node.input[0])?;
                let input_shape = get(&node.input[1])?;

                // Check that the shape tensor is 1D
                if input_shape.rank() != 1 {
                    panic!(
                        "Expand expects 'shape' input to be 1D tensor: {:?}",
                        input_shape
                    );
                }
                let input_tensor_dims = input_tensor.shape();
                let input_shape_dims = input_shape
                    .to_vec1::<i64>()?
                    .into_iter()
                    .map(|x| x as usize)
                    .collect::<Vec<_>>();

                let target_shape = broadcast_shape(input_tensor_dims, input_shape_dims.as_slice())?;

                let expanded_tensor = input_tensor.broadcast_as(target_shape)?;

                values.insert(node.output[0].clone(), expanded_tensor);
            }*/
            //https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum
            // Version 13 impl
            "ReduceSum" => {
                let input = get(&node.input[0]);
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop_with_empty_axes = get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                    .copied()
                    .unwrap_or(0);

                let axes = match axes {
                    Some(axes) => {
                        let axes: Vec<i64> = axes.clone().try_into()?;
                        axes
                        .into_iter()
                        .map(|x| x as isize)
                        .collect::<Vec<_>>()
                    }
                    None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input.rank() as isize).collect()
                        }
                    }
                };

                let output = if keepdims == 1 {
                    input.sum_kd(axes)?
                } else {
                    input.sum(axes)?
                };

                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2
            // Version 18 impl
            /*"ReduceL2" => {
                let input = get(&node.input[0])?;
                let axes = get_opt(1);
                let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
                let noop_with_empty_axes = get_attr_opt::<i64>(node, "noop_with_empty_axes")?
                    .copied()
                    .unwrap_or(0);

                let input_sq = input.sqr()?;

                let axes = match axes {
                    Some(axes) => axes?
                        .to_vec1::<i64>()?
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<_>>(),
                    None => {
                        if noop_with_empty_axes == 1 {
                            vec![]
                        } else {
                            (0..input_sq.rank()).collect()
                        }
                    }
                };

                let output = if keepdims == 1 {
                    input_sq.sum_keepdim(axes)?.sqrt()?
                } else {
                    input_sq.sum(axes)?.sqrt()?
                };

                values.insert(node.output[0].clone(), output);
            }
            random_type @ ("RandomUniform" | "RandomNormal") => {
                let dt: i64 = get_attr_opt(node, "dtype")?.copied().unwrap_or(1); // 1 is float
                                                                                  // type by
                                                                                  // default
                let dtype = match DataType::try_from(dt as i32) {
                    Ok(dt) => match dtype(dt) {
                        Some(DType::U8 | DType::U32 | DType::I64) => {
                            panic!(
                                "unsupported 'dtype' value {dt:?}, only floats are allowed, for {random_type} {}",
                                node.name
                            )
                        }
                        Some(dt) => dt,
                        None => {
                            panic!(
                                "unsupported 'dtype' value {dt:?} for {random_type} {}",
                                node.name
                            )
                        }
                    },
                    Err(_) => {
                        panic!(
                            "unsupported 'dtype' value {dt:?} for {random_type} {}",
                            node.name
                        )
                    }
                };
                let seed: Option<f32> = get_attr_opt(node, "seed")?.copied();
                if seed.is_some() {
                    panic!("seed for {random_type} is currently not supported")
                };
                let shape: Vec<usize> = get_attr::<[i64]>(node, "shape")?
                    .iter()
                    .map(|x| *x as usize)
                    .collect();
                let output = if random_type == "RandomUniform" {
                    let low: f32 = get_attr_opt(node, "low")?.copied().unwrap_or(0.0);
                    let high: f32 = get_attr_opt(node, "high")?.copied().unwrap_or(1.0);
                    Tensor::rand(low, high, shape, &Device::Cpu)?.to_dtype(dtype)?
                } else {
                    let mean: f32 = get_attr_opt(node, "mean")?.copied().unwrap_or(0.0);
                    let scale: f32 = get_attr_opt(node, "scale")?.copied().unwrap_or(1.0);
                    Tensor::randn(mean, scale, shape, &Device::Cpu)?.to_dtype(dtype)?
                };
                values.insert(node.output[0].clone(), output);
            }*/
            /*"ArgMin" => {
                let input = get(&node.input[0])?;
                let axis_i64: i64 = get_attr_opt(node, "axis")?.copied().unwrap_or(0);
                let rank_i64: i64 = input.rank().try_into().unwrap();
                if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
                    panic!(
                        "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                        axis_i64,
                        -rank_i64,
                        rank_i64 - 1
                    )
                }
                let axis = input.normalize_axis(axis_i64)?;
                let keepdims: i64 = get_attr_opt(node, "keepdims")?.copied().unwrap_or(1);
                let select_last_index: i64 = get_attr_opt(node, "select_last_index")?
                    .copied()
                    .unwrap_or(0);
                if select_last_index == 1 {
                    panic!("select_last_index for ArgMin is currently not supported")
                }
                let output = if keepdims == 1 {
                    input.argmin_keepdim(axis)?
                } else {
                    input.argmin(axis)?
                }
                .to_dtype(DType::I64)?;
                values.insert(node.output[0].clone(), output);
            }*/
            /*"ArgMax" => {
                let input = get(&node.input[0])?;
                let axis_i64: i64 = get_attr_opt(node, "axis")?.copied().unwrap_or(0);
                let rank_i64: i64 = input.rank().try_into().unwrap();
                if axis_i64 < -rank_i64 || axis_i64 >= rank_i64 {
                    panic!(
                        "axis ({}) out of accepted range [-rank, rank-1] which was [{}, {}]",
                        axis_i64,
                        -rank_i64,
                        rank_i64 - 1
                    )
                }
                let axis = input.normalize_axis(axis_i64)?;
                let keepdims: i64 = get_attr_opt(node, "keepdims")?.copied().unwrap_or(1);
                let select_last_index: i64 = get_attr_opt(node, "select_last_index")?
                    .copied()
                    .unwrap_or(0);
                if select_last_index == 1 {
                    panic!("select_last_index for ArgMin is currently not supported")
                }
                let output = if keepdims == 1 {
                    input.argmax_keepdim(axis)?
                } else {
                    input.argmax(axis)?
                }
                .to_dtype(DType::I64)?;
                values.insert(node.output[0].clone(), output);
            }*/
            "LeakyRelu" => {
                let input = get(&node.input[0]);
                let dt = input.dtype();
                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(0.01);
                let output = input.leaky_relu(alpha);
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm
            "Gemm" => {
                let a = get(&node.input[0]);
                let b = get(&node.input[1]);
                let c = get(&node.input[2]);

                let alpha = get_attr_opt::<f32>(node, "alpha")?.copied().unwrap_or(1.0);
                let beta = get_attr_opt::<f32>(node, "beta")?.copied().unwrap_or(1.0);

                let alpha = Tensor::full(a.shape(), alpha)?;
                let beta = Tensor::full(c.shape(), beta)?;

                let trans_a = get_attr_opt::<i64>(node, "transA")?.copied().unwrap_or(0);
                let trans_b = get_attr_opt::<i64>(node, "transB")?.copied().unwrap_or(0);

                let a = if trans_a == 0 { a.clone() } else { a.t() };
                let b = if trans_b == 0 { b.clone() } else { b.t() };

                let output = (a * alpha).matmul(&b)? + c * beta;
                values.insert(node.output[0].clone(), output);
            }
            op_type => panic!("unsupported op_type {op_type} for op {node:?}"),
        }
    }
    graph
        .output
        .iter()
        .map(|output| match values.remove(&output.name) {
            None => panic!("cannot find output {}", output.name),
            Some(value) => Ok((output.name.clone(), value)),
        })
        .collect()
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
