type TensorId = u32;

pub(super) enum Node {
    Leaf(usize),
    Cast(TensorId),
    ReLU(TensorId),
    Neg(TensorId),
    Inv(TensorId),
    Cos(TensorId),
    Sin(TensorId),
    Exp(TensorId),
    Ln(TensorId),
    Sqrt(TensorId),
    Tanh(TensorId),
    Add(TensorId, TensorId),
    Sub(TensorId, TensorId),
    Mul(TensorId, TensorId),
    Div(TensorId, TensorId),
    Pow(TensorId, TensorId),
    Cmplt(TensorId, TensorId),
    Where(TensorId, TensorId, TensorId),
}