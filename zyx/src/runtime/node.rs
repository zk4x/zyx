use crate::tensor::Tensor;

pub(super) enum Node {
    Leaf(usize),
    Cast(Tensor),
    ReLU(Tensor),
    Neg(Tensor),
    Inv(Tensor),
    Cos(Tensor),
    Sin(Tensor),
    Exp(Tensor),
    Ln(Tensor),
    Sqrt(Tensor),
    Tanh(Tensor),
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),
    Pow(Tensor, Tensor),
    Cmplt(Tensor, Tensor),
    Where(Tensor, Tensor, Tensor),
}