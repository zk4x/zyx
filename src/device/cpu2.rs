use crate::shape::{Strides, Shape};

struct View {
    strides0: Strides,
    shape0: Shape,
    strides1: Strides,
    shape1: Shape,
    strides2: Strides,
}
