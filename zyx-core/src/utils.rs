use crate::dtype::DType;
use crate::node::Node;
use crate::shape::Shape;
use crate::tensor::Id;

/// Recursive search to get shape of x in nodes
pub fn shape(nodes: &[Node], mut x: Id) -> &Shape {
    loop {
        let node = &nodes[x.i()];
        match node {
            Node::LeafF32(shape)
            | Node::IterF32(_, shape)
            | Node::UniformF32(shape, ..)
            | Node::LeafI32(shape)
            | Node::IterI32(_, shape)
            | Node::Reshape(_, shape)
            | Node::Expand(_, shape)
            | Node::Permute(.., shape)
            | Node::Sum(.., shape)
            | Node::Max(.., shape) => return shape,
            _ => x = node.parameters().next().unwrap(),
        }
    }
}

/// Recursive search to get dtype of x in nodes
pub fn dtype(nodes: &[Node], mut x: Id) -> DType {
    loop {
        let node = &nodes[x.i()];
        match node {
            Node::LeafF32(..) | Node::IterF32(..) | Node::UniformF32(..) | Node::CastF32(..) => {
                return DType::F32
            }
            Node::LeafI32(..) | Node::IterI32(..) | Node::CastI32(..) => {
                return DType::I32
            }
            _ => x = node.parameters().next().unwrap(),
        }
    }
}
