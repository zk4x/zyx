use std::collections::BTreeSet;
use crate::dtype::DType;
use crate::node::Node;
use crate::shape::Shape;
use crate::tensor::Id;

/// Recursive search to get shape of x in nodes
pub fn get_shape(nodes: &[Node], mut x: Id) -> &Shape {
    loop {
        let node = &nodes[x.i()];
        match node {
            Node::Leaf(shape, ..)
            | Node::Uniform(shape, ..)
            | Node::IterF32(_, shape)
            | Node::IterF64(_, shape)
            | Node::IterI32(_, shape)
            | Node::Reshape(_, shape)
            | Node::Expand(_, shape)
            | Node::Permute(.., shape)
            | Node::Pad(.., shape)
            | Node::Sum(.., shape)
            | Node::Max(.., shape) => return shape,
            _ => x = node.parameters().next().unwrap(),
        }
    }
}

/// Recursive search to get dtype of x in nodes
pub fn get_dtype(nodes: &[Node], mut x: Id) -> DType {
    loop {
        let node = &nodes[x.i()];
        match node {
            Node::Leaf(_, dtype) | Node::Uniform(_, dtype) | Node::Cast(_, dtype) => return *dtype,
            Node::IterF32(..)  => return DType::F32,
            Node::IterF64(..)  => return DType::F64,
            Node::IterI32(..) => return DType::I32,
            _ => x = node.parameters().next().unwrap(),
        }
    }
}

/// Puts graph of nodes into dot language for visualization
pub fn plot_graph_dot(ids: &BTreeSet<Id>, nodes: &[Node], rcs: &[u8]) -> alloc::string::String {
    //let ids = &(0..nodes.len()).map(crate::tensor::id).collect::<alloc::vec::Vec<Id>>();
    use alloc::format;
    use alloc::string::String;
    use core::fmt::Write;
    let mut user_rc = rcs.to_vec();
    for (i, node) in nodes.iter().enumerate() {
        // not all nodes are alive :)
        if rcs[i] > 0 {
            for param in node.parameters() {
                user_rc[param.i()] -= 1;
            }
        }
    }
    //std::println!("User {:?}", user_rc);
    let mut res = String::from("strict digraph {\n  ordering=in\n  rank=source\n");
    let mut add_node = |i: usize, text: &str, shape: &str| {
        let fillcolor = if user_rc[i] > 0 { "lightblue" } else { "grey" };
        /*if let Some(label) = labels.get(&NodeId::new(id)) {
            write!(res, "  {id}[label=\"{}NL{} x {}NL{}NL{}\", shape={}, fillcolor=\"{}\", style=filled]",
                label, id, rc[id], text, get_shape(NodeId::new(id)), shape, fillcolor).unwrap();
        } else {*/
        write!(
            res,
            "  {i}[label=\"{} x {}NL{}NL{}\", shape={}, fillcolor=\"{}\", style=filled]",
            crate::tensor::id(i),
            rcs[i],
            text,
            get_shape(nodes, crate::tensor::id(i)),
            shape,
            fillcolor
        )
        .unwrap();
        writeln!(res).unwrap();
    };
    let mut edges = String::new();
    for id in ids {
        let id = id.i();
        let node = &nodes[id];
        match node {
            Node::IterF32(_, sh) => add_node(id, &format!("Iter({sh}, F32)"), "box"),
            Node::IterF64(_, sh) => add_node(id, &format!("Iter({sh}, F64)"), "box"),
            Node::IterI32(_, sh) => add_node(id, &format!("Iter({sh}, I32)"), "box"),
            Node::Leaf(sh, dtype) => add_node(id, &format!("Leaf({sh}, {dtype})"), "box"),
            Node::Uniform(sh, dtype) => add_node(id, &format!("Uniform({sh}, {dtype})"), "box"),
            Node::Add(x, y) => add_node(id, &format!("Add({x}, {y})"), "oval"),
            Node::Sub(x, y) => add_node(id, &format!("Sub({x}, {y})"), "oval"),
            Node::Mul(x, y) => add_node(id, &format!("Mul({x}, {y})"), "oval"),
            Node::Div(x, y) => add_node(id, &format!("Div({x}, {y})"), "oval"),
            Node::Cmplt(x, y) => add_node(id, &format!("Cmplt({x}, {y})"), "oval"),
            Node::Where(x, y, z) => add_node(id, &format!("Cmplt({x}, {y}, {z})"), "oval"),
            Node::Pow(x, y) => add_node(id, &format!("Pow({x}, {y})"), "oval"),
            Node::Neg(x) => add_node(id, &format!("Neg({x})"), "oval"),
            Node::Exp(x) => add_node(id, &format!("Exp({x})"), "oval"),
            Node::ReLU(x) => add_node(id, &format!("ReLU({x})"), "oval"),
            Node::Ln(x) => add_node(id, &format!("Ln({x})"), "oval"),
            Node::Sin(x) => add_node(id, &format!("Sin({x})"), "oval"),
            Node::Cos(x) => add_node(id, &format!("Cos({x})"), "oval"),
            Node::Sqrt(x) => add_node(id, &format!("Sqrt({x})"), "oval"),
            Node::Tanh(x) => add_node(id, &format!("Tanh({x})"), "oval"),
            Node::Expand(x, ..) => add_node(id, &format!("Expand({x})"), "oval"),
            Node::Pad(x, padding, ..) => add_node(id, &format!("Pad({x}, {padding:?})"), "oval"),
            Node::Cast(x, dtype) => add_node(id, &format!("CastI32({x}, {dtype})"), "oval"),
            Node::Reshape(x, ..) => add_node(id, &format!("Reshape({x})"), "oval"),
            Node::Permute(x, axes, ..) => {
                add_node(id, &format!("Permute({x}, {axes:?})"), "oval")
            }
            Node::Sum(x, axes, ..) => add_node(id, &format!("Sum({x}, {axes:?})"), "oval"),
            Node::Max(x, axes, ..) => add_node(id, &format!("Max({x}, {axes:?})"), "oval"),
        }
        for param in node.parameters() {
            writeln!(edges, "  {} -> {id}", param.i()).unwrap();
        }
    }
    res = res.replace("NL", "\n");
    write!(res, "{edges}}}").unwrap();
    res
}

