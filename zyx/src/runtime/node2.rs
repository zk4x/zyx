#[derive(Debug, Clone, Copy)]
enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
}

#[derive(Debug, Clone, Copy)]
enum UOp {
    Exp,
    Ln,
    Tanh,
    Inv,
    Sqrt,
    Sin,
    Cos,
}

#[derive(Debug)]
enum ROp {
    Sum,
    Max,
}

type Axis = usize;

type Dimension = usize;

#[derive(Debug)]
pub(crate) enum Node {
    Leaf {
        shape: Vec<Dimension>,
    },
    Expand {
        x: Id,
        shape: Vec<Dimension>,
    },
    Permute {
        x: Id,
        axes: Vec<Axis>,
    },
    Split {
        axis: Axis,
        dimensions: Vec<Dimension>,
    },
    // TODO perhaps also add axis join, not only split
    Reshape {
        x: Id,
        shape: Vec<Dimension>,
    },
    Pad {
        x: Id,
        pad: Vec<(isize, isize)>,
    },
    Unary {
        x: Id,
        uop: UOp,
    },
    Binary {
        x: Id,
        y: Id,
        bop: BOp,
    },
    Reduce {
        x: Id,
        y: Id,
        axes: Vec<Axis>,
        rop: ROp,
    },
}
