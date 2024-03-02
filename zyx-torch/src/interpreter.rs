use alloc::{collections::{BTreeMap, btree_map::Entry}, vec::Vec};
use tch::{Kind, Tensor};
use zyx_core::{
    error::ZyxError, node::Node, runtime::RuntimeBackend, scalar::Scalar, tensor::Id,
};
use zyx_core::dtype::DType;

pub struct Interpreter {
    tensors: BTreeMap<Id, Tensor>,
}

impl Interpreter {
    pub(crate) fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
        }
    }
}

impl RuntimeBackend for Interpreter {
    fn is_evaluated(&self, x: Id) -> bool {
        self.tensors.contains_key(&x)
    }

    fn is_free_id(&self, x: Id) -> bool {
        !self.tensors.contains_key(&x)
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        self.tensors.remove(&x);
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, _numel: usize) -> Result<Vec<T>, ZyxError> {
        Ok(match T::dtype() {
            DType::F32 => {
                unsafe { core::mem::transmute::<Vec<f32>, Vec<T>>(self.tensors[&x].data().flatten(0, -1).try_into().unwrap()) }
            }
            DType::F64 => {
                unsafe { core::mem::transmute::<Vec<f64>, Vec<T>>(self.tensors[&x].data().flatten(0, -1).try_into().unwrap()) }
            }
            DType::I32 => {
                unsafe { core::mem::transmute::<Vec<i32>, Vec<T>>(self.tensors[&x].data().flatten(0, -1).try_into().unwrap()) }
            }
        })
    }

    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
    where
        IT: IntoIterator<Item=T>,
        IT::IntoIter: ExactSizeIterator,
    {
        match T::dtype() {
            DType::F32 => self.tensors.insert(x, <Vec<f32> as TryInto<Tensor>>::try_into(iter.into_iter().map(|x| x.into_f32()).collect::<Vec<f32>>()).unwrap()),
            DType::F64 => self.tensors.insert(x, <Vec<f64> as TryInto<Tensor>>::try_into(iter.into_iter().map(|x| x.into_f64()).collect::<Vec<f64>>()).unwrap()),
            DType::I32 => self.tensors.insert(x, <Vec<i32> as TryInto<Tensor>>::try_into(iter.into_iter().map(|x| x.into_i32()).collect::<Vec<i32>>()).unwrap()),
        };
        Ok(())
    }

    fn evaluate(
        &mut self,
        mut rcs: BTreeMap<Id, u32>,
        order: &[Id],
        nodes: &[Node],
    ) -> Result<(), ZyxError> {
        for nid in order.iter().copied() {
            //std::println!("Processing: {:?}", nodes[nid.i()]);
            match &nodes[nid.i()] {
                Node::Leaf(..) => {}
                Node::Uniform(..) => { todo!() }
                Node::Cast(x, dtype) => {
                    self.tensors.insert(nid, match dtype {
                        DType::F32 => self.tensors[x].to_kind(Kind::Float),
                        DType::F64 => self.tensors[x].to_kind(Kind::Double),
                        DType::I32 => self.tensors[x].to_kind(Kind::Int),
                    });
                }
                Node::Detach(x) => {
                    self.tensors.insert(nid, self.tensors[x].copy());
                }
                Node::Neg(x) => {
                    self.tensors.insert(nid, self.tensors[x].neg());
                }
                Node::ReLU(x) => {
                    self.tensors.insert(nid, self.tensors[x].relu());
                }
                Node::Sin(x) => {
                    self.tensors.insert(nid, self.tensors[x].sin());
                }
                Node::Cos(x) => {
                    self.tensors.insert(nid, self.tensors[x].cos());
                }
                Node::Ln(x) => {
                    self.tensors.insert(nid, self.tensors[x].log());
                }
                Node::Exp(x) => {
                    self.tensors.insert(nid, self.tensors[x].exp());
                }
                Node::Tanh(x) => {
                    self.tensors.insert(nid, self.tensors[x].tanh());
                }
                Node::Sqrt(x) => {
                    self.tensors.insert(nid, self.tensors[x].sqrt());
                }
                Node::Add(x, y) => {
                    self.tensors.insert(nid, &self.tensors[x] + &self.tensors[y]);
                }
                Node::Sub(x, y) => {
                    self.tensors.insert(nid, &self.tensors[x] - &self.tensors[y]);
                }
                Node::Mul(x, y) => {
                    self.tensors.insert(nid, &self.tensors[x] * &self.tensors[y]);
                }
                Node::Div(x, y) => {
                    self.tensors.insert(nid, &self.tensors[x] / &self.tensors[y]);
                }
                Node::Pow(x, y) => {
                    self.tensors.insert(nid, self.tensors[x].pow(&self.tensors[y]));
                }
                Node::Cmplt(x, y) => {
                    let kind = self.tensors[x].kind();
                    self.tensors.insert(nid, self.tensors[x].lt_tensor(&self.tensors[y]).to_kind(kind));
                }
                Node::Where(..) => { todo!() }
                Node::Reshape(x, sh) => {
                    self.tensors.insert(nid, self.tensors[x].reshape(sh.vi64()));
                }
                Node::Expand(x, sh) => {
                    self.tensors.insert(nid, self.tensors[x].expand(sh.vi64(), true));
                }
                Node::Permute(x, axes, ..) => {
                    self.tensors.insert(nid, self.tensors[x].permute(axes.vi64()));
                }
                Node::Pad(x, padding, ..) => {
                    let padding: (Vec<i64>, Vec<i64>) = padding.iter().copied().unzip();
                    let padding: Vec<i64> = padding.0.into_iter().chain(padding.1).collect();
                    self.tensors.insert(nid, self.tensors[x].pad(&padding, &"constant", 0.0));
                }
                Node::Sum(x, axes, ..) => {
                    self.tensors.insert(nid, self.tensors[x].sum_dim_intlist(axes.vi64(), true, None));
                }
                Node::Max(x, axes, ..) => {
                    self.tensors.insert(nid, self.tensors[x].amax(axes.vi64(), true));
                }
            }
            for p in nodes[nid.i()].parameters() {
                if let Entry::Occupied(e) = rcs.entry(p).and_modify(|rc| *rc -= 1) {
                    if *e.get() == 0 {
                        self.remove(p)?;
                    }
                }
            }
        }
        Ok(())
    }
}
