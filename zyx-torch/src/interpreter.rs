use alloc::{
    collections::{btree_map::Entry, BTreeMap},
    vec::Vec,
};
use tch::{Kind, Tensor};
use zyx_core::dtype::DType;
use zyx_core::{error::ZyxError, node::Node, runtime::RuntimeBackend, scalar::Scalar, tensor::Id};

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
    fn is_empty(&self, x: Id) -> bool {
        !self.tensors.contains_key(&x)
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        //std::println!("Torch num tensors: {}", self.tensors.len());
        self.tensors.remove(&x);
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, _numel: usize) -> Result<Vec<T>, ZyxError> {
        Ok(match T::dtype() {
            DType::F32 => unsafe {
                core::mem::transmute::<Vec<f32>, Vec<T>>(
                    self.tensors[&x].data().flatten(0, -1).try_into().unwrap(),
                )
            },
            DType::F64 => unsafe {
                core::mem::transmute::<Vec<f64>, Vec<T>>(
                    self.tensors[&x].data().flatten(0, -1).try_into().unwrap(),
                )
            },
            DType::I32 => unsafe {
                core::mem::transmute::<Vec<i32>, Vec<T>>(
                    self.tensors[&x].data().flatten(0, -1).try_into().unwrap(),
                )
            },
        })
    }

    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
    where
        IT: IntoIterator<Item = T>,
        IT::IntoIter: ExactSizeIterator,
    {
        match T::dtype() {
            DType::F32 => self.tensors.insert(
                x,
                <Vec<f32> as TryInto<Tensor>>::try_into(
                    iter.into_iter().map(|x| x.into_f32()).collect::<Vec<f32>>(),
                )
                .unwrap(),
            ),
            DType::F64 => self.tensors.insert(
                x,
                <Vec<f64> as TryInto<Tensor>>::try_into(
                    iter.into_iter().map(|x| x.into_f64()).collect::<Vec<f64>>(),
                )
                .unwrap(),
            ),
            DType::I32 => self.tensors.insert(
                x,
                <Vec<i32> as TryInto<Tensor>>::try_into(
                    iter.into_iter().map(|x| x.into_i32()).collect::<Vec<i32>>(),
                )
                .unwrap(),
            ),
        };
        Ok(())
    }
}
