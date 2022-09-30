use crate::{ops, tensor::{Tensor, TensorGrad, TensorFunc}};
use std::rc::Rc;

impl<S> ops::ReLU for Tensor<S>
where
    for<'a> &'a S: ops::ReLU<Output = S>,
{
    type Output = Tensor<S>;
    fn relu(self) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.relu()),
        }
    }
}

// Maybe it will be needed to just rewrite everything in this way
// It will make TensorFunc clonable and also allow us to have nicer
// print output while printing TensorFunc
/*trait Backward<S> {
    fn backward(self, res_grad: S);
}

struct ReLUBackward<'a, S> {
    self_grad: &'a std::cell::RefCell<S>,
    self_data: Rc<S>,
}

impl<'a, S> Clone for ReLUBackward<'a, S> {
    fn clone(&self) -> Self {
        Self {
            self_grad: self.self_grad,
            self_data: self.self_data.clone(),
        }
    }
}

impl<'a, S> Backward<S> for ReLUBackward<'a, S>
where
    for<'b> &'b S: ops::ReLU<Output = S>
        + ops::DReLU<Output = S>
        + std::ops::Mul<Output = S>
        + std::ops::Add<Output = S>,
{
    fn backward(self, res_grad: S) {
        use ops::DReLU;
        self.self_grad.replace_with(|grad| &*grad + &(&res_grad * &self.self_data.drelu()));
    }
}*/

impl<'g, S> ops::ReLU for &'g TensorGrad<S>
where
    S: 'g,
    for<'a> &'a S: ops::ReLU<Output = S>
        + ops::DReLU<Output = S>
        + std::ops::Mul<Output = S>
        + std::ops::Add<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn relu(self) -> Self::Output {
        use ops::DReLU;
        let self_grad = &self.grad;
        let self_data = self.data();
        TensorFunc {
            data: Rc::new(self_data.relu()),
            func: move |res_grad: S| { self_grad.replace_with(|grad| &*grad + &(&res_grad * &self_data.drelu())); },
        }
    }
}

impl<S, F> ops::ReLU for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::ReLU<Output = S> + ops::DReLU<Output = S> + std::ops::Mul<Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn relu(self) -> Self::Output {
        use ops::DReLU;
        let self_func = self.func;
        let self_data = self.data;
        TensorFunc {
            data: Rc::new(self_data.relu()),
            func: move |res_grad| self_func(&res_grad * &self_data.drelu()),
        }
    }
}