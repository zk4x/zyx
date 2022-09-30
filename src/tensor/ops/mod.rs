mod to_vec;
mod get_shape;
mod relu;
mod exp;
mod ln;
mod tanh;
mod neg;
mod sum;
mod max;
mod min;
mod reshape;
mod expand;
mod permute;
mod add;
mod sub;
mod matmul;

/*impl<S> ops::Pow for Tensor<S>
where
    for<'a> &'a S: ops::Pow<Output = S>,
{
    type Output = Tensor<S>;
    fn pow(self, exponent: i32) -> Self::Output {
        Tensor {
            data: Rc::new(self.data.pow(exponent)),
        }
    }
}

impl<S> ops::Pow for TensorGrad<S>
where
    for<'a> &'a S: ops::Pow<Output = S>
        + std::ops::Mul<Output = S>
        + Add<Output = S>
        + ops::Pow<Output = S>
        + std::ops::Mul<i32, Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn pow(self, exponent: i32) -> Self::Output {
        let self_grad = Rc::downgrade(&self.grad);
        let self_data = self.data();
        TensorFunc {
            data: Rc::new(self_data.pow(exponent)),
            func: move |res_grad: | {
                
                    self_grad.replace_with(|grad| {
                        Rc::new(
                            grad.as_ref()
                                + &(res_grad.as_ref() * &(&self_data.pow(exponent - 1) * exponent)),
                        )
                    });
                }
            }))),
        }
    }
}

impl<S, F> ops::Pow for TensorFunc<S, F>
where
    F: FnOnce(S),
    for<'a> &'a S: ops::Pow<Output = S>
        + std::ops::Mul<Output = S>
        + ops::Pow<Output = S>
        + std::ops::Mul<i32, Output = S>,
{
    type Output = TensorFunc<S, impl FnOnce(S)>;
    fn pow(self, exponent: i32) -> Self::Output {
        let self_func = Rc::downgrade(&self.func);
        let self_data = self.data.clone();
        TensorFunc {
            data: Rc::new(self.data.pow(exponent)),
            func: move |res_grad: | {
                if let Some(func) = self_func
                    .upgrade()
                    .unwrap_or_else(|| Rc::new(Cell::new(None)))
                    .take()
                {
                    func(Rc::new(
                        res_grad.as_ref() * &(&self_data.pow(exponent - 1) * exponent),
                    ));
                }
            }))),
        }
    }
}*/
