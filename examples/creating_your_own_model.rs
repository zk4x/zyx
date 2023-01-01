// This is a tutorial how to create your own model

// Decide where to do the computations.
use zyx::device::cpu; // let's use cpu

// Create your model
//
// 'd is the lifetime of the device where your parameters are stored.
// Since activations don't have parameters, they don't need lifetimes.
use zyx::nn; // nn module contains functors and layers
struct MyNet<'d, const IN: usize, const OUT: usize> {
    l1: nn::Linear<'d, IN, 1000>, // second generic parameter of Linear layer is size of the input and third is size of the output
    a1: nn::ReLU,
    l2: nn::Linear<'d, 1000, OUT>,
    a2: nn::Tanh,
}

use zyx::shape::{Ax2, Sh2}; // get access to shapes and axes
use zyx::tensor::*; // here we get access to Variable, Tensor and backward ops

// Implement nn::Module for your model to get forward function:
impl<'p, 'd, const IN: usize, const OUT: usize> nn::Module<'p, cpu::Buffer<'d, Sh2<1, IN>>>
    for MyNet<'d, IN, OUT>
where
    'd: 'p,
    // Beware the lifetimes here. Buffers live for device lifetime, while borrows of parameters
    // live for HasParameters<'p> lifetime. This is extremely important, otherwise borrow checker will complain
    // in the actual calling of parameters function.
    // Borrow of MyNet must be dropped as soon as Self::Params are dropped
{
    // Since the graph is created at compile time, it is necessary to write the whole backward pass here.
    // It is a very long type, but it is automatically inferred by the compiler, it just requires you to write it explicitly
    type Output = Tensor<
        cpu::Buffer<'d, Sh2<1, OUT>>,
        TanhBackwardT<
            cpu::Buffer<'d, Sh2<1, OUT>>,
            AddBackwardTV<
                'p,
                cpu::Buffer<'d, Sh2<1, OUT>>,
                MatMulBackwardTV<
                    'p,
                    cpu::Buffer<'d, Sh2<1000, OUT>>,
                    cpu::Buffer<'d, Sh2<1, 1000>>,
                    cpu::Buffer<'d, Sh2<1000, OUT>>,
                    ReLUBackwardT<
                        cpu::Buffer<'d, Sh2<1, 1000>>,
                        AddBackwardTV<
                            'p,
                            cpu::Buffer<'d, Sh2<1, 1000>>,
                            MatMulBackwardSV<
                                'p,
                                cpu::Buffer<'d, Sh2<1, IN>>,
                                cpu::Buffer<'d, Sh2<IN, 1000>>,
                            >,
                        >,
                    >,
                >,
            >,
        >,
    >;

    fn forward(&'p self, x: cpu::Buffer<'d, Sh2<1, IN>>) -> Self::Output {
        // use nn::ApplyModule to get access to the following syntax
        use nn::ApplyModule;

        // Without ApplyModule you would need to write the forward pass like this:
        // let x = self.l1.forward(x);
        // let x = self.a1.forward(x);
        // let x = self.l2.forward(x);
        // let x = self.a2.forward(x);
        // return x;

        // But with ApplyModule you can just use this syntax:
        x.apply(&self.l1)
            .apply(&self.a1)
            .apply(&self.l2)
            .apply(&self.a2)
    }
}

// We also need to get simple access to your Module's parameters:
// Again be carefull with lifetimes.
impl<'p, 'd, const IN: usize, const OUT: usize> nn::parameters::HasParameters<'p>
    for MyNet<'d, IN, OUT>
where
    'd: 'p,
{
    // Compiler will automatically infer the return type, but we still need to copy it here.
    // Note that all parameters have lifetime 'p which is generic parameter to HasParameters trait.
    type Params = (
        (
            &'p mut Variable<cpu::Buffer<'d, Sh2<IN, 1000>>>,
            &'p mut Variable<cpu::Buffer<'d, Sh2<1, 1000>>>,
        ),
        (
            &'p mut Variable<cpu::Buffer<'d, Sh2<1000, OUT>>>,
            &'p mut Variable<cpu::Buffer<'d, Sh2<1, OUT>>>,
        ),
    );

    // One advantage of such verbosity is that we can exactly see what are parameters of your network.
    // Then we can simply calculate the number of f32 numbers stored as parameters of your network:
    // IN * 1000 + 1 * 1000 + 1000 * OUT + 1 * OUT
    // We need to have at least twice that amount of free RAM in order to store gradients
    // and some intermediate buffers.

    fn parameters(&'p mut self) -> Self::Params {
        // return a tuple with all
        (self.l1.parameters(), self.l2.parameters())
    }
}

fn main() {
    // create your access point to the device
    let device = cpu::Device::default(); // will use the main cpu on your computer

    // Create and instance of your network with input size 50 and output dimension of 10
    let mut my_net = MyNet::<'_, 20, 5> {
        l1: nn::Linear::new(&device), // here we must tell the linear layers where they should store their parameters
        a1: nn::ReLU {},
        l2: nn::Linear::new(&device),
        a2: nn::Tanh {},
    };

    // now we can do the actual logic

    // zyx prelude gives us access to traits that allow us to call various functions, like operations (exp, matmul, etc.)
    // and buffer initialization functions like uniform and rand
    use zyx::prelude::*;

    // This is completely random dataset
    // Thanks to const shapes is the size of the input checked at compile time and is guaranteed to run without runtime errors
    // The first value in each tuple is input, the second value represents the correct output
    let dataset = vec![
        (
            device.buffer([[
                0.7507571, 0.7217026, 0.6980746, 0.7066041, 0.8864105, 0.61744523, 0.7969053,
                0.18718922, 0.9568634, 0.5328448, 0.92357934, 0.5294405, 0.7425239, 0.8428292,
                0.7640828, 0.9104457, 0.15281391, 0.06508052, 0.03200066, 0.20160127,
            ]]),
            device.buffer([[
                0.7507571, 0.7217026, 0.6980746, 0.7066041, 0.8864105, 0.61744523, 0.7969053,
                0.18718922, 0.9568634, 0.5328448,
            ]]),
        ),
        (
            device.buffer([[
                0.9024651, 0.7985842, 0.70478046, 0.7703233, 0.70260584, 0.32990253, 0.9954214,
                0.61785877, 0.7901434, 0.3326832, 0.21076417, 0.66955245, 0.8554325, 0.6663033,
                0.24682295, 0.4037094, 0.43269646, 0.48595917, 0.08413601, 0.06380451,
            ]]),
            device.buffer([[
                0.9024651, 0.7985842, 0.70478046, 0.7703233, 0.70260584, 0.32990253, 0.9954214,
                0.61785877, 0.7901434, 0.3326832,
            ]]),
        ),
        (
            device.buffer([[
                0.5865792, 0.6768781, 0.14213097, 0.5009351, 0.61870027, 0.54153633, 0.22999549,
                0.9271833, 0.42734194, 0.9949113, 0.83520794, 0.5763148, 0.2692014, 0.37496507,
                0.9917865, 0.74197793, 0.19458747, 0.6341822, 0.28314948, 0.8105364,
            ]]),
            device.buffer([[
                0.5865792, 0.6768781, 0.14213097, 0.5009351, 0.61870027, 0.54153633, 0.22999549,
                0.9271833, 0.42734194, 0.9949113,
            ]]),
        ),
        (
            device.buffer([[
                0.6747533, 0.05631852, 0.62192833, 0.81391823, 0.71077156, 0.8345542, 0.6213664,
                0.93497634, 0.8096837, 0.8661665, 0.5389962, 0.72498846, 0.67866004, 0.8322798,
                0.08960307, 0.24475288, 0.17254615, 0.85972965, 0.3564037, 0.8204987,
            ]]),
            device.buffer([[
                0.6747533, 0.05631852, 0.62192833, 0.81391823, 0.71077156, 0.8345542, 0.6213664,
                0.93497634, 0.8096837, 0.8661665,
            ]]),
        ),
        (
            device.buffer([[
                0.9909533, 0.74521554, 0.46223414, 0.5594574, 0.5624963, 0.10301626, 0.67522275,
                0.6146512, 0.9411292, 0.08030832, 0.7745633, 0.79155946, 0.6679106, 0.29721928,
                0.93910086, 0.9580766, 0.8319746, 0.854673, 0.56412864, 0.8099259,
            ]]),
            device.buffer([[
                0.9909533, 0.74521554, 0.46223414, 0.5594574, 0.5624963, 0.10301626, 0.67522275,
                0.6146512, 0.9411292, 0.08030832,
            ]]),
        ),
    ];

    // let's choose a loss function
    let mse_loss = (
        nn::MSELoss,
        nn::Mean::<Ax2<0, 1>>::new()
    );

    // Choose your optimizer
    let optimizer = zyx::optim::SGD::new().with_learning_rate(0.01);

    // Go over the whole dataset once (one epoch)
    for sample in &dataset {
        // Zero gradients of your parameters
        my_net.parameters().zero_grad();

        // forward pass
        let prediction = my_net.forward(sample.0.clone());

        // Correct output
        let correct_output = sample.1.clone();

        // Calculate loss using your loss function
        let loss = (prediction, correct_output).apply(&mse_loss);

        // Since you got the lifetimes correctly, if you call my_net.parameters().zero_grad() here,
        // it will not compile! The compiler knows, that loss has access to the gradients and does not allow
        // you to change them!

        // Calculate gradients for your parameters
        loss.backward();

        // Update your parameters using their gradients
        my_net.parameters().step(&optimizer);
    }
}
