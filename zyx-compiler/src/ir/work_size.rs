use alloc::{boxed::Box, vec::Vec, collections::BTreeSet};
use core::iter::repeat;
use zyx_core::axes::{Axes, IntoAxes};
use zyx_core::shape::Shape;
use zyx_core::view::View;

fn reshape_and_permute_kernel_args(mut ast_arg_views: Vec<View>, ast_shape: &Shape, ast_reduce_axes: &Option<Axes>) -> (Vec<View>, Shape, Option<usize>) {
    if let Some(reduce_axes) = ast_reduce_axes {
        // TODO pad sum reduce kernels with zeros and max reduce kernels with min_value
        // if they have dimensions that are not multiples of 16.
        // Although max reduce kernels may be padded with zeros in other dimensions,
        // so it may be little more complicated.
        let rank = ast_shape.rank();
        let permute_axes = (0..rank as i64)
            .filter(|a| !reduce_axes.contains(*a as usize))
            .chain(reduce_axes.iter().map(|a| *a as i64))
            .collect::<Box<_>>()
            .into_axes(rank);
        let shape = if rank > 4 || reduce_axes.len() > 1 {
            let d1: usize = ast_shape
                .iter()
                .enumerate()
                .filter_map(|(a, d)| {
                    if reduce_axes.contains(a) {
                        Some(*d)
                    } else {
                        None
                    }
                })
                .product();
            let d0 = ast_shape.numel() / d1;
            // TODO make these dimensions more reasonable, not just 1, 1, d0, d1
            let shape: Shape = [1, 1, d0, d1].into();
            for view in &mut ast_arg_views {
                *view = view.permute(&permute_axes).reshape(&shape);
            }
            shape
        } else {
            for view in &mut ast_arg_views {
                *view = view.permute(&permute_axes);
                let shape = view.shape();
                let shape = repeat(1).take(4-shape.rank()).chain(shape.iter().copied()).collect::<Vec<usize>>().into();
                *view = view.reshape(&shape);
            }
            let shape = ast_shape.permute(&permute_axes);
            let shape = repeat(1).take(4-shape.rank()).chain(shape.iter().copied()).collect::<Vec<usize>>().into();
            shape
        };
        let reduce_dim = shape[-1];
        (ast_arg_views, shape, Some(reduce_dim))
    } else {
        let shape = if ast_shape.rank() > 3 {
            let n = ast_shape.numel();
            for view in &mut ast_arg_views {
                *view = view.reshape(&[1, 1, n].into());
            }
            // TODO make these dimensions more reasonable, not just 1, 1, n
            // this is important when working with expanded buffers
            [1, 1, n].into()
        } else {
            for view in &mut ast_arg_views {
                let shape = view.shape();
                let shape = repeat(1).take(3-shape.rank()).chain(shape.iter().copied()).collect::<Vec<usize>>().into();
                *view = view.reshape(&shape);
            }
            repeat(1).take(3-ast_shape.rank()).chain(ast_shape.iter().copied()).collect::<Vec<usize>>().into()
        };
        (ast_arg_views, shape, None)
    }
}

/// Calculates arg_views, reduce dim size, global, local and register work sizes,
/// each across three dimensions.
/// Last work size is in reduce dimension if it is reduced kernel.
pub(super) fn calculate_work_sizes(
    ast_reduce_axes: &Option<Axes>,
    ast_shape: &Shape,
    ast_arg_views: Vec<View>,
    max_local_work_size: usize,
    _max_num_registers: usize,
) -> (
    Vec<View>,
    Shape,
    Option<usize>,
    Vec<usize>, // global work size
    Vec<usize>, // local work size
    Vec<usize>, // register work size
    BTreeSet<u8> // tiling axes
) {
    let max_local_work_size_dim = (max_local_work_size as f64).sqrt() as usize;
    // Register tiling can be disabled by setting max_register_work_size_dim to 1
    let max_register_work_size_dim = if ast_reduce_axes.is_some() { 4 } else { 1 };

    let (arg_views, shape, reduce_dim) = reshape_and_permute_kernel_args(ast_arg_views, ast_shape, ast_reduce_axes);
    let mut tiling_axes = select_tiling_axes(&arg_views, shape.rank());
    // Reduce dimension is always tiled
    if reduce_dim.is_some() && !tiling_axes.is_empty() {
        tiling_axes.insert(3);
    }
    // To disable tiling, just set empty tiling_axes here.

    let mut lws = 1;
    let mut register_work_size = Vec::new();
    let global_work_size: Vec<usize> = shape
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut d = *d;
            register_work_size.push(1);
            if tiling_axes.contains(&(i as u8)) {
                while d % 2 == 0 && register_work_size[i] * 2 <= max_register_work_size_dim {
                    register_work_size[i] *= 2;
                    d /= 2;
                }
            }
            d
        })
        .collect();

    // Runtimes are horrible at inferring local work sizes, we just have to give it our
    let local_work_size: Vec<usize> = global_work_size
        .iter()
        .zip(register_work_size.iter())
        .rev()
        .enumerate()
        .map(|(i, (gd, _rd))| {
            if reduce_dim.is_some() && i == 0 {
                let mut x = 1;
                let mut tiling_axes = tiling_axes.clone();
                if reduce_dim.is_some() {
                    tiling_axes.pop_last();
                }
                if tiling_axes.len() < 2 {
                    while gd % (x * 2) == 0 && x * lws < max_local_work_size {
                        x *= 2;
                    }
                } else {
                    while gd % (x * 2) == 0 && x < max_local_work_size_dim {
                        x *= 2;
                    }
                }
                x
            } else {
                let mut x = 1;
                let mut tiling_axes = tiling_axes.clone();
                if reduce_dim.is_some() {
                    tiling_axes.pop_last();
                }
                if tiling_axes.len() < 2 {
                    while gd % (x * 2) == 0 && x * lws < max_local_work_size {
                        x *= 2;
                    }
                } else {
                    while gd % (x * 2) == 0 && x < max_local_work_size_dim {
                        x *= 2;
                    }
                }
                lws *= x;
                x
            }
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    std::println!("global: {global_work_size:?}");
    std::println!("local: {local_work_size:?}");
    std::println!("register: {register_work_size:?}");
    (
        arg_views,
        shape,
        reduce_dim,
        global_work_size,
        local_work_size,
        register_work_size,
        tiling_axes
    )
}

/// Selects tiling axes so that we know which global (and possibly local) work dimensions
/// should be reduce by register tiling dimension.
/// Rank includes reduce dimension.
fn select_tiling_axes(arg_views: &[View], rank: usize) -> BTreeSet<u8> {
    let mut tiling_axes = BTreeSet::new();
    for view in arg_views {
        let strides = view.strides();
        let shape = view.shape();
        //std::println!("{shape}, {strides}, {rank}");
        let mut buffer_tiled_axes_count = 0;
        for a in (0..rank).rev() {
            if strides[a] == 0 && shape[a] > 1 && buffer_tiled_axes_count < 2 {
                buffer_tiled_axes_count += 1;
                tiling_axes.insert(a as u8);
            }
        }
    }
    tiling_axes
}
