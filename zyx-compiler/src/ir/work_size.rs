use alloc::{boxed::Box, vec::Vec, collections::BTreeSet};
use zyx_core::axes::{Axes, IntoAxes};
use zyx_core::shape::Shape;
use zyx_core::view::View;

fn reshape_and_permute_kernel_args(mut ast_arg_views: Vec<View>, ast_shape: &Shape, ast_reduce_axes: &Option<Axes>) -> (Vec<View>, Shape, Option<usize>) {
    if let Some(reduce_axes) = ast_reduce_axes {
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
            let shape: Shape = [d0, d1].into();
            for view in &mut ast_arg_views {
                *view = view.permute(&permute_axes).reshape(&shape);
            }
            shape
        } else {
            for view in &mut ast_arg_views {
                *view = view.permute(&permute_axes);
            }
            ast_shape.permute(&permute_axes)
        };
        let reduce_dim = shape[-1];
        (ast_arg_views, shape, Some(reduce_dim))
    } else {
        let mut arg_views = ast_arg_views.clone();
        let shape = if ast_shape.rank() > 3 {
            let n = ast_shape.numel();
            for view in &mut arg_views {
                *view = view.reshape(&[n].into());
            }
            [n].into()
        } else {
            ast_shape.clone()
        };
        (arg_views, shape, None)
    }
}

/// Calculates arg_views, reduce dim size, global, local and register work sizes,
/// each across three dimensions
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
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
) {
    let (arg_views, shape, reduce_dim) = reshape_and_permute_kernel_args(ast_arg_views, ast_shape, ast_reduce_axes);

    let mut lws = 1;
    let rank = shape.rank();
    let mut _register_work_size = Vec::new();
    let mut global_work_size: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, d)| {
            if reduce_dim.is_some() && i == rank - 1 {
                None
            } else {
                let d = *d;
                /*register_work_size.push(1);
                while d % 2 == 0 && register_work_size[i] * 2 <= max_register_work_size {
                    register_work_size[i] *= 2;
                    d /= 2;
                }*/
                Some(d)
            }
        })
        .collect();
    //let mut full_reduce = false; // reduce across all axes
    if global_work_size.len() == 0 {
        //full_reduce = true;
        global_work_size.push(1);
    }
    // Runtimes are horrible at inferring local work sizes, we just have to give it our
    let local_work_size: Vec<usize> = global_work_size
        .iter()
        .rev()
        .map(|d| {
            let mut x = 1;
            while d % (x * 2) == 0 && x * lws < max_local_work_size {
                x *= 2;
            }
            lws *= x;
            x
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    (
        arg_views,
        shape,
        reduce_dim,
        global_work_size,
        local_work_size,
        _register_work_size,
    )
}

/// Which buffers should be tiled in the reduce kernel?
fn choose_tiled_buffers(arg_views: &[View]) -> BTreeSet<u8> {
    todo!()
}
