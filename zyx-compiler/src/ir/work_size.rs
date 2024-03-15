use alloc::{boxed::Box, vec::Vec, collections::BTreeSet};
use core::iter::repeat;
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
        let mut arg_views = ast_arg_views.clone();
        let shape = if ast_shape.rank() > 3 {
            let n = ast_shape.numel();
            for view in &mut arg_views {
                *view = view.reshape(&[1, 1, n].into());
            }
            // TODO make these dimensions more reasonable, not just 1, 1, n
            // this is important when working with expanded buffers
            [1, 1, n].into()
        } else {
            repeat(1).take(3-ast_shape.rank()).chain(ast_shape.iter().copied()).collect::<Vec<usize>>().into()
        };
        (arg_views, shape, None)
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
    BTreeSet<u8>, // tiled buffers
    BTreeSet<usize> // tiling axes
) {
    let max_local_work_size_dim = (max_local_work_size as f64).sqrt() as usize;
    let max_register_work_size_dim = if ast_reduce_axes.is_some() { 1 } else { 1 };

    let (arg_views, shape, reduce_dim) = reshape_and_permute_kernel_args(ast_arg_views, ast_shape, ast_reduce_axes);
    let (tiled_buffers, tiling_axes) = select_tiled_buffers_and_tiling_axes(&arg_views, shape.rank());

    let mut lws = 1;
    let mut register_work_size = Vec::new();
    let mut global_work_size: Vec<usize> = shape
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut d = *d;
            register_work_size.push(1);
            if tiling_axes.contains(&i) {
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
        .map(|(i, (gd, rd))| {
            if reduce_dim.is_some() && i == 0 {
                1
            } else {
                let mut x = 1;
                if tiling_axes.len() < 2 {
                    while gd % (x * rd * 2) == 0 && x * lws < max_local_work_size {
                        x *= 2;
                    }
                } else {
                    while gd % (x * rd * 2) == 0 && x < max_local_work_size_dim {
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
    (
        arg_views,
        shape,
        reduce_dim,
        global_work_size,
        local_work_size,
        register_work_size,
        tiled_buffers,
        tiling_axes
    )
}

/// Which buffers should be tiled in the reduce kernel?
fn select_tiled_buffers_and_tiling_axes(arg_views: &[View], rank: usize) -> (BTreeSet<u8>, BTreeSet<usize>) {
    // All expanded buffers are tiled
    let mut tiled_buffers = BTreeSet::new();
    for (id, view) in arg_views.iter().enumerate() {
        if (0..rank).any(|a| view.is_expanded_axis(a)) {
            tiled_buffers.insert(id as u8);
        }
    }
    // Tiling axes are all axes where at least one of tiled_buffers is expanded
    // and also one common axis where none of tiled buffers is expanded.
    // TODO if all buffers are expanded in one axis, then this axis should be removed
    // from the kernel alltogether
    let mut tiling_axes = BTreeSet::new();
    for a in 0..rank {
        let mut expanded_axis = false;
        for (id, view) in arg_views.iter().enumerate() {
            if tiled_buffers.contains(&(id as u8)) {
                if view.is_expanded_axis(a) {
                    expanded_axis = true;
                }
            }
        }
        // If exists at least one buffer expanded in this axis
        if expanded_axis {
            tiling_axes.insert(a);
        }
    }
    // TODO here we add reduce axis to tiling axes, but is it always the best thing to do?
    tiling_axes.insert(rank-1);

    //std::println!("Tiled buffers: {tiled_buffers:?}, tiling axes: {tiling_axes:?}");
    (tiled_buffers, tiling_axes)
}
