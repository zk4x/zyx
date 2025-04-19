//! Converts graph to kernels and schedules them to devices

use crate::{
    backend::Device, graph::Graph, runtime::Pool, slab::Id, tensor::TensorId, DebugMask, Map, Set, ZyxError
};

#[allow(clippy::cognitive_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn schedule(
    graph: &Graph,
    order: &[TensorId],
    // RCS are only ref counts from parameters, excluding ref counts from being in to_eval/user rcs
    rcs: Map<TensorId, u32>,
    to_eval: &Set<TensorId>,
    devices: &mut [Device],
    memory_pools: &mut [Pool],
    optimizer: &mut KernelCache,
    search_iters: usize,
    realized_nodes: Set<TensorId>,
    debug: DebugMask,
) -> Result<(), ZyxError> {
    //let t = crate::Timer::new("realize_graph");
    let begin = std::time::Instant::now();
    let mut kernels = kernelize(
        graph,
        order,
        rcs,
        to_eval,
        memory_pools,
        &realized_nodes,
        debug,
    );

    let elapsed = begin.elapsed().as_micros();
    let mut min_ops = u32::MAX;
    let mut max_ops = 0;
    let mut avg_ops = 0;
    for kernel in kernels.values() {
        let n = u32::try_from(kernel.ops.len()).unwrap();
        if n > max_ops {
            max_ops = n;
        }
        if n < min_ops {
            min_ops = n;
        }
        avg_ops += n;
    }
    let kernels_len = kernels.len();
    if debug.perf() {
        println!("Scheduled {kernels_len} kernels, scheduling took {elapsed}us, ops per kernel: min: {min_ops}, max: {max_ops}, avg: {}", avg_ops/kernels_len);
    }
    //println!("Expand clones: {expa_u}, reshape clones: {resh_u}, pad clones: {pad_u}, permute clones: {perm_u}, reduce clones: {red_u}");
    // Timer
    /*for (name, (time, iters)) in crate::ET.lock().iter() {
        println!(
            "Timer {name} took {time}us for {iters} iterations, {}us/iter",
            time / iters
        );
    }*/

    // Check for small kernels (to improve performance)
    /*for kernel in kernels.values() {
        if kernel.ops.len() < 20 {
            kernel.debug();
        }
    }*/

    //panic!();

    let mut realized_nodes = realized_nodes;
    realized_nodes.extend(to_eval);

    // Launch all kernels
    let mut ids: Vec<Id> = kernels.ids().collect();
    while !ids.is_empty() {
        let mut i = 0;
        while i < ids.len() {
            let kid = ids[i];
            if kernels[kid].depends_on.is_empty() {
                ids.remove(i);
                let mut kernel = unsafe { kernels.remove_and_return(kid) };
                #[cfg(debug_assertions)]
                if !kernel.has_stores() {
                    kernel.debug();
                    panic!("Trying to launch kernel without stores");
                }
                let event =
                    kernel.launch(graph, devices, memory_pools, optimizer, search_iters, debug)?;
                for kernel in kernels.values_mut() {
                    kernel.depends_on.remove(&kid);
                }
                let loads: Set<TensorId> = kernel
                    .ops
                    .iter()
                    .filter_map(|op| {
                        if let Op::Load { x, .. } = op {
                            Some(kernel.tensors[x])
                        } else {
                            None
                        }
                    })
                    .collect();
                let mut loads: Set<TensorId> = loads.difference(&realized_nodes).copied().collect();
                for kernel in kernels.values() {
                    for tensor in kernel.tensors.values() {
                        loads.remove(tensor);
                    }
                }
                for tensor in loads {
                    for pool in &mut *memory_pools {
                        if let Some(buffer_id) = pool.buffer_map.remove(&tensor) {
                            let mut events = Vec::new();
                            for buffers in pool.events.keys() {
                                if buffers.contains(&buffer_id) {
                                    events.push(pool.events.remove(&buffers.clone()).unwrap());
                                    break;
                                }
                            }
                            // Push event from the current kernel
                            if let Some(event) = &event {
                                events.push(event.clone());
                            }
                            pool.pool.deallocate(buffer_id, events);
                        }
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    Ok(())
}
