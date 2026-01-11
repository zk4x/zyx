use crate::{ZyxError, backend::Device, graph::Graph, runtime::Pool, shape::Dim, tensor::TensorId};
use std::collections::BTreeSet;

pub fn schedule(
    loads: &[TensorId],
    stores: &[TensorId],
    graph: &Graph,
    devices: &[Device],
    pools: &mut [Pool],
) -> Result<
    (
        usize,
        usize,
        Vec<crate::backend::Event>,
        BTreeSet<crate::backend::BufferId>,
        Vec<crate::backend::BufferId>,
    ),
    ZyxError,
> {
    let required_stores_memory: Dim = stores
        .iter()
        .map(|&tid| graph.shape(tid).iter().product::<Dim>() * graph.dtype(tid).byte_size() as Dim)
        .sum::<Dim>();
    let mut dev_ids: Vec<usize> = (0..devices.len()).collect();
    dev_ids.sort_unstable_by_key(|&dev_id| devices[dev_id].free_compute());
    dev_ids.reverse();
    let mut device_id = None;
    for dev_id in dev_ids {
        let mpid = devices[dev_id].memory_pool_id() as usize;
        // Check if kernel arguments fit into associated memory pool
        let free_memory = pools[mpid].pool.free_bytes();
        // required memory is lowered by the amount of tensors already stored in that memory pool
        let missing_loads_memory = loads
            .iter()
            .map(|tid| {
                if pools[mpid].buffer_map.contains_key(tid) {
                    0
                } else {
                    graph.shape(*tid).iter().product::<Dim>() * graph.dtype(*tid).byte_size() as Dim
                }
            })
            .sum::<Dim>();
        //println!("Free memory {free_memory} B, missing loads memory {missing_loads_memory} B");
        let required_memory = required_stores_memory + missing_loads_memory;
        if free_memory > required_memory {
            device_id = Some(dev_id);
            break;
        }
    }
    let Some(dev_id) = device_id else {
        return Err(ZyxError::AllocationError(
            format!("no device has enough memory to store {required_stores_memory} B for intermedite tensors.").into(),
        ));
    };
    let _ = device_id;
    let mpid = devices[dev_id].memory_pool_id() as usize;
    let mut event_wait_list = Vec::new();
    for &tid in loads {
        if !pools[mpid].buffer_map.contains_key(&tid) {
            if !pools[mpid].buffer_map.contains_key(&tid) {
                // Check where the tensor is
                let mut old_mpid = usize::MAX;
                for (i, pool) in pools.iter().enumerate() {
                    if pool.buffer_map.contains_key(&tid) {
                        old_mpid = i;
                        break;
                    }
                }
                debug_assert_ne!(old_mpid, usize::MAX);

                let bytes = graph.shape(tid).iter().product::<Dim>() * graph.dtype(tid).byte_size() as Dim;
                // No need to initialize here, other than rust is bad.
                let mut byte_slice = vec![0u8; bytes as usize];

                let src = pools[old_mpid].buffer_map[&tid];
                println!("Loading tensor {tid:?} at buffer id {src:?}");

                // Move the tensor from old pool into temporary in RAM
                // TODO later we can implement direct GPU to GPU movement, it's easy here,
                // a bit harder for the backends.
                // Pool to host blocks on event, so we can remove that event.
                let mut event_wait_list = Vec::new();
                for buffers in pools[old_mpid].events.keys() {
                    if buffers.contains(&src) {
                        let buffers = buffers.clone();
                        // Pool to host blocks on event, so we can remove that event.
                        let event = pools[old_mpid].events.remove(&buffers).unwrap();
                        event_wait_list.push(event);
                        break;
                    }
                }
                pools[old_mpid].pool.pool_to_host(src, &mut byte_slice, event_wait_list)?;

                // Delete the tensor from the old pool
                pools[old_mpid].pool.deallocate(src, vec![]);
                pools[old_mpid].buffer_map.remove(&tid);
                //println!("{byte_slice:?}");

                let (dst, event) = pools[mpid].pool.allocate(bytes)?;
                let event = pools[mpid].pool.host_to_pool(&byte_slice, dst, vec![event])?;
                // We have to sync here, because byte_slice does not exist any more.
                // The other solution would be to put this into temp_data.
                // But perhaps we should figure some better async.
                pools[mpid].pool.sync_events(vec![event])?;
                pools[mpid].buffer_map.insert(tid, dst);
                //memory_pools[mpid].events.insert(BTreeSet::from([dst]), event);
            }
        }
    }
    let mut output_buffers = BTreeSet::new();
    for &tid in stores {
        let bytes = graph.shape(tid).iter().product::<Dim>() * graph.dtype(tid).byte_size() as Dim;
        let (buffer_id, event) = pools[mpid].pool.allocate(bytes)?;
        pools[mpid].buffer_map.insert(tid, buffer_id);
        event_wait_list.push(event);
        output_buffers.insert(buffer_id);
    }
    let mut args = Vec::new();
    for tid in loads {
        args.push(pools[mpid].buffer_map[tid]);
    }
    for tid in stores {
        args.push(pools[mpid].buffer_map[tid]);
    }
    Ok((dev_id, mpid, event_wait_list, output_buffers, args))
}
