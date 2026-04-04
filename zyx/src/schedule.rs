// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::{
    backend::{BufferId, Device, PoolBufferId, PoolId},
    graph::Graph,
    runtime::Pool,
    shape::Dim,
    tensor::TensorId,
    Map, ZyxError,
};
use std::collections::BTreeSet;

pub fn schedule(
    loads: &[TensorId],
    stores: &[TensorId],
    graph: &Graph,
    devices: &[Device],
    pools: &mut [Pool],
    buffer_map: &mut Map<TensorId, BufferId>,
) -> Result<
    (
        usize,
        usize,
        Vec<crate::backend::Event>,
        BTreeSet<PoolBufferId>,
        Vec<PoolBufferId>,
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
        let free_memory = pools[mpid].pool.free_bytes();
        let missing_loads_memory = loads
            .iter()
            .map(|tid| {
                if buffer_map.get(tid).map_or(false, |b| usize::from(b.pool) == mpid) {
                    0
                } else {
                    graph.shape(*tid).iter().product::<Dim>() * graph.dtype(*tid).byte_size() as Dim
                }
            })
            .sum::<Dim>();
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
    let mpid = devices[dev_id].memory_pool_id() as usize;

    let mut event_wait_list = Vec::new();
    for &tid in loads {
        let in_target = buffer_map.get(&tid).map_or(false, |b| usize::from(b.pool) == mpid);
        if !in_target {
            let Some(buf_id) = buffer_map.get(&tid) else {
                panic!("Tensor {tid:?} not found in any pool");
            };
            let old_mpid = usize::from(buf_id.pool);
            let src = buf_id.buffer;
            let bytes = graph.shape(tid).iter().product::<Dim>() * graph.dtype(tid).byte_size() as Dim;
            let mut byte_slice = vec![0u8; bytes as usize];

            let mut ev_wait = Vec::new();
            for buffers in pools[old_mpid].events.keys() {
                if buffers.contains(&src) {
                    let buffers = buffers.clone();
                    let event = pools[old_mpid].events.remove(&buffers).unwrap();
                    ev_wait.push(event);
                    break;
                }
            }
            pools[old_mpid].pool.pool_to_host(src, &mut byte_slice, ev_wait)?;

            buffer_map.remove(&tid);
            if !buffer_map.values().any(|b| b.buffer == src) {
                pools[old_mpid].pool.deallocate(src, vec![]);
            }

            let (dst, event) = pools[mpid].pool.allocate(bytes)?;
            let event = pools[mpid].pool.host_to_pool(&byte_slice, dst, vec![event])?;
            pools[mpid].pool.sync_events(vec![event])?;
            buffer_map.insert(tid, BufferId { pool: PoolId::from(mpid), buffer: dst });
        }
    }
    let mut output_buffers = BTreeSet::new();
    for &tid in stores {
        let bytes = graph.shape(tid).iter().product::<Dim>() * graph.dtype(tid).byte_size() as Dim;
        let (buffer_id, event) = pools[mpid].pool.allocate(bytes)?;
        buffer_map.insert(tid, BufferId { pool: PoolId::from(mpid), buffer: buffer_id });
        event_wait_list.push(event);
        output_buffers.insert(buffer_id);
    }
    let mut args = Vec::new();
    for tid in loads {
        args.push(buffer_map[tid].buffer);
    }
    for tid in stores {
        args.push(buffer_map[tid].buffer);
    }
    Ok((dev_id, mpid, event_wait_list, output_buffers, args))
}
