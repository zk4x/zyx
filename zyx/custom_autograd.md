# Custom Backward Plan

## Problem

`gradient()` holds `RT.lock()` during the entire backward pass. Calling `RT.lock()` inside (for Tensor ops) deadlocks (non-reentrant spinlock). We need the user's custom backward to use normal Tensor ops.

## Solution: drop/re-acquire the lock around the user's backward call

### 1. Fix `try_lock()` in `mutex.rs`

Currently `try_lock()` never returns `Err` — it spins forever like `lock()`. Change to a single CAS attempt:

```rust
pub fn try_lock(&self) -> Result<MutexGuard<'_, T>, ()> {
    if self.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
        Ok(MutexGuard { mutex: self })
    } else {
        Err(())
    }
}
```

This makes `Tensor::drop`'s `try_lock()` actually skip release when we hold the lock — essential for safety.

### 2. Registration struct + map

On `Runtime` (`runtime.rs`):

```rust
pub(super) struct CustomGradRegistration {
    pub inputs: Vec<TensorId>,
    pub saved: Vec<TensorId>,
    pub backward: fn(&[&Tensor], &[&Tensor]) -> Vec<Option<Tensor>>,
}

pub struct Runtime {
    // ... existing fields ...
    pub custom_backwards: Map<TensorId, CustomGradRegistration>,
}
```

### 3. `Tensor::set_custom_backward` (tensor/mod.rs)

```rust
impl Tensor {
    pub fn set_custom_backward(
        &self,
        inputs: &[&Tensor],
        saved: &[&Tensor],
        backward: fn(&[&Tensor], &[&Tensor]) -> Vec<Option<Tensor>>,
    ) {
        let mut rt = RT.lock();
        for &t in inputs { rt.graph.retain(t.id()); }
        for &t in saved  { rt.graph.retain(t.id()); }
        rt.custom_backwards.insert(self.id(), CustomGradRegistration {
            inputs: inputs.iter().map(|t| t.id()).collect(),
            saved: saved.iter().map(|t| t.id()).collect(),
            backward,
        });
    }
}
```

### 4. `Graph::build_topo` gets `custom_stops` (graph/mod.rs)

Add `custom_stops: &Set<TensorId>` parameter. In the walk loop:

```rust
if tape.contains(&nid) && !custom_stops.contains(&nid) {
    params.extend(self.nodes[nid].1.parameters());
}
```

### 5. Restructure `gradient()` (autograd.rs)

Restructure so `gradient_persistent` locks RT and passes the guard to `gradient_impl`, which can drop/reacquire it:

```rust
// GradientTape::gradient_persistent
pub fn gradient_persistent<'a>(
    &self,
    target: &Tensor,
    sources: impl IntoIterator<Item = &'a Tensor>,
) -> Vec<Option<Tensor>> {
    let sources: Vec<TensorId> = sources.into_iter().map(Tensor::id).collect();
    let sources_set: Set<TensorId> = sources.iter().copied().collect();
    let guard = RT.lock();
    let result = gradient_impl(guard, target.id(), &sources_set, &sources);
    result
}
```

`gradient_impl` owns the guard, can drop and reacquire:

```rust
fn gradient_impl(
    mut rt: MutexGuard<'_, Runtime>,
    x: TensorId,
    sources: &Set<TensorId>,
    source_list: &[TensorId],
) -> Vec<Option<Tensor>> {
    fn insert_or_add_grad(
        r: &mut Runtime,
        grads: &mut Map<TensorId, TensorId>,
        nid: TensorId,
        grad: TensorId,
    ) {
        match grads.entry(nid) {
            std::collections::hash_map::Entry::Vacant(e) => { e.insert(grad); }
            std::collections::hash_map::Entry::Occupied(e) => {
                let (k, prev_grad) = e.remove_entry();
                grads.insert(k, r.graph.push(Node::Binary {
                    x: prev_grad, y: grad, bop: BOp::Add,
                }));
                r.release(prev_grad);
                r.release(grad);
            }
        }
    }

    let custom_stops: Set<TensorId> = rt.custom_backwards.keys().copied().collect();
    let topo = rt.graph.build_topo(x, sources, &custom_stops);
    let req_grad: Set<TensorId> = topo.iter().copied().chain(sources.iter().copied()).collect();

    let mut grads: Map<TensorId, TensorId> = Map::default();
    grads.insert(x, rt.ones(rt.shape(x).into(), rt.dtype(x)));

    for &nid in &topo {
        let grad = grads[&nid];

        // --- Custom backward dispatch ---
        if let Some(reg) = rt.custom_backwards.get(&nid) {
            let saved_ids: Vec<TensorId> = reg.saved.clone();
            let input_ids: Vec<TensorId> = reg.inputs.clone();
            let backward = reg.backward;
            let grad_id = grad;

            drop(rt);  // release lock

            // ManuallyDrop wrappers — no refcount manipulation
            let saved: Vec<ManuallyDrop<Tensor>> = saved_ids
                .iter().map(|&id| ManuallyDrop::new(Tensor { id })).collect();
            let grad_t = ManuallyDrop::new(Tensor { id: grad_id });
            let saved_refs: Vec<&Tensor> = saved.iter().map(|md| &**md).collect();
            let grads_refs = vec![&*grad_t];
            let input_grads = (backward)(&saved_refs, &grads_refs);

            // Extract TensorIds with mem::forget to prevent Drop from running
            let result_ids: Vec<Option<TensorId>> = input_grads.into_iter()
                .map(|opt| opt.map(|t| { let id = t.id(); std::mem::forget(t); id }))
                .collect();

            rt = RT.lock();  // re-acquire

            for (i, g) in result_ids.into_iter().enumerate() {
                let input_id = input_ids[i];
                if let Some(grad_id) = g {
                    if sources.contains(&input_id) {
                        insert_or_add_grad(&mut rt, &mut grads, input_id, grad_id);
                    } else {
                        rt.release(grad_id);
                    }
                }
            }
            continue;
        }

        // --- Standard backward (unchanged from current code) ---
        match rt.graph[nid] {
            // ... all existing ops: Binary, Unary, Cast, Reshape, Expand, Permute, Pad, Reduce ...
            Node::Const { .. } | Node::Leaf { .. } | Node::Custom(_) => {}
        }
    }

    // Extract source gradients
    let mut res = Map::default();
    for (k, v) in grads {
        if sources.contains(&k) {
            res.insert(k, v);
        } else {
            rt.release(v);
        }
    }

    source_list.iter().map(|sid| {
        res.get(sid).copied().map(|id| {
            rt.graph.retain(id);
            Tensor { id }
        })
    }).collect()
}
```

### 6. Cleanup

In `Runtime::release()` (`runtime.rs`), remove the custom backward entry when the tensor's refcount reaches 0:

```rust
pub(super) fn release(&mut self, x: TensorId) {
    self.custom_backwards.remove(&x);
    let to_remove = self.graph.release(&[x]);
    deallocate_tensors(
        &to_remove,
        &mut self.pools,
        &mut self.events,
        &mut self.buffer_map,
        &mut self.temp_data,
    );
    if self.graph.is_empty() && self.buffer_map.is_empty() {
        self.deinitialize();
    }
}
```

Also release the retained refcounts on the inputs/saved when removing a registration (e.g., in `drop_gradient_tape`):

```rust
pub(super) fn drop_gradient_tape(&mut self) {
    self.graph.gradient_tape_ref_count -= 1;
    if self.graph.gradient_tape_ref_count == 0 {
        // Release retained refs for custom backwards
        for (_, reg) in self.custom_backwards.drain() {
            for id in reg.inputs { self.graph.release(&[id]); }
            for id in reg.saved { self.graph.release(&[id]); }
        }
        // ... rest of existing cleanup ...
    }
}
```

### 7. User API

```rust
fn straight_through(
    saved: &[&Tensor],
    grads: &[&Tensor],
) -> Vec<Option<Tensor>> {
    vec![Some(*grads[0])]
}

let y = x.sin() * x.cos();
y.set_custom_backward(&[&x], &[&x], straight_through);
let [dx] = tape.gradient(&y, [&x])?;
```

Or with non-capturing closure:

```rust
y.set_custom_backward(&[&x], &[&x], |s, g| {
    let mask = s[0].cmpgt(&Tensor::from(0.0));
    vec![Some(g[0] * mask)]
})?;
```

### Files Changed

| File | What |
|------|------|
| `mutex.rs` | `try_lock()` returns `Err` on contention |
| `runtime.rs` | Add `custom_backwards` field + `CustomGradRegistration` struct; update `release()`, `drop_gradient_tape()`, `new()` |
| `graph/mod.rs` | `build_topo` gets `custom_stops: &Set<TensorId>` param |
| `autograd.rs` | Restructure `gradient` → `gradient_impl` owning guard, add custom dispatch before standard dispatch |
| `tensor/mod.rs` | Add `Tensor::set_custom_backward` |
