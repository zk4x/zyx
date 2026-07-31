# Tape Drop & Graph Lifecycle — Design & Implementation Guide

This document records the design discussion for making `Tape::drop` minimal, and
how to implement it: the invariants, the debug asserts that enforce them, and the
reference counting that replaces the current linear scan.

## 1. Decision log

Decisions reached in the discussion:

1. **`Tape::drop` must be minimal.** No linear scan over all tensors. It only:
   removes the graph and converts leaf tensors back to eager mode.
2. **Invariants govern when tensors may be realized.** See §3.
3. **`realize`/`replay` own eager conversion.** They set their output tensors back
   to eager state after executing the plan. Drop never touches outputs.
4. **Generation safety without per-tensor storage.** A `ref_count` and a
   `dropped` flag live on the `Graph`, not on the tensor. The graph is kept in the
   slab as long as any tensor references it (id reuse becomes impossible while a
   stale tensor exists). This replaces the generational index idea.
5. **`release` reclaims dead intermediates.** When a graph tensor's rc hits 0, it
   is removed from the slab immediately (leaf exception). This is what makes "no
   scan in drop" sound.
6. **Promotion fix (a bug in current code).** An already-realized eager tensor
   being promoted must become a **leaf** directly. Today its buffer survives in
   `buffer_map` while it becomes a non-leaf graph tensor, violating I2.
7. **Invalid tensor usage panics** (full panic, not `debug_assert`): using a graph
   tensor whose tape scope has ended is a hard user error.
8. **Central `new_graph_tensor` helper.** All graph-tensor births go through one
   `Runtime` method that increments the graph's ref count in a single place.

## 2. The problem: current `Tape::drop`

`zyx/src/tape.rs:231-279`. It:

1. Removes the graph.
2. Scans **all** tensors in the runtime.
3. Per graph tensor: rc==0 → dealloc buffer + remove tensor; rc>0 with buffer →
   convert to eager via a fresh `LoadView` kernel; rc>0 without buffer → nothing
   (silently dangles).

Problems:

- **O(peak) scan + allocation per drop.** `Slab::values` never shrinks; `iter()`
  walks every slot with a set-lookup filter (slab.rs:144-150). Drop runs every
  training iteration and peak grows monotonically. It also allocates a `Vec` of
  all tids just to iterate.
- **No verification.** The TODO at tape.rs:236-239 ("debug assert that no
  intermediate tensors are kept alive") is unimplemented; the rc>0-without-buffer
  case dangles silently.
- **Mixed responsibilities.** Drop does graph teardown, dead-tensor freeing, *and*
  eager conversion — but conversion of boundary tensors belongs to `realize`.

## 3. Core invariants

**I1 — No computation on graph tensors during graph construction.**
Graph ops only build nodes. Computation *can* and *does* take place on **eager**
tensors (e.g. buffer materialization during promotion). This distinction is the
foundation: it guarantees I2.

**I2 — Before `realize`/`replay` runs:**
- the only realized tensors are **leaves**,
- **all** leaves are realized,
- **no** other graph tensors are realized.

**I3 — All computation happens only in `realize`/`replay`.**
A tape dropped without computation has never executed a kernel.

**I4 — After `execute_plan`:**
- no class has a buffer other than graph inputs (leaves) and outputs,
- all outputs are realized.

**I5 — `Drop` performs no computation and no scans.**

## 4. Design

### 4.1 Minimal drop

```rust
impl Drop for Tape {
    fn drop(&mut self) {
        let mut rt = RT.lock();
        // 1. Mark the graph dead.
        // 2. Walk leaf_map: convert alive leaves to eager, remove dead leaves.
        // 3. Remove the graph if nothing references it anymore.
    }
}
```

No scan. The only enumeration is `leaf_map`, which is already per-graph.

### 4.2 Promotion fix (runtime.rs `promote_to_graph`, line 484)

Current bug: a *realized* eager tensor (in `buffer_map`) being promoted keeps its
buffer and becomes a non-leaf graph tensor → violates I2.

Intended behavior:

- **If the tensor is in `buffer_map` → it is a leaf directly.** Create a `Leaf`
  class, insert into `leaf_map`/`leaf_classes`, map it to its tid. Its buffer is
  read by the plan as an input; its value is preserved and not recomputed.
  (A realized eager tensor is, by construction, independent of the graph — safe to
  treat as a constant input.)
- **Otherwise → replay the eager kernel on the graph** (current path): walk the
  kernel ops, materialize missing loads via `add_store`, so the realized *load*
  nodes become leaves and the promoted tensor becomes a graph intermediate.

Option (a) — dropping the buffer on promotion — was rejected.

### 4.3 ref_count + dropped flag on the Graph (instead of per-tensor generation)

`zyx/src/graph/mod.rs:291-300` — add to `Graph`:

```rust
pub struct Graph {
    // ...
    /// Number of alive tensors still referencing this graph.
    pub ref_count: u64,
    /// Tape scope has ended; no new ops may use this graph.
    pub dropped: bool,
}
```

Rules:

- **Increment** at every graph-tensor birth (central `new_graph_tensor`, §4.8).
- **Decrement** when a tensor referencing the graph dies (rc hits 0 in `release`)
  or is converted away from graph state (eagerification in `realize`/drop).
- The graph is removed from the slab only when `dropped && ref_count == 0`.

Consequences:

- A **live** graph tensor (rc>0) always has its graph present in the slab (it is
  counted). Ops therefore check `graphs[graph_id].dropped`, never `contains_key`.
- **GraphId reuse is impossible while a stale tensor exists** — the graph stays in
  the slab until the last referencing tensor dies. The dead graph *is* the
  tombstone. This is the generational-index guarantee without per-tensor bytes.
- Zero extra bytes per tensor; the u64 lives on the (few) graphs.

### 4.4 release reclaims dead intermediates (runtime.rs:223-259)

The current Graph branch (runtime.rs:226-233) only acts when the graph is *gone*:

```rust
TensorState::Graph { rc, graph_id, .. } => {
    *rc -= 1;
    if *rc == 0 && !self.graphs.contains_key(*graph_id) { /* remove */ }
}
```

New behavior at rc→0:

```rust
TensorState::Graph { rc, graph_id, .. } => {
    *rc -= 1;
    if *rc == 0 {
        if !is_leaf(x, graph_id) {
            debug_assert!(!self.buffer_map.contains_key(&x),
                "dead non-leaf graph tensor holds a buffer");
            self.tensors.remove(x);
        }
        self.graphs[graph_id].ref_count -= 1;
        if self.graphs[graph_id].dropped && self.graphs[graph_id].ref_count == 0 {
            self.remove_dead_graph(graph_id);
        }
    }
    return;
}
```

Notes:

- **Leaf exception.** A dead leaf (param dropped mid-scope) must stay in the slab
  and in `leaf_map`, because `realize`/`freeze` still read `leaf_map` →
  `buffer_map[tid]`. It is swept when the graph is removed.
- **`is_leaf` check is O(1) via the class.** `TensorState::Graph` carries the
  `class_id`, and during graph creation each class has exactly one node. So
  `Graph::is_leaf(class_id)` (checks the node type) is the check; no `leaf_tids`
  set needed.
- **The `debug_assert!(!buffer_map.contains_key(&x))` is valid only after** the
  promotion fix (§4.2) and after `realize` eagerifies outputs immediately (§4.5).
  Then no non-leaf graph tensor can ever hold a buffer.

`remove_dead_graph(graph_id)`:
1. Sweep dead leaves: for each tid in `leaf_map`, if still in `Graph` state for
   this graph → dealloc buffer (via `drain_events_for_buf`, plan.rs:191, made
   `pub(crate)`) + remove from the tensors slab.
2. `self.graphs.remove(graph_id)`.

### 4.5 Eagerify helper

Extract the current drop's eager-conversion (tape.rs:260-276) into a `Runtime`
method, reused by `realize` and `Tape::drop`:

```rust
pub fn eagerify(&mut self, tid: TensorId) {
    let (rc, graph_id) = match self.tensors[tid].state {
        TensorState::Graph { rc, graph_id, .. } => (rc, graph_id),
        _ => return, // already eager
    };
    let shape: Vec<Dim> = self.shape(tid).into();
    let dtype = self.dtype(tid);
    let op = Op::LoadView(Box::new((dtype, View::contiguous(&shape))));
    let kernel_id = self.kernels.push(KernelData {
        outputs: Vec::new(), loads: Vec::new(), stores: Vec::new(),
        kernel: Kernel::new(DeviceId::AUTO),
    });
    let op_id = self.kernels[kernel_id].kernel.push_back(op);
    self.kernels[kernel_id].loads.push(tid);
    self.tensors[tid].state = TensorState::Eager { kernel_id, op_id, pending: KernelId::NULL };
    for _ in 0..rc {
        self.kernels[kernel_id].outputs.push(tid);
    }
    self.graphs[graph_id].ref_count -= 1;
}
```

### 4.6 realize / freeze / replay

- **`realize`** (tape.rs:131-220): after `execute_plan` inserts the output buffers
  into `buffer_map`, call `eagerify` on each passed output tensor. Drop (which
  runs at the end of `realize`, since it consumes `self`) then only handles
  leaves. Leaves that were passed to realize are already eager — drop skips them.
- **`freeze`** (tape.rs:281-345): no computation, no eagerification. Its output
  tensors stay in graph state (rc>0, no buffer). They are intentionally invalid
  after freeze; they keep the dead graph alive until the user drops them.
  `FrozenTape::replay` creates fresh eager tensors per call and never touches the
  original graph.
- **`replay`** (tape.rs:354-374): unchanged, but benefits from the `execute_plan`
  asserts (§7).

### 4.7 Invalid tensor usage → panic (not assert)

For any live graph tensor the graph is guaranteed present, so ops check:

```rust
assert!(!self.graphs[graph_id].dropped,
    "tensor belongs to a tape scope that has ended \
     (Tape dropped or realized without this tensor being an output)");
```

Add this check (ideally via one small helper) at every place a graph tensor is
consumed: `promote_to_graph` (also giving it a proper dead-graph-aware message
instead of the unconditional `panic!()` at runtime.rs:489), and the graph-op
branches in `unary`, `binary`, `cast`, `bitcast`, `reduce`, movement ops.

This replaces the current misleading `"tensor was never realized..."` panic
(runtime.rs:819, 826).

### 4.8 Central `new_graph_tensor` helper

```rust
pub fn new_graph_tensor(
    &mut self, graph_id: GraphId, class_id: ClassId,
    shape_id: ShapeId, dtype: DType,
) -> TensorId {
    self.graphs[graph_id].ref_count += 1;
    self.tensors.push(TensorData {
        shape_id, dtype,
        state: TensorState::Graph { class_id, rc: 1, graph_id },
    })
}
```

Migrate all 13 graph-tensor birth sites to it (see §6 step 2). A missed increment
is the dangerous failure: ref_count too low → graph removed while a live tensor
still references it → `graphs[graph_id]` index panic at the worst moment.

## 5. Reference counting — full contract

| Event | Action | Site |
|---|---|---|
| Graph tensor born | `ref_count += 1` | `new_graph_tensor` + `promote_to_graph` |
| Graph tensor dies (rc→0) | `ref_count -= 1`; remove non-leaf from slab | `release` |
| Eagerified (realize output, drop leaf) | `ref_count -= 1` | `eagerify` |
| Tape dropped | `dropped = true`; eagerify alive leaves; remove dead leaves | `Tape::drop` |
| `dropped && ref_count == 0` | sweep dead leaves; remove graph | `release` / `Tape::drop` |

`retain`/clones do not touch `ref_count` — it counts distinct tensors, not
handles. A tensor with rc>0 counts once.

## 6. Implementation steps (in order)

> Order matters: the promotion fix must land first, otherwise the I2 asserts fire.

### Step 0 — Fix `promote_to_graph` (runtime.rs:484)

After the existing "already in this graph" check, add:

```rust
// Realized eager tensors become leaves directly (invariant I2).
if self.buffer_map.contains_key(&tid) {
    let (shape_id, dtype) = (self.tensors[tid].shape_id, self.tensors[tid].dtype);
    let (_, class_id) = self.push_leaf_node(graph_id, dtype, shape_id);
    self.graphs[graph_id].leaf_map.insert(class_id, tid);
    self.graphs[graph_id].leaf_classes.push(class_id);
    self.graphs[graph_id].leaf_tids.insert(tid);   // if using the set, §4.4

    let rc = self.kernels[kernel_id].outputs.iter().filter(|&&o| o == tid).count() as u32;
    self.kernels[kernel_id].outputs.retain(|&o| o != tid);
    self.tensors[tid].state = TensorState::Graph { class_id, rc, graph_id };
    self.graphs[graph_id].ref_count += 1;
    return Ok(class_id);
}
// else: existing "replay kernel on graph" path, ending with ref_count += 1.
```

Also make the "already in a different graph" branch panic clearly, checking
`dropped` first for a precise message.

### Step 1 — Graph fields (graph/mod.rs)

Add `ref_count: u64`, `dropped: bool`, and (recommended) `leaf_tids: Set<TensorId>`
to `Graph`; initialize in `Graph::new`.

### Step 2 — `new_graph_tensor` + migrate the 13 sites

Sites (all `tensors.push(TensorData { .. state: TensorState::Graph { rc: 1, .. } })`):

- runtime.rs: 703, 729, 756, 813, 919, 943, 1015, 1088, 1140, 1185
- autograd.rs: 553, 567
- promote_to_graph (runtime.rs:691 — the state transition)

Replace each with the helper. Grep for `TensorState::Graph` to confirm none are
missed.

### Step 3 — `release` (runtime.rs:223) + `remove_dead_graph`

Per §4.4. Reuse `drain_events_for_buf` (make it `pub(crate)` in plan.rs).

### Step 4 — `eagerify` helper

Per §4.5, on `Runtime`.

### Step 5 — `realize` / `freeze`

In `realize`, after the `execute_plan` + `buffer_map.insert` loop, call
`eagerify(tid)` for each output tid. `freeze` stays as-is.

### Step 6 — rewrite `Tape::drop` (tape.rs:231)

```rust
impl Drop for Tape {
    fn drop(&mut self) {
        let mut rt = RT.lock();
        let graph_id = self.graph_id;
        rt.graphs[graph_id].dead = true;

        let leaves: Vec<TensorId> = rt.graphs[graph_id].leaf_map.values().copied().collect();
        for tid in leaves {
            let (rc, gid) = match rt.tensors[tid].state {
                TensorState::Graph { rc, graph_id, .. } => (rc, graph_id),
                _ => continue, // already eagerified by realize
            };
            if gid != graph_id {
                continue;
            }
            if rc == 0 {
                // Dead leaf (dropped mid-scope): buffer was kept for realize;
                // ref_count already decremented at its release.
                if let Some(buf) = rt.buffer_map.remove(&tid) {
                    let wait = drain_events_for_buf(&mut rt.events, buf);
                    rt.pools[buf.pool].deallocate(buf.buffer, wait);
                }
                rt.tensors.remove(tid);
            } else if rt.buffer_map.contains_key(&tid) {
                rt.eagerify(tid);
            }
        }

        if rt.graphs[graph_id].ref_count == 0 {
            rt.remove_dead_graph(graph_id);
        }
    }
}
```

Notes:
- Leaves come from `leaf_map.values()` (no `leaf_tids` set; see §4.4).
- Realized leaves (buffer present) are eagerified; unrealized leaves held by the
  user stay in `Graph` state (rc>0, no buffer) and keep the dead graph alive until
  released.
- If user-held invalid tensors keep `ref_count > 0`, the dead graph stays; it is
  removed by `release` when the last one dies.

### Step 7 — invalid-tensor panics

Add the `dropped` panic (§4.7) at the graph-consumption points. Full `panic!`,
not `debug_assert!`.

### Step 8 — post-execute_plan buffer check (in `realize`)

After `execute_plan` runs in `Tape::realize` (both cache-hit and main paths), add:

```rust
rt.debug_assert_no_stray_buffers(graph_id, &output_tids);
```

`Runtime::debug_assert_no_stray_buffers` scans graph tensors in `Graph` state and
asserts that no non-leaf, non-output tensor holds a buffer (I4 at the tensor
level). No extra field on `ExecPlan`.

Note: this can surface a pre-existing leak — `ExecPlan::new` only emits
`Deallocate` for *inputs* whose rc hits 0 (plan.rs:92). A kernel-output class that
is not consumed by any extracted kernel and not in `output_set` is allocated and
never deallocated. Fix it when the check fails (add a `Deallocate` for unconsumed
kernel outputs, or prune such kernels in `Graph::extract`).

### Step 9 — invariant debug asserts

At the top of `Tape::realize` and `Tape::freeze`:

```rust
// I2: all leaves realized.
for &tid in rt.graphs[graph_id].leaf_tids.iter() {
    debug_assert!(rt.buffer_map.contains_key(&tid), "leaf {tid} not realized");
    debug_assert!(matches!(rt.tensors[tid].state,
        TensorState::Graph { graph_id: g, .. } if g == graph_id));
}
// I2: no non-leaf graph tensor is realized. (Debug-only scan, once per step.)
if cfg!(debug_assertions) {
    for (tid, td) in rt.tensors.iter() {
        if let TensorState::Graph { graph_id: g, .. } = td.state {
            if g == graph_id && !rt.graphs[graph_id].leaf_tids.contains(&tid) {
                debug_assert!(!rt.buffer_map.contains_key(&tid),
                    "non-leaf graph tensor {tid} realized before realize");
            }
        }
    }
}
```

These fail on today's promoted-realized bug until Step 0 lands.

### Step 10 — documentation

- Rewrite the module docs of `tape.rs` (lines 1-31) to state the invariants of §3,
  the drop contract, and the ref-count lifecycle.
- Comment `TensorState::Graph` (runtime.rs:119-123) with the lifecycle contract.
- Comment the `ref_count`/`dropped` fields on `Graph`.

### Step 11 — tests

In `zyx/tests/` (run from `zyx/zyx`):

- Tape dropped without realize → params usable eagerly; a held intermediate panics
  on use with the "tape scope has ended" message.
- `realize` → passed outputs are eager; leaves are eager after drop.
- Realized eager tensor promoted → becomes a leaf, value preserved, no
  recomputation, invariant asserts pass.
- Frozen tape: outputs invalid after freeze (panic on use); `replay` still produces
  correct results. (`FrozenTape::replay` had a bug — it never collected its output
  tids — fixed as part of this step.)
- `execute_plan` leftover-buffer check does not fire on existing plans (fix the
  leak first if it does).

## 7. Open questions / risks

- **Dead-leaf sweep correctness** relies on `leaf_tids`/`leaf_map` being complete;
  the debug asserts in Step 9 guard this.
- **Missed increment sites** are the main implementation hazard; Step 2 must end
  with a grep audit.
- **The `dropped` panic in ops** adds one field read per graph op; negligible.
- **Dead graphs linger** while the user holds invalid tensors — bounded by user
  behavior, removed on their release.
- Frozen outputs keeping the original graph alive is accepted; revisit if it shows
  up in profiles.
