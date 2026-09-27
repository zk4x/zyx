# ⚠️ FORBIDDEN, ABSOLUTELY: "pre-existing" (never use it) · "builds clean" (build silently) · "git checkout" (use `git restore`) · python/sed/awk/heredoc scripts for file edits (use the edit tool — scripts destroyed files twice already). READ FIRST, THEN ASK OR EDIT. ⚠️

# Agent Guidelines for zyx

## Know Your Limits

- **FIRST tool call of every session:** `rust_analyzer_rust_analyzer_set_workspace` (workspace: `/home/x/Dev/rust/zyx/zyx`).
- **READ CODE FIRST. ALWAYS.** Before any question, claim, or edit, open the relevant files. "I haven't read it" is never an answer. Reading is ITERATIVE: every newly raised detail gets investigated in source first; internal "questions" are research prompts. Only STOP and ask when the code genuinely underdetermines the answer (multiple valid designs, missing spec, intended semantics). Never ask what the code can answer; never implement on top of a guess.
- **On a hint from the user: read the implicated code, then report findings.** If a task references a `todo!()`/stub, read surrounding code before asking.
- **Design vs implementation.** The user owns design. When asked to analyze or compare, do it thoroughly; when implementing, do NOT pick implementation choices — ask.
- When the user says they'll do it themselves, stop. Only act when asked for a specific edit.
- If a task is hardware-specific, deeply subtle, or keeps failing, SAY SO plainly up front.
- **Reconsidering = ask.** If "reconsider"/"on reflection" appears in your thinking about a design or plan, immediately ask the user instead of reasoning onward.
- **Symbolic vs numeric:** dims/shapes/lengths/strides are usually `OpId`s (symbolic IR nodes), not `Dim`s. If a computation needs numeric values, that may signal the design intends symbolic ops — ask before writing code.
- **FORBIDDEN WORDS** (responses AND reasoning AND inner monologue, every token): "Actually", "wait", "key insight", "let me", "Hmm", "But wait". No exception for "it was in reasoning".

## Reasoning Discipline

- Keep premises, deductions, and assumptions explicit. Never promote an inference into a premise without stating it. If premises underdetermine the answer, report the underdetermination — don't resolve it by plausibility.

## No Workarounds

- Never paper over a bug by rerouting the caller away from the buggy path (e.g. swapping `index_select` for `slice`, dropping an API). Reproduce it in a test (e.g. `zyx/tests/`) and report it. If avoiding the path seems necessary, STOP and ask first.

## Optimization Pass Performance Budget

- **No pass in `zyx/src/kernel/` may exceed 30µs per invocation** (most run in single digits; 30µs is a ceiling, not a target). Applies to every pass and to `default_epilogue`/`default_optimizations` as a whole.
- A pass that blows this budget is a **performance bug** — it runs once per autotune variant and per compiled kernel, so hundreds of µs compound into seconds. It is almost always redundant or quadratic work (e.g. re-walking the IR per `If`, recomputing bounds — `compute_bounds` in verify.rs is the usual suspect; it must stay linear in op count and only guarantees conservative bounds). Fix the algorithmic cost; don't paper over it.
- Wrap passes in `time_pass!`/`Instant` measurement (stderr) to keep them observable; verify with a targeted test, not the full suite.

## Key Commands

Repo root has **no `Cargo.toml`** — crates are independent packages; **always run cargo from `zyx/`**.

```bash
cd zyx && cargo build
cd zyx && AGENT=1 cargo test               # AGENT=1 strips ANSI colors
cd zyx && AGENT=1 cargo test --test <file> <name>
cd zyx && AGENT=1 cargo test -- --nocapture   # ALWAYS add for ZYX_DEBUG/debug output
cd zyx && AGENT=1 cargo test --doc
cd zyx && cargo fmt
```

- Tenstorrent: `TT_METAL_ROOT=/home/x/Dev/cpp/tt-metal cargo build --features tenstorrent`
- **TT simulator** (no silicon required): the runtime process inherits the env, so export these before any `cargo test --features tenstorrent`:
  ```
  export TT_METAL_SIMULATOR=~/Dev/cpp/tt-sim/libttsim_bh.so   # .so must share a dir with soc_descriptor.yaml
  export TT_METAL_SLOW_DISPATCH_MODE=1                        # recommended; fast dispatch is under-tested
  ```
  See `~/Dev/cpp/tt-sim` (ttsim, https://github.com/tenstorrent/ttsim). Bit-exact vs silicon; avoid timing-dependent reductions.
- Clippy is NOT a gate — don't run it.
- Tests live in `zyx/tests/` as `{number}_{category}.rs` (+ `mnist.rs` with `mnist.py`/`.safetensors` fixtures). They return `Result<(), ZyxError>` and use `assert!`/`is_equal()` for floats.
- Python: use `python3.12`.

## Layout

```
zyx/           core tensor library (all the real work)
  src/backend/   one file per backend (c, cuda, hip, opencl, vulkan, disk, host, dummy, tenstorrent, wgpu)
  src/graph/     the egraph + autograd + kernelizer + plan
  src/kernel/    kernel IR, codegen, and ALL optimization passes
  tests/         integration tests
zyx-nn, zyx-optim, zyx-onnx, zyx-fuzzy, zyx-py, zyx-bench, zyx-book
examples/      virtual workspace; each model is its own crate
               (examples/data/ = datasets, examples/models/ = weights/gguf/configs)
```

Root docs: `ENV_VARS.md` (ZYX_DEBUG), `CONFIG.md`, `STYLE.md`, `ADDING_BACKENDS.md`.

## The Graph & Egraph

Lazy, dynamic, **one graph serves both laziness and autograd**; TensorId is `u32`. `Node` (`src/graph/mod.rs`) has 14 variants: Const, Leaf, Expand, Permute, Reshape, PadZeros, Flip, Reduce, Cast, Unary, Binary, Assign, ToDevice, Kernel.

The egraph lives in `src/graph/` and is **entirely OUTSIDE `src/kernel/`** — no imports or calls in either direction. Interface: the egraph picks among fusion variants of tensor ops; each chosen fusion compiles to a `Kernel` (linear-SSA seed) and goes to the autotuner; measured timings flow back for extraction. Everything in `kernel/` sees only plain kernels.

- `mod.rs` — the egraph. Heuristics generate a few fusion variants (CSE via hashconsing, algebraic/layout/shape rewrites). It has **NO cost model**: each variant is individually autotuned and `Graph::extract` picks by measured timing. Rewrites must be semantically exact — a wrong rewrite surfaces as a WRONG (not slow) kernel.
- **Vendor kernels** (backend language or zyx IR API) compete alongside heuristic variants and are extracted by the same timing-driven path.
- **Custom kernels BYPASS the egraph AND autotune.** Builder API (`Kernel::new` → `CompiledKernel::compile`, `kernel/custom.rs`) runs only `linearize` → `instruction_schedule` → `constant_folding` → `dead_code_elimination` → `verify` → `device.compile` (codegen runs; autotune passes don't). `BeamSearch::run_` must be called explicitly for autotuning. The egraph treats custom kernels as opaque fused units.
- `kernelizer.rs` — pattern-matches subgraphs → custom JIT kernels (add matches there); `autograd.rs` — gradients on the same graph; `plan.rs` — execution plans.
- `ZYX_DEBUG=2` prints the egraph after realize.

## Symbolic Dims

- **Eager AND graph must both work with symbolic dims everywhere.** No path may assume concrete shapes; every consumer (load kernels, autograd, passes, backends) handles them.
- Symbolic dims bottom out in `Param { Variable }` scalars (`IDX_T` = `DType::I64`). **Never cast dim/index values to `u32`** (overflow above 4.29×10⁹ silently corrupts codegen); `TensorId`/`OpId`/`ClassId` are `u32` only as identifiers. Any dim expression can be evaluated to a concrete `Constant`: walk the tree, fold via `Constant::unary`/`binary`, read variable slots at `Param { Variable }` leaves.
- NEVER fabricate sentinels or fallbacks (`-1`, `0`, `42`, ...) for unknown values — fail loudly (`debug_assert!`/`expect`/`todo!()`) at the exact spot, or ask. No `unwrap_or(<number>)`, no default arms. Load kernels must never emit fabricated consts as group lengths: evaluate the dim first.

## Code Style

- **Simplicity/debuggability > "clean"**: duplicate first, abstract later. Enums, not `dyn Trait`. Explicit returns. Minimize Arc/Rc. ~1000 LOC per module; new files only when necessary. Import order: `crate::` → `super::` → external.
- **Better `todo!()` than silent failure.**
- **No assumptions without a `debug_assert!`** — including user-guaranteed properties. No asserts = no assumption.
- **NEVER implement a bug knowingly — the single worst sin.** Every new `BOp`/`UOp` variant gets a CORRECT implementation in EVERY exhaustive match (IR, all codegen backends, constant-folding, autodiff, verify, string maps). Wrong semantics (e.g. `Cmpge` aliased to `>`) compiles fine and is still a bug. If you can't implement an arm correctly now, write `todo!()` — never an alias, never a silent no-op — or stop and ask. Adding a variant is complete only after `cargo build` AND an audit of every `match` over that type.
- When a symbol you assumed exists isn't found, widen the search (name variants, glob, callers) before concluding; only ask after a genuine exhaustive search. WHICH API to use is the user's call when options exist.
- New files: license header `// Copyright (C) 2025 zk4x` / `// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0`. Public items need docs; error constructors use `#[track_caller]` + file:line:column; use `From`.

## Backends

- Most backends are always compiled in, selected at runtime via config. **NEVER touch `~/.config/zyx/config.json`** (user-owned). Only `--features wgpu` and `--features tenstorrent` exist.
- Defaults with no config: C on, Dummy off, CUDA/HIP/Vulkan/OpenCL try to init and silently skip if the driver is missing; HIP always tries (skipped only without `libamdhip64.so`). If all fail, tests print nothing.

### HARDWARE SAFETY — the Tenstorrent board (READ BEFORE ANYTHING TT-RELATED)

**The board is HALF A YEAR of the user's income. Irreplaceable.** Every rule exists because an agent chained `tt-smi -r` with a device-init test (2026-09-02): the mid-wedge reset hard-rebooted the PC (power-level event) and flipped the board into fallback firmware.

- `cargo test` (including TT device init) is allowed freely. **Reset-class operations are the user's alone: the agent NEVER resets, flashes, or reinits the board** (`tt-smi -r`, `tt-flash`, firmware access) — not on approval, not ever. If a reset is needed, suggest it in one line and wait.
- **A failed device init WEDGES the board** (later reads return 0xffffffff). After a failed init: STOP, report, ASK — never retry-loop, never touch the device again.
- **Aborting a TT run does NOT stop the board**: killing the host strands spinning cores (heat) and wedges it. After ANY aborted/hung TT run, make NO further device access (not even compile-only runs — they init the device too).
- **ALL invalid IR must be REJECTED at compile time** (CB overflow, OOB index, negative loop length, ...) — codegen must never emit kernels that wedge the board at runtime.
- **Launch policy (2026-09-15):** compile-time traffic-correspondence checks (sections, CB count/pages, push/pop totals, unpacker inputs, DRAM untouched from compute) reject invalid IR loudly; several have fired correctly. A passing compile is a sufficient gate to launch — no per-launch permission. `ZYX_TT_DUMP_ONLY=1` proves little (no C++ JIT); eye-reviewing `ZYX_DEBUG=16` dumps is good practice for brand-new kernel shapes. Custom-kernel tests should finish in seconds; >60s = hung, abort and check.

## gws (Global Work Size)

`gws` is backend-specific only (`GwsDim` in `src/backend/mod.rs`), not a launch arg or kernel concept. Dynamic (`Param`-backed) group lengths MUST work. Each gws dim is a `Group` index length: `Op::Const` → `GwsDim::Const(size)`; `Op::Param { Variable }` (dtype `IDX_T`) → `GwsDim::Param(ordinal)`; anything else is unreachable. Backends walk their own `Op::Range` ops — no kernel helpers. Compile stores one `GwsDim` per axis; launch derives the grid from it + args (`Param` reads `args[ordinal]` via `get_variable` → `Constant::as_dim()`). **Arg-order guarantee:** `args` at launch are in the same flat, head order as `Param` defines in the compiled IR (all kinds; `Op::Storage` is not a parameter; a `Param`'s ordinal counts every `Op::Param` from head).

## Optimization Passes

All passes live in `zyx/src/kernel/` (`autotune.rs` driver + one file per pass). If a sequence breaks correctness, the pass that produced invalid IR is BUGGY — fix or disable it. Run full `cargo test` after touching any pass. Kernels are NOT guaranteed to work with zero optimizations: backend-specific epilogue passes are required before codegen can consume the IR on complex hardware (intended — keeps things in the IR, simplifies codegen).

- **Autotuner** (`autotune.rs` `BeamSearch`): runs per graph kernel, egraph-agnostic (world = linear SSA). NOT used for custom kernels. Inputs: seed iterator (linearized, `default_epilogue`-applied), `default_optimizations` make fns, epilogue closure, cost closure. Generates thousands of variants via `MakeOpt` (`nconfigs()`/`apply(&mut Kernel, config)`); ranked by `cost`, only top `n_launches` (default 1) compiled+launched; measured `nanos` picks the winner for `Graph::extract`.
- **`default_epilogue()`** (`autotune.rs:80`) — always-on fixed order on every seed and after every step: `unroll_len1_loops` → `constant_folding` → `move_constants_to_beginning` → `loop_invariant_code_motion` → `fold_accs` → `delete_zero_len_indices` → `delete_zero_len_loops` → `unfold_pows` → `algebraic_simplifications` → `simplify_accumulating_loop` → `swap_commutative` → `common_subexpression_elimination` → `instruction_schedule` → `dead_code_elimination` (**must stay last**; backends fail on unused ops), then `exp_to_exp2`/`ln_to_log2` (or inverse) and `opt_tenstorrent_tile` if tenstorrent. Result has flat loops; splitting/coarsening only via autotune.
- **`default_optimizations()`** (`autotune.rs:61`) — 8 make fns (9th `opt_vectorize` disabled): `opt_split_global_to_local`, `opt_reassociate_commutative`, `opt_coarsen`, `opt_register_blocking`, `opt_local_reduce`, `opt_split_loop`, `opt_merge_nested_loops`, `opt_fuse_mad`.
- **Adding one:** define `Optimization` + a `MakeOpt` fn scanning at a stable state (may embed `OpId`s; nothing runs between make and apply); add it to `default_optimizations()`. Must preserve semantics; if it can produce invalid IR, gate it or `todo!()` — never alias to a wrong op.
- **Gotchas:** `apply_seq` = `opt.apply` → `epilogue`, so variants are epilogue-cleaned. After unfolding, loop indices flow through `Mad`→Cast→Binary chains — `check_loop` must trace to the loop var. Pattern matchers must handle interleaved op ordering. `alloc_buffers` walks `Param` defines in head order, splits read-only vs `GlobalMut`; in `BeamSearch::run_` buffers are allocated once per launch set — never inside the variant loop. `get_hash` (AHasher) avoids re-launching duplicates; `verify()` runs on every seed.

## Debugging

1. Run the failing test in isolation: `cd zyx && AGENT=1 cargo test --test <file> <name>`.
2. Read the backtrace and the named code; follow callers outward until the violation is found. Add `eprintln` markers where reading doesn't localize (allowed WITHOUT asking; remove before finishing). `kernel.debug()` prints IR at any pipeline point.
3. Ask the user only what the code cannot settle (intended semantics, fix direction) — in plain text, before touching files.
- **Architecture-first debugging:** for non-trivial bugs (lifecycle, ref-counting, tape/eager boundary, graph state) the user provides architecture + invariants up front — write them as a summary block, verify each claim against source, treat the user's design as ground truth. The block is a candidate for permanent placement in AGENTS.md.
- **DO NOT silently iterate edit → test → edit → test while a feature is broken.** Each failure after your change is a NEW task: stop, report exactly what panicked and where, ask before touching files again. One fix per question. Debug instrumentation is allowed during this loop (remove it before finishing).
- In `cargo test`, output only shows with `-- --nocapture`. When reading debug output do NOT pipe through `rg`/`grep`/`head` — run with `-- --nocapture` and view the full output.
- **GPU launch failures** (CUDA_ERROR_ILLEGAL_ADDRESS etc.): capture IR (`ZYX_DEBUG=8`) + code (`ZYX_DEBUG=16`); compare kernel signature arg count/order vs buffers passed. CUDA signature = `MemScope::Global`/`Variable` defines in head order (`codegen/cuda.rs`); `alloc_buffers` counts the same; `device.launch` maps each arg to a param (`backend/cuda.rs` ~713 — add an `eprintln!` printing `args.len()` vs define count to confirm). Check whether eager still works (`tests/3_movement.rs`) to isolate graph-specific bugs. `plan.debug()` (`graph/plan.rs`) shows pool/class bindings.
- **Broken optimization pass:** identify bad kernel (`ZYX_DEBUG=16`, O(N²) loops), capture IR (`ZYX_DEBUG=8` → `/tmp/ir.txt`), write a unit test reconstructing that IR exactly, assert the pass does NOT optimize it (reproduced), fix the pattern matching, assert it now does, run all tests.
- **Crashes (segfault):** minimal reproducer + `panic!("A")`, `panic!("B")`, ... markers along the path (panics flush; `eprintln!` may not before SIGSEGV).
- **Hangs:** `println!` markers + `-- --nocapture` + `timeout N` — the LAST marker localizes it.
- **Lock watchdog:** `src/mutex.rs` has a commented-out watchdog (panics ~lines 88/107 when `RT.lock()` is held too long). It is NOT a reentrancy deadlock detector — a panic means some path holds the lock excessively (e.g. slow autotune). Uncomment it when a hang appears to name the holder.

### ZYX_DEBUG (bitmask, `ENV_VARS.md`; combine by summing)

| Value | Output |
|-------|--------|
| 1 | devices + kernel launches |
| 2 | egraph print (after realize) |
| 4 | scheduler kernels — IR BEFORE linearization (DAG of buffer ops, consts, expands; no loops) |
| 8 | IR AFTER linearization + optimization (flat loops, loads/stores) — printed during compile, before any GPU run |
| 16 | generated assembly/code |
| 32 | launch + memory movement |
| 64 | alloc/dealloc |
| 128 | kernel compilation |
| 256 | autotune exploration |

Capture without a GPU hang: `timeout 10 bash -c 'ZYX_DEBUG=8 cargo run 2>&1' > /tmp/ir.txt`.

## Tape Design (`src/tape.rs`)

Training-loop API around the graph.

- `Tape::realize` takes **persistent state only** (params + optimizer internals); intermediates don't need it.
- **Promotion to graph happens in only TWO cases:** (1) registered via `Tape::new`/`Tape::add`, or (2) a binary op with one graph-tensor operand. No temporal "tape scope" — routing is per-op by operands; all-eager operands run eagerly even mid-forward (optimizer momentum buffers carry via case 2).
- Training step: `Tape::new(&net)` → forward/loss → `tape.gradient(&loss, &net)` → `optim.update(&mut net, grads)` → `tape.realize(net.iter().chain(optim.iter()))`. Fixed control flow: `tape.freeze(&outputs)` once, then `frozen.replay(&inputs)` per step.

### Eager vs graph paths

Every op has two paths, chosen per op by operands. **Eager:** `Runtime::{pad_zeros, unary, binary, ...}` pushes the `Op` into the tensor's current kernel (`eager_ids`, e.g. `runtime.rs:1221`) using the custom-kernel API; the kernel launches automatically once no further fusion is possible — the launch IS the realization. **Graph:** the op pushes a `Node` into the egraph (`push_node`, e.g. `graph/mod.rs:1238`); kernels appear only at `compile_graph` via the kernelizer. Changing an `Op` means changing it in **both** places unless they share a construction point. Inside a `Tape` there is **no launch at all**: the ONLY execution is `Tape::realize(states)`, which **consumes the tape** — no partial realization (`loss.item::<f32>()` works only after `realize`).

### Forcing execution / shape conventions

- Tensors are lazy. Force with `tensor.item::<T>()` (direct, NOT `Result`; scalars only) or `sum([axes]).item::<T>()`; inside a tape via `Tape::realize`. **There is NO `Tensor::realize()`** — only `Tape::realize`.
- `Tensor::shape()` is lazy (symbolic/output shape, no compute) and will NOT reveal a `-1` dim that appears only inside a kernel at execution. Do NOT treat "shape() looks fine" as a blocker — write the reproducer to force execution (`item`/`sum`/`to_vec`). A test that hangs or fails because it documents a bug is valid.
- **NULL `shape` in IR = SCALAR (rank 0)** — a display convention, not an error. At `ZYX_DEBUG=4` non-scalars have real shapes; at `ZYX_DEBUG=8` shape is NULL for everything (lowered into loop bounds/index arithmetic). Post-linearize, buffer dims live in loop bounds/group-index lengths, not `shape`.
- **A negative dimension (`-1`, `r4294967295`, ~4.29×10⁹) on a `Param` shape or as a loop/group length is a BUG** — it makes a ~4.29×10⁹-element loop and hangs. `kernel::verify` must catch it loudly, not let it hang.
- Resolve an `OpId` to a concrete dim via `self.resolve_const(op).and_then(Constant::as_dim)` — there is NO `Kernel::resolve_dim` (that name is the private `resolve_dim_op` in `runtime.rs`). `Loop.len` and `RangeKind::Group(len)` are `OpId`s; resolve and assert `>= 0` if constant.

## Interaction Rules

- **ALWAYS ASK BEFORE ADDING ANY FUNCTION OR CHANGING A SIGNATURE** — including private/`pub(crate)` helpers and backend free functions. Signature changes (params, return type, generics, `Result`) are design changes too. Don't "just refactor" to thread a capability through; if an edit needs a new helper, ask whether/where to add it or compute differently.
- **Every prompt ends with an implicit "Any questions?"** — before any edit, launch, or state-changing command, surface genuine questions first and halt until answered. Reads (read/grep/glob, listings) are allowed to ground the questions. Only proceed when no question remains.
- **RT lock discipline — a live guard must never overlap a `Tensor` drop.** `Tensor::drop` takes the lock via `RT.try_lock()`, and `try_lock` SPINS until free (it is not non-blocking). So `cur = Tensor { id: RT.lock().f(cur.id)? }` self-deadlocks: the guard temporary outlives the statement and the assignment drops old `cur` while it is held. Always bind ids in their own statement first (`let nid = RT.lock().f(cur.id)?; cur = Tensor { id: nid };`), so the guard is dead before any drop.
- If a message asks anything (contains `?` or is interrogative in plain words — "are you capable", "do you see", "isn't it", "what do you think", "why"), **answer and stop** — no file edits, no writes, no launches, no state-changing commands of any kind. Reads (read/grep/glob, listings) are allowed only to ground the answer. The `?` character is sufficient but not necessary: interrogative intent alone triggers this rule.
- **Before accessing an external resource** (repo, docs, download, clone), ASK how to get it — never assume a source or fetch on your own.
- **Follow the literal ask exactly — quantity included.** "A test" = ONE test. Extras (more tests, renames, refactors, extra fixes) are unrequested work.
- **Never act on ambiguous scope — ask first.** If an instruction admits two readings (which files, how much to revert, what "it" or "that" refers to), state the readings and ask; do not pick one and act. Acting on a guess is a violation even if the guess was reasonable.
- **Never implement beyond the literal ask.** If the change requires fixing other code (e.g. a test exposes a library bug), STOP and ask before writing any fix.
- **NEVER chain fixes without asking between steps.** `fix A → fails → fix B → ...` is the forbidden pattern; one fix per question. Test failures are test failures — you don't fix them unless told.
- A test's pass/fail is not the deliverable unless the user says so; a test that currently fails because it documents a bug is valid.
- **Never leave temporary debug code behind**; when reverting, revert completely.
- **Never commit unless explicitly asked** (`git add` + concise commit, zero extra commentary). Never `git stash` or `git checkout --` — use `git restore`. Never discard or hide changes.
- No silent fixes, no post-hoc commentary ("Done.", "X lines changed."), no verbatim git diffs.
- Edit precisely, don't cascade: change only what was referenced; propose (don't do) anything else.

## Defensive Programming: Good vs Evil

GOOD — fail loudly at the exact spot the invariant breaks (`graph/autograd.rs` expand backward): `expect("expand backward with symbolic dim")` turned a silent wrong-gradient bug into a pinpointed panic, making the fix obvious.

EVIL — launder failure into a fake value (`kernel/mod.rs` `Kernel::shape`): `Op::Const(c) => c.as_dim().unwrap_or(0)` + `_ => -1` fabricated dims that flowed into `dot()`'s reshapes and exploded three kernels away; tracing cost a whole session.

Rules:
- Never `unwrap_or(<number>)` on resolution; use `expect("context")`/`todo!()`.
- No sentinels/fallbacks for unknown behaviour; fail loudly at the spot or ask. No `-1`-means-symbolic, no default arms.
- Don't pattern-match only the trivial case (`Op::Const`) when full resolution exists (`resolve_const`); "can't resolve" must mean genuinely unresolvable.
- NEVER use `_ =>` catch-alls on enums — every variant gets an EXPLICIT arm (exception: the final `unreachable!()`/`todo!()` in an eval match after all real variants are named).
