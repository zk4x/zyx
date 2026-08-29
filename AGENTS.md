# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
# ⚠️  "pre-existing" IS FORBIDDEN. NEVER USE IT.  ⚠️
# ⚠️  "Builds clean" / similar                    ⚠️
# ⚠️  IS FORBIDDEN. BUILD SILENTLY.               ⚠️
# ⚠️  "git checkout" IS FORBIDDEN. NEVER USE.     ⚠️
# ⚠️  Use "git restore" instead.                   ⚠️
# ⚠️  python/sed/awk/heredoc SCRIPTS FOR FILE      ⚠️
# ⚠️  EDITS ARE FORBIDDEN. USE THE edit TOOL.      ⚠️
# ⚠️  VIOLATIONS = BROKEN. REPEATEDLY. FOREVER.   ⚠️
# ⚠️  ASK OR EDIT. DO NOT THINK.                   ⚠️
# ⚠️  (asking questions, editing, or both; never
# ⚠️   burn context on internal deliberation)      ⚠️
# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

# Agent Guidelines for zyx

## Know Your Limits

- **ALWAYS run `rust_analyzer_rust_analyzer_set_workspace` (workspace: `/home/x/Dev/rust/zyx/zyx`) as the FIRST tool call of every session**, before any other work.
- **ALWAYS ASK QUESTIONS FIRST. NO MATTER WHAT.** This is unconditional — not just when you perceive ambiguity. Before writing ANY code for a task, ask whatever needs clarifying in your reply as plain text (never the `question` tool). Do not skip asking because the task "seems clear" or you think you know the answer. Not asking is the failure. If you are about to start an edit and have not asked questions, stop and ask first.
- **When the user gives you guidance/a hint, STOP and ask what they want you to do with it before investigating or editing.** Do not take a hint as a license to autonomously read code, form a plan, and edit. Ask: "do you want me to apply this to fix X, or is there a specific approach you have in mind?" If you are reading code to form a fix, that is a signal you should be asking questions instead of planning edits.
- If a task is hardware-specific, deeply subtle, or keeps failing, SAY SO plainly up front. "I can't solve this" beats two days of guesses.
- When the user says they'll do it themselves, stop. Only act when asked for a specific edit.
- **ASK QUESTIONS.** Ask about EVERYTHING — not only design decisions, but also implementation details: HOW to implement, which approach, the exact shape/semantics/values of a change, what a method should do, naming, structure. Whenever you are about to guess or invent an answer — a design choice OR an implementation detail — ask BEFORE implementing. Guessing wrong and writing broken code wastes more time than asking.
- **Asking is ITERATIVE, not one-shot.** Getting answers once does not mean you can now go implement silently. Every newly raised detail gets asked, again, in plain text — including mid-implementation. If between two edits a question appears ("which callers change?", "what should X return here?"), STOP and ask it. Never resolve such questions by reading code: opening files/grepping to figure out how something works or what to change IS the failure, even right after the user approved a plan. The user answers; you do not research. Never let a time budget make you skip asking.
- **ASK EARLY AND OFTEN. You are NOT asking enough.** Default to asking: the first reply to a non-trivial task should usually contain questions, not edits. Do not begin implementing until the design/spec is clear. If the task references a `todo!()` or a stub, ask what the intended behavior/value/semantics are before guessing at them.
- **The user is NEVER tired of you asking.** Questions are never a nuisance, never a sign of weakness, never "too many". Do not ration questions or apologize for asking. If you are hesitating to ask because you fear it will annoy the user, that is exactly when you must ask. Erring on the side of one more question is always correct.
- **NEVER use the `question` tool. Ask questions directly in your reply, as plain text.** If you reach for the `question` tool, that is your signal you are about to guess — stop and write the question out instead.
- **Do not implement on top of your own guesses.** Researching the code to inform a question is good; implementing code to "find out" what should happen is not. Ask first, implement after the answer.
- **Ask the user, don't investigate the answer yourself.** When you catch yourself forming a question in your reasoning and then going to read code to answer it, STOP and ask the user instead — they know the answer. Self-investigation to resolve a design or implementation question is wasted time and an implicit refusal to ask. Your internal "questions" are questions to the user, not research prompts.
- **Reconsidering = ask.** Whenever the word "reconsider"/"reconsidering"/"on reflection" appears in your thinking about a design, spec, or your own plan, immediately ask the user what to do — do not keep reasoning through it on your own. Every instance of reconsidering is a question you are deciding not to ask.
- **Confirm whether values are symbolic or numeric before touching them.** Dims, shapes, lengths, and strides are usually `OpId`s (symbolic IR nodes), not `Dim`s you can multiply directly. If a computation would need numeric values (e.g. `index_len`/`as_dim` to multiply), that is a sign the design may intend symbolic ops instead — ASK which is intended, and how strides/shapes are meant to be expressed, before writing code.
- **FORBIDDEN WORDS** — never say these (in responses AND in your reasoning AND in your inner monologue): "Actually", "wait", "key insight", "let me", "Hmm", "But wait". The rule is absolute, applies to every token you emit (including tool call parameters, file contents, and planning), and violations are never excused by "it was in reasoning" or "I was thinking out loud".

## No Workarounds

- **NO WORKAROUNDS.** Never paper over a real bug by rerouting the caller away from the buggy code path (e.g. swapping `index_select` for `slice`, or dropping an API the user asked for) instead of reproducing and fixing the bug. A workaround hides the bug and shifts the symptom somewhere else, making it far harder to find later. When you find a bug, **reproduce it in a test** (e.g. in `zyx/tests/`) and report it — do not route around it. If avoiding the path seems necessary, STOP and ask the user first.

## Optimization Pass Performance Budget

- **NO optimization pass may take longer than 30 microseconds (µs) on a single invocation.** Most passes run well under that (single-digit µs); 30µs is a generous ceiling, not a target. This applies to every pass in `zyx/src/kernel/` (algebraic, verify/compute_bounds, fold_*, autotune drivers, etc.) and to `run_always_on_optimizations` / `AVAILABLE_OPTIMIZATIONS` as a whole.
- A pass that blows this budget is a **performance bug**, not an acceptable cost: it runs once per autotune variant and once per compiled kernel, so even a few hundred µs compounds into seconds across a run. Such a pass is almost always doing redundant or quadratic work (e.g. re-walking the whole IR per `If`, recomputing bounds many times). Fix the algorithmic cost — do NOT paper over it.
- **Add timings to measure.** Wrap each pass in a `time_pass!`/`Instant` measurement (printing to stderr) so per-pass cost is observable, then keep it under 30µs. Verify with a targeted test, not the full suite, when measuring.
- `compute_bounds` (verify.rs) is the usual suspect: it walks the whole IR and is called repeatedly. It must stay linear in the number of ops; never re-derive or re-scan the IR per-`If` (that is O(K·N)). Remember `compute_bounds` only guarantees **conservative (never too tight)** bounds — see its docs.

## Key Commands

The repo root has **no `Cargo.toml`** — crates (`zyx`, `zyx-nn`, ...) are independent packages. **Always run cargo from the `zyx/` subdirectory.**

```bash
cd zyx && cargo build                      # build
cd zyx && AGENT=1 cargo test               # test (AGENT=1 strips ANSI colors)
cd zyx && AGENT=1 cargo test <name>        # single test
cd zyx && AGENT=1 cargo test --test <file> <name>  # one test in one file
cd zyx && AGENT=1 cargo test -- --nocapture   # with output (ALWAYS add -- --nocapture when capturing debug/env-var output, e.g. ZYX_DEBUG)
cd zyx && AGENT=1 cargo test --doc         # doc tests
cd zyx && cargo fmt                        # format
```

- Tenstorrent: `TT_METAL_ROOT=/home/x/Dev/cpp/tt-metal cargo build --features tenstorrent`
- Clippy is NOT a gate (lint `#![deny(...)]` block in `src/lib.rs` is commented out). Don't run it.
- Tests live in `zyx/tests/`, named `{number}_{category}.rs` (e.g. `1_unary.rs`) plus `mnist.rs` (`mnist.py` + `.safetensors` are fixtures). Tests return `Result<(), ZyxError>` and use `assert!` / `is_equal()` for floats.
- Python: use `python3.12` for everything.

## Layout

```
zyx/           core tensor library (all the real work)
  src/backend/   one file per backend (c, cuda, hip, opencl, vulkan, disk, host, dummy, tenstorrent, wgpu)
  src/graph/     the egraph + autograd + kernelizer + plan
  src/kernel/    kernel IR, codegen, and ALL optimization passes
  tests/         integration tests
zyx-nn, zyx-optim, zyx-onnx, zyx-fuzzy, zyx-py, zyx-bench, zyx-book
examples/      virtual workspace; each model is its own crate (mnist, rnn, phi, llama, ...)
                examples/data/ holds datasets, examples/models/ holds weights/gguf/configs
```

Root docs worth reading: `ENV_VARS.md` (ZYX_DEBUG), `CONFIG.md` (backend config), `STYLE.md`, `ADDING_BACKENDS.md`.

## The Graph

Zyx is all about the graph. Lazy (no compute until `Tensor::realize`), dynamic, and **one graph serves both laziness and autograd**. TensorId is a `u32`.

`Node` (`src/graph/mod.rs`) has 14 variants: Const, Leaf, Expand, Permute, Reshape, PadZeros, Flip, Reduce, Cast, Unary, Binary, Assign, ToDevice, Kernel.

## The Egraph

Lives in `zyx/src/graph/`:
- `mod.rs` — the equivalence graph. Heuristics generate a **few fusion variants** of each subgraph (CSE via hashconsing, algebraic rewrites, layout rewrites, shape rewrites). The egraph itself has **NO cost model** — it does not rank or pick a form statically. Each fusion variant is individually **autotuned** (see Optimization Passes); the egraph then uses the measured **timings** returned by the autotuner to extract the fastest path (`Graph::extract`).
- **Vendor kernels**: a vendor can contribute a kernel for a subgraph written either in any backend language (CUDA, SPIR-V, ...) **or** in the zyx kernel IR API. Kernels built via the kernel IR API go through all the usual optimizations, including autotuning. Vendor kernels **compete alongside** the heuristics-based fusion variants in the egraph — they are extracted by the same timing-driven path, so the fastest measured variant (heuristic or vendor) wins.
- `kernelizer.rs` — pattern-matches subgraphs and replaces them with custom JIT kernels
- `autograd.rs` — gradients on the same graph
- `plan.rs` — execution plans from compiled graphs

How to work with it:
- `ZYX_DEBUG=2` prints the egraph (class structure) after realize — the starting point for graph debugging.
- To replace a graph pattern with a custom kernel, add the match in `kernelizer.rs`.
- Rewrites must preserve semantic equivalence. The egraph does **NOT** pick a form by cost — it relies on the autotuner's measured timings to extract the fastest path. A wrong rewrite therefore surfaces as a WRONG (not slow) kernel, so every rewrite must be semantically exact.

## Symbolic Dims

- **Eager and graph must BOTH work with symbolic dims — always and everywhere.** No path may assume concrete shapes. A dim that is not statically known is symbolic, and EVERY consumer (load kernels, autograd, optimization passes, backends) must handle it correctly.
- Every symbolic dim bottoms out in `Param { Variable }` scalars (`IDX_T`) whose actual values live in the backend pools' variable slots. `IDX_T` is `DType::I64` (i64) — **never `u32`**. Dimension/index values must NEVER be cast to `u32`: they overflow above 4.29×10⁹ (e.g. `60000·120000` wraps to `2905032704`), which silently corrupts downstream codegen. `TensorId`/`OpId`/`ClassId` are `u32` only as compact identifiers, never as dimension values. Because of that, ANY dim expression can be evaluated to a concrete `Constant` at any time: walk the expression tree, take `Const` leaves as-is, fold `Unary`/`Binary` via `Constant::unary` / `Constant::binary`, and read the variable slot wherever a leaf is a `Param { Variable }`.
- NEVER fabricate sentinel or fallback values (`-1`, `0`, `42`, ...) for unknown behaviour anywhere — not in resolution failures, not in match arms, not as defaults. If code cannot determine a value, that is either a bug or a missing design decision: fail loudly (`debug_assert!` / `expect` / `todo!()`) at the exact spot, or STOP and ask the user what the value should be. No `unwrap_or(<number>)`, no default-value arms, no silent substitutes.
- Load kernels (e.g. the fresh loader in `Runtime::add_store`) must NEVER emit fabricated consts (`Const(-1)` etc.) as group lengths: evaluate the dim to its concrete value first.

## Code Style

- **Simplicity/debuggability > "clean"**: duplicate code first, abstract later. Use enums, not `dyn Trait`. Explicit returns. Minimize Arc/Rc.
- **Better `todo!()` than silent failures**: an explicit unimplemented panic beats a wrong result or garbage output. Leave `todo!()` in for unimplemented paths rather than faking them.
- **No assumptions without a `debug_assert`**: never rely on an invariant (e.g. "this kernel is store-free", "this value is non-negative", "this tensor is contiguous") unless the code asserts it with a `debug_assert!`. Every unchecked assumption is a latent bug. **If there are no asserts, there are no assumptions** — claiming an invariant without an `assert!`/`debug_assert!` backing it in the code is meaningless. Do not state "it's safe because X is assumed" unless that X is actually asserted; either add the assert or stop claiming the assumption. This applies to user-guaranteed properties too: if the user says "you can assume X" (e.g. "reduce axes are unique"), the code must still carry a `debug_assert!` for it.
- **NEVER implement a bug knowingly. THE SINGLE WORST SIN IS adding a variant/arm that produces a semantically WRONG result on purpose** — e.g. aliasing `BOp::Cmpge` to emit `>` (that is `Cmpge == Cmpgt`). This has been the most serious failure on this repo. Every newly-added `BOp`/`UOp` variant MUST get a correct implementation in EVERY exhaustive match (kernel IR, all codegen backends, constant-folding, autodiff, verify/range analysis, debug/string maps). If you cannot implement a match correctly right now, write `todo!()` in that arm — NEVER a wrong result, NEVER an alias to a different op, NEVER a silent no-op that changes semantics. `todo!()` panics loudly; a wrong result is a silent, insidious bug that may go unnoticed forever. If you are ever tempted to "make it compile" with an incorrect arm, STOP and ask the user instead — it is always preferable to ask.
- **Adding a new op/variant is not complete until `cargo build` passes AND every exhaustive match over that type has been audited for semantic correctness — not just compilation.** Compiling is the floor, not the bar. A wrong semantic (e.g. `>=` lowered to `>`) compiles fine and is still a bug. After adding a variant, grep for every `match` over that type and check each arm's emitted semantics.
- **When you fail to find something, ASK.** If you go searching for a method/function/symbol you assumed exists and it isn't where you expected (e.g. hunting through files for a `cse` pass you never confirmed exists), STOP and ask the user what it's called or whether it exists. Never resolve a "I can't find X" into more grepping — that is self-investigation to answer your own question, which the rules forbid. Ask the user directly. This applies to any lookup that does not immediately succeed: first grep missing or empty → ask, don't retry variants. **And even a SUCCESSFUL lookup is a failure if it answered a question you should have asked the user.** If you find yourself grepping to pick the right API/method/pass name to satisfy the task (e.g. which CSE/DCE method to call), that means you were about to make a design choice yourself — STOP, ask the user which one they want, and do not go look it up until they tell you. The decision of WHICH method/API to use belongs to the user, not to your grep.
- Keep ~1000 LOC per module; add new files only when necessary.
- License header on every new file: `// Copyright (C) 2025 zk4x` / `// SPDX-License-Identifier: LGPL-3.0-only`
- Public items need docs (`#![warn(missing_docs)]` is active). Error constructors use `#[track_caller]` and embed file:line:column. Use `From` for conversions.
- Import order: `crate::` → `super::` → external crates.

## Backends

- Most backends are **always compiled in**, selected at runtime via the zyx config file.
- **NEVER touch `~/.config/zyx/config.json`** (never read/write/create/modify/delete it). Backends are chosen by the user. If a test needs a specific backend, ask the user to configure it.
- Only two backends need cargo features: `--features wgpu` and `--features tenstorrent`.
- Defaults with no config: C on, Dummy off, CUDA/HIP/Vulkan/OpenCL try to init and silently skip if the driver is missing, HIP always tries (ignores config; skipped only if `libamdhip64.so` is missing). If all backends fail, tests print nothing.

## gws (Global Work Size)

`gws` is **not** a launch argument and is **not** a kernel concept. It is purely
backend-specific (`GwsDim` lives in `src/backend/mod.rs`). Dynamic (`Param`-backed)
group lengths MUST work — this is not a future nicety.

- Each gws dimension is a `Group` index length. Its length `op_id` is either
  `Op::Param { kind: Variable }` (dtype `IDX_T`) or `Op::Const`; **anything else is
  unreachable**.
- Backends walk their own `Op::Index` ops themselves — **no kernel helpers**. For each
  `IdxKind::Group(op_id)`:
  - `Op::Const(c)` → `GwsDim::Const(size)`
  - `Op::Param { Variable }` → `GwsDim::Param(ordinal)`
- At compile each backend stores one `GwsDim` per gws axis in its program struct.
- At launch each backend derives the actual grid from the stored `GwsDim` + `args`:
  `Const(size)` uses the size directly; `Param(ordinal)` reads `args[ordinal]` from the
  pool (a scalar `Variable`, via `get_variable` → `Constant::as_dim()`).
- **Arg-ordering guarantee:** `args` passed to launch are in the SAME order as `Param`
  defines in the kernel IR given to compile — flat, head order, all kinds
  (`Variable`/`Global`/`GlobalMut`). `Op::Storage` is NOT a kernel parameter. A `Param`'s
  ordinal is its position counting every `Op::Param` from head.

## Optimization Passes

All kernel optimizations live in `zyx/src/kernel/` (`autotune.rs` driver + one file per pass: `split_loops.rs`, `coarsen.rs`, `vectorize.rs`, `fold_loops.rs`, `algebraic.rs`, ...).

**Correctness is critical**: no optimization is required for tests to pass; ALL tests must pass with ANY optimization sequence (including empty). If a sequence breaks correctness, the pass that produced invalid IR from valid code is BUGGY — fix or disable it. Run the full `cargo test` after touching any pass; a single failing integration test means the pass produces wrong results.

### The autotuner

The autotuner runs on each kernel (every egraph fusion variant). For a given kernel it generates **thousands of variants** by running the different optimizations and the different **configurations** of those optimizations (each `config()` returns a `#variants` count — see below). Variant selection uses a **combination of a cost model and measured timings**: the cost model is **regression / neural-net based** (learned, not a hand-written analytic estimate), and the **user configures the ratio** between how many variants are scored by the cost model versus actually launched and timed. Regardless of that ratio, the autotuner **always runs (times) each variant at least once** — there is no path that picks a kernel purely on the cost model without a real launch. The measured timing is what the egraph consumes to extract the fastest path.

### Two pipelines

- `run_always_on_optimizations` (`autotune.rs:336`) — always runs before compilation, fixed order, **`dead_code_elimination` must stay last** (backends fail on unused ops, e.g. missing ref-count entries).
- `AVAILABLE_OPTIMIZATIONS` (`autotune.rs:59`) — 9 passes the autotuner tries in combination to generate the thousands of per-kernel variants (split_global_to_local, reassociate_commutative, coarsen, register_blocking, local_reduce, split_loop, pad_index, vectorize, merge_nested_loops).

### Adding an optimization

1. Config fn returns `(Optimization, usize)` (optimization + #variants to try; `1` if no tunable params): `pub fn my_opt_config(&self, _dev: &DeviceInfo) -> (Optimization, usize)`.
2. Apply fn on `Kernel`: `pub fn my_optimization(&mut self, config: usize)`.
3. Register in `AVAILABLE_OPTIMIZATIONS`.

### Key gotchas

- Pipeline order matters: loop-splitting/coarsening run only during autotune, AFTER `run_always_on_optimizations`. Always-on IR has flat loops.
- After unfolding, loop indices flow through `Mad`→Cast→Binary chains — `check_loop` must trace the chain to the loop var.
- Pattern matchers must handle interleaved op ordering (accumulate computation can sit between `load(acc)` and `Add`).
- **Buffer allocation in autotune**: allocate buffers ONCE and reuse across all variant launches (`compile_and_launch`), then deallocate. Never alloc inside the variant loop.
- Hashing kernels avoids re-launching duplicates.

## Debugging

### First steps

1. Run the failing test in isolation, pointing at its test file so the other tests aren't recompiled:
   `cd zyx && AGENT=1 cargo test --test <file> <name>`
2. Use `ZYX_DEBUG` (below) to see the graph, kernel IR, and generated code instead of guessing.

### Architecture-first debugging

For non-trivial bugs (lifecycle, ref-counting, ownership across tape/eager boundary, graph state), the user provides the architecture and invariants up front. The agent writes them into the conversation as a short summary (a doc-like block), and that crystallizes the issue — invariants the code violates, transitions that aren't handled, states the code doesn't model. The agent does not grep to discover them; the user knows.

When the user says "the architecture is X, the invariant is Y, and it breaks because Z", treat that as ground truth. Reproduce, then ask the next question. The summary block is also a candidate for permanent placement in AGENTS.md under the relevant topic (Tape Design, Backends, ...).

### Investigate the MINIMUM, then ASK — debug TOGETHER

We debug **together**. You do NOT read code to find bugs; you ask the user and they answer. The fastest path to a fix is the user pasting the IR/debug output and walking through it with you — not you grepping files. Debugging this way finds and fixes a bug in minutes, not the hour you spent reading/instrumenting on your own.

The user knows the answer to your design questions. Over-investigation is a hard failure. Follow this exact order:

1. Reproduce with a test run (produces the backtrace).
2. Read the **backtrace** and form the diagnosis directly from it. If the backtrace names a call chain (X calls Y at known lines), the bug is there — do NOT go digging deeper, and do NOT add `eprintln`.
3. **Ask the user** the question the backtrace raises, in plain text, before touching any file.
4. Iterate: the user answers / pastes debug output; you reason over what they gave you and ask the next question. Do not switch to reading source to "check" — stay in the ask-answer loop.

Automatic "STOP and ask" triggers (each one alone means you must ask, not investigate):
- You want to read a second file to resolve a question → ask.
- You've used more than ~2 investigative tool calls after the backtrace → ask.
- You find yourself forming a question, then going to read code to answer it → that question is for the user; ask them.

EXCEPTION — debug prints: adding temporary `eprintln!`/`println!` debug instrumentation does NOT require asking. Just add it, run, and bring the output. Remember to remove temporary prints before finishing (never leave debug code behind).

**Never run a Read/Grep/Bash tool to investigate a bug until you have asked the user the question you are trying to answer.** Reading code to answer your own question IS the failure.

**DO NOT SILENTLY ITERATE edit → test → edit → test while a feature is broken.** This is the failure from the reshape debugging: the test panicked, I patched, it panicked somewhere new, I patched again, on and on with no question in between — a whole session of unrequested surgery. The rules are:
- Every time the test fails after you made a change, that failure is a **NEW task**. Stop. Report exactly what panicked and where. Ask the user how to proceed before touching any file again.
- One fix per question. You may NOT run the test again and patch whatever breaks next without a question in between.
- If you are about to run the test a second time in a row to see "what happens next", you are chaining fixes — stop and ask.
- Adding any `eprintln`/`println` debug instrumentation during this loop is allowed WITHOUT asking (see exception above); just remember to remove it before finishing.

### ZYX_DEBUG (bitmask, `ENV_VARS.md`)

| Value | Output |
|-------|--------|
| 1 | hardware devices + kernel launches |
| 2 | egraph print (after realize) |
| 4 | kernels created by scheduler — IR BEFORE linearization (tensor graph, no loops/indices/loads/stores) |
| 8 | kernel IR AFTER linearization + optimization (loops, indices, loads, stores) |
| 16 | generated assembly/code |
| 32 | kernel launch + memory movement |
| 64 | memory alloc/dealloc |
| 128 | kernel compilation |
| 256 | autotune exploration |

Combine by summing (`24` = ir + asm). The two IR views show different stages: `ZYX_DEBUG=4` prints the kernel as it comes out of the scheduler (a DAG of buffer ops, scalar consts, expands — no loops yet), `ZYX_DEBUG=8` prints the same kernel after linearization and the optimization passes (flat loops over element indices, per-op loads/stores). `ZYX_DEBUG=8` prints IR during compilation, BEFORE any GPU kernel runs — use `timeout 10 bash -c 'ZYX_DEBUG=8 cargo run 2>&1' > /tmp/ir.txt` to capture it without a GPU hang.

In `cargo test`, library `eprintln!` output only shows with `-- --nocapture` (`AGENT=1 cargo test -- --nocapture`), otherwise it's hidden by the test harness.

### Inspecting kernel IR

Add `kernel.debug()` temporarily in code to print the IR (op IDs, args, loop scopes) at any pipeline point. `AGENT=1` strips ANSI colors; `AGENT=0` keeps them.

### Debugging GPU launch failures (CUDA_ERROR_ILLEGAL_ADDRESS etc.)

1. Run the failing test in isolation (see First steps).
2. Capture the failing kernel's IR (`ZYX_DEBUG=8`) and generated code (`ZYX_DEBUG=16`). Compare the kernel signature arg count/order against the buffers passed at launch:
   - CUDA codegen builds the kernel signature from `MemScope::Global` and `MemScope::Variable` defines in head order (`codegen/cuda.rs`).
   - `alloc_buffers` in `kernel/autotune.rs` counts the same defines (stops at first non-Define op) and allocates a fresh buffer per define.
   - `device.launch` in the CUDA backend maps each arg buffer to a kernel param (`backend/cuda.rs`); a `CUDABuffer::Buffer` passes a pointer, `CUDABuffer::Variable` passes the scalar value.
3. A mismatch between the signature arg count and the number of buffers passed causes ILLEGAL_ADDRESS. Add an `eprintln!` in the CUDA `launch` (backend/cuda.rs ~713) printing `args.len()` and the compiled signature's define count to confirm. Print the kernel IR too.
4. To test whether a change broke the graph path but not eager, run the equivalent eager test (`tests/3_movement.rs`) — if it passes, the bug is graph-specific.
5. Print the ExecPlan (`plan.debug()` in `graph/plan.rs`) to see which classes are bound to which pools and how `Launch` binds `load_classes`/`store_classes`.
6. When reading debug output, do NOT pipe through `rg`/`grep`/`head` — `plan.debug()` output may be swallowed by the test harness; run with `-- --nocapture` and view the full output. Debug output must be visible in the actual run.

### Fixing a broken optimization pass

1. Identify the bad kernel: `ZYX_DEBUG=16` → look for O(N²) loops that should have folded.
2. Capture its IR (`ZYX_DEBUG=8` to `/tmp/ir.txt`). The last kernel printed is usually the one compiled before a hang.
3. Write a unit test that reconstructs that IR **exactly** (copy the debug output op by op — `Kernel::new()`, `k.define()`, `k.loop_()`, ...); assert the pass does NOT optimize it, confirming you've reproduced the failure.
4. Fix the pass's pattern matching (small, targeted), assert the test now DOES optimize, then run all tests.

### Crashes (segfault etc.)

Write a minimal reproducer test, then add `panic!("A")`, `panic!("B")`, ... along the suspect path (panics flush; `eprintln!` may not before a SIGSEGV). Narrow down from the last printed marker.

### Hangs

Add `println!` markers along the suspect path and run with `-- --nocapture`: the LAST printed marker localizes the hang. Combine with `timeout N` so the run terminates on its own, then read the tail of the output.

## Tape Design

`src/tape.rs` — the training-loop API around the graph.

- `Tape::realize` takes **persistent state only** (params + optimizer internals). Intermediates don't need realization.
- Eager tensors used inside a tape scope auto-promote to graph (`promote_to_graph`) — optimizer momentum buffers carry across steps without manual `tape.add()`.
- Training step: `Tape::new(&net)` → forward/loss → `tape.gradient(&loss, &net)` → `optim.update(&mut net, grads)` → `tape.realize(net.iter().chain(optim.iter()))`. After drop, realized tensors become eager again.
- Fixed control flow: `tape.freeze(&outputs)` once, then `frozen.replay(&inputs)` per step.

### Two execution paths: eager vs graph

Every tensor op has **two** possible paths, chosen by whether the caller is inside a tape scope:

- **Eager path** (no tape): `Runtime::{pad_zeros, unary, binary, ...}` directly pushes the
  `Op` (a `MoveOp`, or a compute op like `Unary`/`Binary`/`Reduce`) into the tensor's
  current kernel (`eager_ids`, e.g. `runtime.rs:1221` for pad). No graph nodes are
  created. The kernel is built with the custom-kernel API (`Kernel::new`/`define`/
  `pad`/`add`/... in `kernel/custom.rs`).
- **Graph path** (inside `Tape`): the same op pushes a `Node` into the egraph
  (`push_node`, e.g. `Node::PadZeros` at `graph/mod.rs:1238`). Only at `compile_graph`
  does the kernelizer (`kernelizer.rs`) turn those nodes into `Op`s.

This dual path applies to **every op** — unary, binary, reduce, pad, cast, etc. all either push an `Op` into an eager kernel (`runtime.rs`) or push a
`Node` into the egraph (`graph/mod.rs`) depending purely on whether the caller is in a
tape scope. When reasoning about kernel identity / recompilation (e.g. the kv-cache
`narrow(0, start, len)` case), consider which path the consumer actually runs, and
note that changing an `Op` means changing it in **both** places unless the two share a
construction point.

**When execution actually happens on each path:**

- **No tape (eager):** ops are recorded directly into a kernel. The kernel is **launched (executed)** automatically once no further fusion is possible (or for any other reason the runtime flushes it). There is no separate "realize" step — the launch *is* the realization. A bare `Tensor` op outside a tape computes as soon as its kernel is flushed.
- **Inside a `Tape`:** there is **no launch at all** — everything stays lazy in the egraph. The ONLY way to execute is `Tape::realize(states)`, and calling it **consumes the tape**. There is therefore **no partial realization**: you cannot realize some tensors while keeping the tape alive; `realize` runs the whole tape graph and then the tape is gone. (`loss.item::<f32>()` only works *after* `realize` because `realize` already executed the graph.)

### Forcing execution of a lazy `Tensor`

Tensors are lazy: an op builds a graph/eager kernel but does **not** compute until forced. To actually run a value (e.g. in a reproducer test), use:

- `tensor.item::<T>()` — returns `T` **directly** (NOT `Result`, NOT `Option`); valid only on a **scalar** (1-element) tensor.
- `tensor.sum([axes]).item::<T>()` — reduce to a scalar, then `item`.
- Inside a `Tape`: `Tape::realize(states)` realizes the whole graph; `loss.item::<f32>()` after `realize` also triggers it.
- **There is NO `Tensor::realize()` method** — `realize` exists only on `Tape`. Do not call it on a `Tensor`.
- `Tensor::shape()` is lazy (returns the symbolic/output shape WITHOUT computing) — it will NOT reveal a `-1` dimension that only appears INSIDE a kernel at execution time. A bug whose only symptom is an internal `-1` dimension (→ a ~4.29×10⁹-element kernel loop / hang) is invisible to `shape()`; it only shows up under `ZYX_DEBUG=8` (kernel IR) or when the kernel actually executes.
- **Do NOT treat "shape() doesn't show the bug" as a blocker.** If a bug only manifests at execution (e.g. an internal `-1` dimension, a hang, a wrong value), write the reproducer test to **force execution** (`item`/`sum`/`to_vec`) and let it reproduce the hang/failure. A test that currently hangs or fails *because it documents a bug* is a valid outcome (see Interaction Rules). Stop raising "but shape() looks fine" — it's expected and irrelevant.

### Kernel IR `shape` conventions (what `NULL` / `-1` mean — common misconceptions)

- **`NULL` `shape` in the IR means SCALAR (rank 0).** It is a display convention, not "missing shape" and not a bug. Do not read a NULL shape as an error.
- **`ZYX_DEBUG=4` (pre-linearize / scheduler output):** every *non-scalar* tensor's `shape` is a real, non-NULL shape. Only scalars have a NULL shape (even here).
- **`ZYX_DEBUG=8` (post-linearize):** the `shape` field on ops is NULL for **everything** — shapes have been lowered into loop bounds and index arithmetic, so a NULL `shape` here is expected and does NOT mean scalar.
- **Buffer dimensions are NOT stored in the post-linearize `shape` field.** Pre-linearize they live in the `shape` field on the `Param` (input) op. After linearization that dimension becomes a loop bound / group-index length.
- **A negative dimension (`-1`, printed as `r4294967295` / ~4.29×10⁹) on a `Param` shape (pre-linearize) or as a loop bound / group-index length (post-linearize) is a BUG.** It is NOT "scalar" and NOT "infer a dimension" — it produces a ~4.29×10⁹-element loop and hangs. `kernel::verify` must catch it loudly (panic on a resolvable negative loop/group length), not let it hang.
- **To resolve an `OpId` to a concrete `Dim` constant in kernel code, use `self.resolve_const(op).and_then(Constant::as_dim)`.** There is NO `Kernel::resolve_dim` method (the name `resolve_dim` refers to the private `resolve_dim_op` in `runtime.rs`). A `Loop`'s `len` and an `IdxKind::Group(len)`'s `len` are `OpId`s; resolve them this way and, if the result is a constant, assert it is `>= 0`.

## Interaction Rules

- **ALWAYS ASK BEFORE ADDING ANY FUNCTION OR CHANGING A SIGNATURE.** You are NOT allowed to add new functions, methods, or helper functions without the user's approval — this includes private/`pub(crate)` helpers and backend-side free functions. Any new function is a design change: ask first, get approval, then write it. Changing the signature of an existing function or method (adding/removing parameters, changing the return type, adding a generic/trait bound) is ALSO a design change and must be approved first — do NOT "just refactor" to thread a new capability through. Never change a signature on your own.
- Every user message: if it contains a `?`, **answer the question and stop** — do not edit/write files. Otherwise proceed.
- **ASK QUESTIONS as your default first move.** For any task involving a `todo!()` stub, an ambiguous value, or a design decision, your first reply should be questions, not code. Do not implement until you have answers.
- When in doubt, ask. Don't guess specs/values — the user has them. Ask before hunting through source. Ask by writing the question out in your reply as plain text — never via the `question` tool.
- **Before accessing an external resource (reference repo, docs, tool, download, clone, etc.), ASK how the user wants you to get it** — whether there is a local checkout, a preferred path, a URL, or whether to fetch it at all. Do NOT assume a source (e.g. `git clone` from GitHub) or download/fetch on your own. Ask first.
- **Follow the literal ask exactly — quantity included.** "A test" means exactly ONE test; "add a test for X" means just that test, not a family of tests. Deliver what was asked and stop. Extras (more tests, renames, refactors, extra fixes) are unrequested work.
- **Never start implementing anything beyond the literal ask without asking first.** If a requested change turns out to require fixing/modifying other parts of the code (e.g. a test exposes a library bug), STOP and ask the user how to proceed before writing any fix. Do not debug-and-fix your way down a rabbit hole unprompted. A single simple question ("want me to fix that too?") beats an hour of unrequested surgery.
- **NEVER chain fixes across modules/passes without asking between steps.** The forbidden pattern: `test fails → fix pass A → test fails → fix pass B → test fails → ...` with no question in between. You fix the literal ask, THEN the test's next failure is a NEW task — stop, report it, and ask whether to fix it before touching that file. Continuing to run the test and patching whatever breaks next (without a question) is the exact failure that keeps happening. One fix per question.
- **Adding any helper method, function, or new API counts as unrequested work.** If an edit needs a helper that doesn't exist yet (e.g. a `broadcast` shape function), do NOT just write it — stop and ask whether the user wants it added (and where), or whether the result should be computed another way.
- **Adding a function or changing a function's signature is a design change — ASK first.** Do not introduce new functions, methods, or new parameters/return types (including changing a return type to `Result`) without consulting the user. Only the user decides whether and how to add/extend an API.
- A test's pass/fail status is not the deliverable unless the user says so. Adding a test that currently fails (because it documents a bug) is a valid outcome — do not "fix" the code underneath it unprompted.
- **Never leave temporary debug code behind** (eprintln!/println! debug blocks, commented-out scaffolding). If you add debug output while investigating, remove it before finishing. When reverting, revert completely — no stray debug prints, no leftover comments.
- **Never commit unless explicitly asked.** When asked, `git add` + `git commit` a concise message matching repo style and produce zero extra commentary.
- Never use `git stash` or `git checkout --`; use `git restore`. Never discard or hide changes.
- No silent fixes. No commentary after doing something ("Done.", "X lines changed."). Don't dump git diffs verbatim.
- Test failures are test failures. You don't need to fix them unless I tell you to fix them.
- Edit precisely, don't cascade: change only what was referenced; propose (don't do) anything else.

## Defensive Programming: Good vs Evil

GOOD — fail loudly at the exact spot the invariant breaks (`graph/autograd.rs`, expand backward):
```rust
let out_shape: Vec<Dim> = out_dims
    .iter()
    .map(|&d| self.graph_const_dim(graph_id, d).expect("expand backward with symbolic dim"))
    .collect();
```
This `expect` turned a silent wrong-gradient bug into an instant, pinpointed panic. It named
the site, so the fix (symbolic-dim broadcast) was obvious in minutes.

EVIL — launder failure into a fake value and let it explode somewhere else entirely:
```rust
// kernel/mod.rs, Kernel::shape (removed)
Op::Const(c) => c.as_dim().unwrap_or(0),   // fabricated 0 dims
_ => -1,                                    // matched only Const, missed const EXPRESSIONS
```
The fabricated values flowed into `dot()`'s reshape dims, became real tensors, and the
symptom appeared three kernels away from the cause. Tracing it cost a whole session.

Rules:
- Never `unwrap_or(<number>)` on resolution of dims/values. Use `expect("context")` / `todo!()`.
- NEVER use sentinels or fallback values for unknown behaviour. If a value cannot be
  determined, that is either a bug or a missing design decision: fail loudly at the exact
  spot, or ask the user. No `-1`-means-symbolic boundaries, no default arms.
- Don't pattern-match only the trivial case (`Op::Const`) when full resolution exists
  (`resolve_const`); "can't resolve" must mean genuinely unresolvable, not "didn't try".
- NEVER use a `_ =>` catch-all arm when matching an enum. Every variant gets an
  EXPLICIT arm. A `_ =>` silently swallows newly added variants; an exhaustive
  match is a compile error that forces you to decide each case. (Exception: the
  final `unreachable!()`/`todo!()` arm in an eval match after all real variants
  are named.)
