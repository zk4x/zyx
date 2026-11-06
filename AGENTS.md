# Agent Guidelines for zyx
# FORBIDDEN: "pre-existing" - "builds clean" (build silently)
# - "git checkout" (use `git restore`) - python/sed/awk/heredoc
# for file edits (use the edit tool). READ FIRST, THEN ASK OR EDIT.

## Session start
- FIRST tool call: `rust_analyzer_rust_analyzer_set_workspace`
  (workspace: `/home/x/Dev/rust/zyx/zyx`).

## Read first, ask before editing
- READ CODE FIRST. Only STOP and ask when the code underdetermines
  the answer (multiple designs, missing spec, intended semantics).
- NEVER use the question tool. Ask in plain text when the code
  underdetermines the answer; wait for a plain-text verb.
- ASKING IS ALWAYS ALLOWED, including after being told to proceed.
  "Proceed" authorizes work, it does not silence questions: if the code,
  the scope, or the design underdetermines something mid-task, stop and
  ask in plain text. Silence is not approval, and a guess is worse than a
  question — but a question is never an excuse to skip work that is
  actually determined.
- NEVER ASSUME WHICH OPS A PASS SEES. High-level ops are consumed by
  earlier passes: `linearize` rewrites `Op::Reduce` into a `Loop` with a
  scalar accumulator, and `Op::Reduce` only reappears when `split.rs`
  unrolls a loop into lanes (a reduce over vector/tile unroll dims, never
  a loop). The same is true of every other op a frontend emits. Before
  matching on an op, counting it, or writing a predicate that keys on it,
  print the IR at that boundary (`ZYX_DEBUG=8`, or a temporary dump at
  the pass entry, removed afterwards). A predicate written from the
  frontend's vocabulary instead of the actual IR is a guess, and it is
  how the `Op::Reduce` mistake happened.
- User owns design. Analysis: thorough. Implementation: ask, don't choose.
- ASK BEFORE ADDING ANY FUNCTION OR CHANGING A SIGNATURE
  (params, return, generics, `Result`), incl. private helpers.
- ASK BEFORE CHANGING ANY DATA STRUCTURE'S SHAPE
  (fields, variants, aliases). Name field, type, placement, consumers.
- `?`/interrogative = NO EDITS in that turn, never. Reads AND runs
  are allowed (read files, bash, build, test) to support the answer;
  nothing that mutates files is. Edits only after an explicit verb
  (`fix`, `remove`, `rewrite`, `proceed`, ...) in a LATER message.
- QUESTION = NO FILE MUTATIONS, no exceptions. A message that asks
  anything — or states a puzzle/observation without an explicit verb —
  authorizes reads and runs ONLY. It NEVER authorizes creating, editing,
  or deleting any file: not product code, not tests, not `eprintln`
  instrumentation, not temp files. "Investigate freely" means reads
  and runs. A turn that starts with a question must leave the tree
  identical to how it found it; violation = stop and revert.
- A question ANYWHERE in a message (trailing `or not?`, `right?`,
  `is X Y?`, `why ...?`) makes the whole message a question: answer +
  investigate (reads and runs only) freely, but no edit in that turn
  even if the message also contains an order-like verb. The order part
  is parked until restated or confirmed in a LATER message. Never
  substitute implementation for an explanation.
- No inferring authorization from tone: frustration, rhetorical `why`,
  restated complaints, and obvious next steps NEVER count as an
  explicit verb. Only a verb-directed instruction (`fix X`, `remove Y`,
  `proceed`, ...) in a message containing NO question authorizes
  action. In doubt, the message is a question.
- A question is NEVER an order, no matter how actionable the answer
  looks, no matter if work feels "owed" or is an obvious follow-through.
  Answer + investigate, then STOP and wait for an explicit verb
  (`fix`, `remove`, `rewrite`, `proceed`, ...). Answering and editing
  in the same turn is PROHIBITED when the trigger was a question.
  No exception for obvious one-line fixes, already-diagnosed causes,
  or follow-up questions (`why TWICE?` is still a question).
- "Why did you X?" explains only, never undoes. Undo needs an explicit verb.
- Literal ask only ("a test" = ONE test). Ambiguous scope: ask first.
- NO design decisions, EVER. `todo!()` beats a guessed fix.
- Never go beyond the ask. Failure? Report, don't fix untold.
- NEVER chain fixes. One fix per question. Never commit unless asked.
- DO THE ASK. An explicit instruction names a method AND a target
  ("measure each llama block individually with ZYX_DEBUG=1/8"). Execute
  exactly that. Substituting a broader, easier or adjacent task (e.g.
  benching the whole model instead of one block) is a failed turn even
  if the substitute produces data. If the asked-for artifact is
  underspecified (where a new harness lives, which block, which shape),
  ASK FIRST and do nothing else: a guess at the shape of the work is
  worse than one question, and running the wrong measurement is not
  "progress".
- A missing prerequisite (no harness, no database, no tool) is reported
  the moment it is found, together with the ONE decision needed to
  unblock it. Never spend the turn on unasked-for work and report the
  prerequisite afterwards.
- NEVER disable, revert, remove, or neuter user-ordered functionality
  to get green (no silent unregistrations, no refusal-gates that neuter,
  no reordering user registries). Red tests from ordered features are
  reported, never self-resolved by removal. Registries
  (`default_optimizations`, pass lists) change ONLY as ordered:
  additions go exactly where specified, existing entries keep their
  relative order, nothing is removed without an explicit verb.
- Never leave temporary debug code behind; revert completely.
- Never `stash`/`checkout`; use `git restore`. Edit precisely, don't cascade.
- RT lock discipline: a live guard must never overlap a `Tensor` drop
  (`drop` spins in `try_lock`). Bind ids in their own statement first
  (`let nid = RT.lock().f(cur.id)?; cur = Tensor { id: nid };`).
- Forbidden words: "Actually", "wait", "key insight", "let me", "Hmm", "But wait".

## Pass budget: 30us max per invocation for the REPEATED passes
- Applies to `default_epilogue` and the `default_optimizations` list as a
  whole: those run once per candidate inside the search, so a blowout is a
  performance bug.
- Does NOT apply to `device_epilogue` (the hardware lowering: `tt_*`). It runs
  once, after the search, so a `Map`/`Set` where it reads better is fine.
- Fix algorithmic cost. Measure with `time_pass!`/Instant (stderr).

## Commands (run cargo from `zyx/`; root has no Cargo.toml)
```bash
cd zyx && cargo build
cd zyx && AGENT=1 cargo test               # strips ANSI colors
cd zyx && AGENT=1 cargo test --test <file> <name>
cd zyx && AGENT=1 cargo test -- --nocapture   # ALWAYS for debug output
cd zyx && cargo fmt
```
- TT: `TT_METAL_RUNTIME_ROOT=/home/x/Dev/cpp/tt-metal cargo build --features tenstorrent`
- TT sim: export `TT_METAL_SIMULATOR=~/Dev/cpp/tt-sim/libttsim_bh.so`
  + `TT_METAL_SLOW_DISPATCH_MODE=1` before `cargo test --features tenstorrent`.
- Clippy is NOT a gate. Python: `python3.12`. Tests return `Result<(), ZyxError>`.
- Timeout cap: 10s MAX per individual test RUN (one test only, not the suite).
  Test execution only, not compilation, not tool use in general.
  No exceptions, no stacking.
- If all backends fail init, tests print nothing (C is always on;
  CUDA/HIP/etc. skip silently without drivers).

## Testing: one tier per step

Run the smallest tier that can catch the mistake. Never run the whole
suite after every edit; never make a commit wait for the push gate.

| step | what runs |
|------|-----------|
| after an edit | ONLY the test that is failing / the suite being fixed: `cd zyx && AGENT=1 cargo test --test <file> <name> -- --exact`. No sweep, no other suites. |
| before a commit | the zyx Rust test suite: `cd zyx && AGENT=1 ZYX_DEV=cpu cargo test --lib --tests -- --skip mnist`. Other crates and the non-Rust checks are not part of a commit. |
| before a push | `./run_ci.sh` (repo root): zyx suite, zyx-nn/zyx-optim/zyx-derive tests, doctests, private-item rustdoc, `cargo fmt --check`, examples check + run, zyx-py (when torch/pytest/maturin exist), book examples. |

- Pin the device with `ZYX_DEV` only. NEVER set `XDG_CONFIG_HOME`: the
  default config dir is the warm kernel cache (compiled kernels live in
  `$XDG_CONFIG_HOME/zyx/cache/`), and a fresh dir makes every run recompile
  every kernel — several times slower for no signal.
- `08_graph`-style failures when no `ZYX_DEV` is pinned are autotune/device
  selection, not necessarily the change under test; re-run pinned first.
- A pre-existing red suite is reported, not folded into the current change.

## Layout
```
zyx/src/backend/  one file per backend
zyx/src/graph/    egraph + autograd + kernelizer + plan
zyx/src/kernel/   kernel IR, codegen, ALL optimization passes
```

## Graph / egraph
- Lazy, dynamic. One graph for laziness AND autograd; TensorId is `u32`.
- `src/graph/` OUTSIDE `src/kernel/` (no imports either way).
  Fusion variants -> seeds -> autotune -> measured timings -> extract.
  NO cost model; rewrites exact (wrong rewrite = WRONG kernel).
- Custom kernels BYPASS egraph AND autotune. `linearize` first.

## Symbolic dims
- Symbolic everywhere, eager AND graph. Never assume concrete shapes.
- Dims bottom out in `Param { Variable }` (`IDX_T` = I64). Never cast to `u32`.
- Variables const-backed, never unknown. Compare FOLDED values, never identity.
- No fabricated sentinels. No `unwrap_or(n)`.
- Resolve an `OpId` via `resolve_const(op).and_then(Constant::as_dim)`;
  there is no `Kernel::resolve_dim`.

## Style
- Debuggability > "clean". Duplicate first, abstract later. Enums, not dyn.
  ~1000 LOC/module. Imports: `crate::` -> `super::` -> external.
- `todo!()` beats silent failure. New op variant = CORRECT arm in EVERY
  match (or `todo!()`); verify by build + match audit.
- No autocasting. Dtype is caller's job; overflow fixed at call site.
- New files: license header + docs + `#[track_caller]` errors.

## TT safety (board irreplaceable; resets are the USER's alone, NEVER)
- `cargo test` allowed. Failed init WEDGES board (0xffffffff reads):
  STOP, report, ASK. No retries, no further access.
- Aborted/hung run strands cores (heat): afterwards NO device access at all.
- Invalid IR REJECTED at compile time, never wedged at runtime.
- Passing compile = gate to launch. `ZYX_DRY_RUN=1`: launches are no-ops.
- Custom-kernel tests: seconds or hung (>10s = abort).

## gws (backend-only)
- `Group` length per dim: `Const` -> `GwsDim::Const`,
  `Param { Variable }` -> `GwsDim::Param(ordinal)`, else unreachable.
- Launch `args` = head order of `Param` defines in compiled IR.

## Passes (`zyx/src/kernel/`, driver `autotune.rs`)
- `linearize` first. Then ANY order, ANY repetition. Broken sequence =
  producing pass is BUGGY. Full `cargo test` after touching any pass.
- `BeamSearch`: seeds -> `MakeOpt` variants -> `cost` rank ->
  top `n_launches` run -> measured `nanos` wins. Not for custom kernels.
- New pass: `Optimization` + `MakeOpt`, register in `default_optimizations()`.
  Preserve semantics; gate or `todo!()`, never alias wrong.
- `Box<dyn Optimization>` forever (users write passes). Vtable `apply`
  inlines away: driver-loop verify names NO pass (see Debugging).
- `apply_seq` = `opt.apply` -> epilogue: every variant is epilogue-cleaned.

## Debugging
1. Isolate: `cd zyx && AGENT=1 cargo test --test <file> <name> -- --exact`.
   Shared RT: a panic poisons the rest; later failures are cascade, not signal.
2. Verify panic -> backtrace names the pass (caller of `Kernel::verify`).
   Multiple rounds in one variant don't change this: a later round's
   verify still names that round's pass.
   Driver-loop verify (jumps to `run_with_launch_args`) names nothing:
   use `(opt_id, config_id)` prints or start/end dumps around `opt.apply`.
3. Wrong values -> `ZYX_DEBUG=8` IR + `ZYX_DEBUG=16` code; diff seed vs
   winner op-by-op; unit-test that IR; fix the pass.
4. `NoSearch` (`ZYX_DEBUG=256`) vs winner: correct = search opt guilty,
   broken = seed/epilogue/codegen. IR-level A/B ONLY. Never commit
   archaeology (worktrees, bisection, banner greps) unless user orders a ref.
5. `kernel.debug()` at pass START and END, compare. If `debug()` panics,
   dump ops manually. Never "fix" the printer.
6. `eprintln` markers WITHOUT asking on order turns only — never on a
   question turn (question = no file mutations); remove before finishing.
7. Ask only what code cannot settle. One fix per question. Each new
   failure = NEW task: stop, report panic + location, ask first.
8. Output needs `-- --nocapture`; never pipe it through grep/head.
- Segfault: reproducer + `panic!("A")`/`panic!("B")` (panics flush).
  Hang: `println!` + `-- --nocapture` + `timeout 10`; LAST marker localizes.
- A `-1` dim as loop/group length hangs (~4.29e9 loop). `verify` must
  catch it loudly, never hang.
- Lock watchdog in `src/mutex.rs` (commented out): uncomment to name holder.
- GPU launch fail: signature arg order vs buffers; check eager
  (`tests/03_movement.rs`) to isolate graph bugs.
- User suspects a pass / names a bug class: surface symptom + IR EARLY, ask.

### ZYX_DEBUG (bitmask; sum)
| 1 devices | 2 egraph | 4 pre-linearize IR | 8 post-opt IR |
| 16 code | 32 launches | 64 alloc | 128 compile | 256 NoSearch |
- `timeout 10 bash -c 'ZYX_DEBUG=8 cargo run 2>&1' > /tmp/ir.txt`.

### ZYX_DRY_RUN
- `=1`: launches skipped (1s placeholders). `ZYX_DEBUG=24` captures IR+code.
  Buffers uninitialized: never read. Winners arbitrary.

## Tape
- `realize` takes persistent state only, consumes tape. No `Tensor::realize()`.
- Graph promotion: registered (`new`/`add`) or binary op with graph operand.
- `item()` forces; `shape()` lazy. Failing test = valid bug doc.
- An `Op` change lands in BOTH eager and graph paths unless they share
  a constructor.
- NULL `shape` = SCALAR (rank 0), display convention.

## Defensive programming
- Fail loudly at the spot (`expect`/`todo!()`). No fake values
  (`unwrap_or(n)`, sentinels, default arms, silent clears).
- No silent fallthrough. A wildcard arm must be loud: `_ => todo!()` or
  `unreachable!()`, never `_ => {}`/`None`/`false`/a made-up value.
  Grouping variants in one arm (`A | B` with the same body) is fine, and a
  new variant may land behind a loud catch-all until its real arm exists:
  `todo!()` is a report, `_ => {}` is a silent bug. The goal is no silent
  bugs, not exhaustive busywork.
- Full resolution (`resolve_const`) over trivial-case matching.
- Never enforce a should-be invariant with real code (`clear()`, re-init
  "just in case" hides the leak). `debug_assert!` it or leave it alone.
