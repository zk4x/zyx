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
- User owns design. Analysis: thorough. Implementation: ask, don't choose.
- ASK BEFORE ADDING ANY FUNCTION OR CHANGING A SIGNATURE
  (params, return, generics, `Result`), incl. private helpers.
- ASK BEFORE CHANGING ANY DATA STRUCTURE'S SHAPE
  (fields, variants, aliases). Name field, type, placement, consumers.
- `?`/interrogative = NO EDITS in that turn, never. Reads AND runs
  are allowed (read files, bash, build, test) to support the answer;
  nothing that mutates files is. Edits only after an explicit verb
  (`fix`, `remove`, `rewrite`, `proceed`, ...) in a LATER message.
- A question ANYWHERE in a message (trailing `or not?`, `right?`,
  `is X Y?`, `why ...?`) makes the whole message a question: answer +
  investigate freely, but no edit in that turn even if the message also
  contains an order-like verb. The order part is parked until restated
  or confirmed in a LATER message. Never substitute implementation for
  an explanation.
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

## Pass budget: 30us max per invocation in `zyx/src/kernel/`
- Applies to `default_epilogue` as a whole. Blowout = performance bug.
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
- If all backends fail init, tests print nothing (C is always on;
  CUDA/HIP/etc. skip silently without drivers).

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
- Custom-kernel tests: seconds or hung (>60s = abort).

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
6. `eprintln` markers WITHOUT asking; remove before finishing.
7. Ask only what code cannot settle. One fix per question. Each new
   failure = NEW task: stop, report panic + location, ask first.
8. Output needs `-- --nocapture`; never pipe it through grep/head.
- Segfault: reproducer + `panic!("A")`/`panic!("B")` (panics flush).
  Hang: `println!` + `-- --nocapture` + `timeout N`; LAST marker localizes.
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
- Every variant its own arm (duplicate bodies, no `A | B`). Only trailing
  `unreachable!()`/`todo!()` after all real arms. Fallthrough needs approval.
  `_ => {}`/`None`/`false` forbidden; `_ => todo!()` desired.
- Full resolution (`resolve_const`) over trivial-case matching.
- Never enforce a should-be invariant with real code (`clear()`, re-init
  "just in case" hides the leak). `debug_assert!` it or leave it alone.
