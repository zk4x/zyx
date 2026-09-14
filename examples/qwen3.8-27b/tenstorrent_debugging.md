# Tenstorrent debugging log — pad_passthrough_tt

Date: 2026-09-10.
Test: `examples/qwen3.8-27b/tests/pad.rs:157` `pad_passthrough_tt_run`.
Builder: `examples/qwen3.8-27b/src/lib.rs:204` `pad_passthrough_tt`.
Constraint: tt-metal 0.72 read-only. No tt-metal edits, no tt-metal recompile. Fixes belong in zyx. Test-only diagnostic prints are acceptable.

## The five cores and the operations they run

There are 5 cores. Reader and writer touch only DRAM and CBs. Compute (pack, math, unpack) touch only CBs and registers.

| Core   | Role                                        | Touches registers? |
|--------|---------------------------------------------|--------------------|
| reader | DRAM → CB                                   | no                 |
| pack   | CB → registers (layout mgmt)                | yes                |
| math   | registers → registers (compute)             | yes                |
| unpack | registers → CB (reverse of pack)            | yes                |
| writer | CB → DRAM                                   | no                 |

### Operations on CBs (all five cores use these)

- `reserve` — reserve space in the CB for one tile.
- `push` — tell compute/writer that the CB value is ready for reading.
- `wait` — wait for the CB value to appear from a `push`.
- `pop` — mark the value as read and no longer needed (frees the slot).

### Operations on DST registers

- By pack/unpack:
  - `lock` = `tile_regs_wait` (PACK-side lock; waits for MATH to commit)
  - `unlock` = `tile_regs_release` (PACK-side unlock)
- By math core:
  - `lock` = `tile_regs_acquire` (MATH-side lock)
  - `unlock` = `tile_regs_commit` (MATH-side unlock)
  - `tile_regs_acquire` zeroes DST (confirmed 2026-09-14): fresh accumulation state on lock; re-acquire inside a loop resets the acc, so lazy acquire text must hoist out of accumulation loops.

MATH and PACK dispatch to separate per-thread queues, so source-line order between a `MATH(...)` block and a `PACK(...)` block is *not* execution order.

### Movement between CBs and registers

- `copy_tile` — CB → registers
- `pack_tile` — registers → CB

### Movement between DRAM and CBs

- `noc_async_read` — DRAM → CB (reader)
- `noc_async_write` — CB → DRAM (writer)
- plus the associated barriers (`noc_barrier`, `init_interfaces`, etc.)

## Builder shape (lib.rs:204-245)

- `data = param(F32)`, `out = param_mut(F16)`.
- `cdata = storage(F32, Circular, 1024)`, `cout = storage(F16, Circular, 1024)`.
- Section 0 reader loop: `load_tile(data, tbase)` -> `store_tile(cdata, zero)`.
- Barrier.
- Section 1 compute loop: `load_tile(cdata, zero)` -> `cast(F16)` -> `store_tile(cout, zero)`.
- Barrier.
- Section 2 writer loop: `load_tile(cout, zero)` -> `store_tile(out, tbase)`.
- `TILES = 160` for `[32, 5120]` tilized input in the test.

## Pasted kernels under test (from user message)

Reader (F32 in, page 4096):

```cpp
uint32_t src0 = get_arg_val<uint32_t>(0);
DEVICE_PRINT("TEMP src0={}\n", src0);
auto args0 = TensorAccessorArgs<0>(0);
auto p0 = TensorAccessor(args0, src0, 4096);
CircularBuffer cb0(tt::CBIndex::c_0);
CircularBuffer cb1(tt::CBIndex::c_1);
// loop r8=160, r9=1024: noc_async_read 4096 B per tile, cb0.push_back(1)
```

Compute:

```cpp
CircularBuffer cb0(tt::CBIndex::c_0);
CircularBuffer cb1(tt::CBIndex::c_1);
init_sfpu(0, 1);
typecast_tile_init<0, 1>();
// per iteration:
//   cb0.wait_front(1); tile_regs_acquire();
//   copy_tile_init(0); copy_tile(0, 0, 0);
//   typecast_tile<0, 1>(0);
//   tile_regs_commit(); tile_regs_wait();
//   cb1.reserve_back(1); pack_tile(0, 1); cb1.push_back(1);
//   cb0.pop_front(1); tile_regs_release();
```

Writer (F16 out, page 2048):

```cpp
uint32_t out1 = get_arg_val<uint32_t>(0);
auto args_out1 = TensorAccessorArgs<0>(0);
auto p_out1 = TensorAccessor(args_out1, out1, 2048);
// loop r8=160: cb1 wait, noc_async_write 2048 B per tile, cb1.pop_front(1)
```

## Zyx emission mapping

- `tt_dtype_format`: `zyx/src/codegen/tenstorrent.rs:29-36`. F32=0, F16=1, BF16=5. Used for `typecast_tile<in_fmt, out_fmt>`, so `<0, 1>` matches F32->F16.
- Reader `TensorAccessor(..., page_size)`: `zyx/src/codegen/tenstorrent.rs:348-354`. F32=4096, F16/BF16=2048. Pasted `4096` matches F32 input.
- Reader `DEVICE_PRINT("TEMP src...")`: `zyx/src/codegen/tenstorrent.rs:342`. Leftover TEMP diagnostic in codegen; harmless for values but clutters device output.
- Reader noc addr math: `zyx/src/codegen/tenstorrent.rs:438,459`. `(idx*elem_size)/page_size` pattern. Pasted `(r12*4)/4096` matches F32 elem_size=4.
- Compute `init_sfpu(in0, out0)`: `zyx/src/codegen/tenstorrent.rs:999-1017`. `in0` found by scanning compute for first `Load` with CB source; `out0` from first tile `Store` CB. Pasted `init_sfpu(0, 1)` matches cdata=0, cout=1.
- Compute `typecast_tile_init`: `zyx/src/codegen/tenstorrent.rs:1027-1029`. Emitted once pre-loop from collected `typecast_inits`.
- Compute per-`Load` copy: `zyx/src/codegen/tenstorrent.rs:1253-1267`. `copy_tile_init(cb_id)` + `copy_tile(cb_id, 0, slot)` per tile in loop. Pasted `copy_tile_init(0)` + `copy_tile(0, 0, 0)` matches.
- Compute `Cast` to `typecast_tile`: `zyx/src/codegen/tenstorrent.rs:1287-1298`. Emits only when `in_fmt != out_fmt`. Pasted `typecast_tile<0, 1>(0)` matches.
- Compute `pack_tile(slot, cb)`: `zyx/src/codegen/tenstorrent.rs:1445,1494`. Pasted `pack_tile(0, 1)` matches.
- Compute guards: `zyx/src/codegen/tenstorrent.rs:1517-1534`. `init_sfpu` + `mm_init` coexistence panics; `copy_tile` under `mm_init` errors. Neither fires here (no matmul).
- Writer `TensorAccessor(..., page_size)`: `zyx/src/codegen/tenstorrent.rs:1590-1595`. F32=4096, F16/BF16=2048. Pasted `2048` matches F16 output.
- Writer noc addr math: `zyx/src/codegen/tenstorrent.rs:1783,1800`. Pasted `(r24*2)/2048` matches F16 elem_size=2.

## CB config mapping (zyx side)

- `cb_map` build: `zyx/src/backend/tenstorrent.rs:746-794`. Sections split by `Barrier`. Reader/writer sections (`section != 1`) register CBs on `Load src` and `Store dst`; compute section CBs reuse those ids. Comment states one id per CB across sections.
- Expected for this kernel: reader `Store cdata` registers cb0; writer `Load cout` registers cb1. Compute reuses both.
- `cb_config` dtype lookup: `zyx/src/backend/tenstorrent.rs:923-940`. For each sorted CB id, find defining op in `cb_map`, read `Op::Storage { dtype }`, map via `dtype_to_tt_fmt` (F32=0, F16=1, BF16=2) and `tile_bytes_of` (F32=4096, F16/BF16=2048).
- Runtime CB creation: `zyx/src/backend/tt_runtime.cpp:602-624`. Maps `0->Float32`, `1->Float16`, `2->Float16_b`, sets page size from `cb_tb`.
- Pasted kernels show cb0 as F32 path and cb1 as F16 path, consistent with `cdata=F32`, `cout=F16`.

## Status of the failure

- `pad_move_tt` passes: empty compute, pure F32 movement. Reader/writer/CB/DRAM path for F32 is sound.
- `pad_copy_tt` was reported with small errors (~0.0003): copy+pack in F32, no cast. Points at compute pack path rather than reader/writer.
- `pad_passthrough_tt` fails badly (~80954/81920, values near +/-1.99 vs expected ~+/-1.2): F32->F16 cast in compute.
- Roundtrip host->TT->host is exact, so tilize/untilize/pool path is sound; corruption is on-device.
- Generated sequence matches the canonical tt-metal typecast shape (`init_sfpu` + `typecast_tile_init` + per-tile `copy_tile_init`/`copy_tile` + `typecast_tile` + `pack_tile`). No `mm_init` clash.

## Open suspects (zyx-side only)

1. CB `fmt` / `tb` actually sent for cb1 on this exact launch. The pasted writer uses 2048, which is right for F16, but the runtime CB creation log (`cb_idx/cb_fmt/cb_tb`) for this run has not been captured here yet. A mismatch here would corrupt pack/writer even with correct C++.
2. `init_sfpu(in0, out0)` id selection if `cb_map` iteration order ever differs from scan order. Codegen now scans for first compute `Load` CB (`tenstorrent.rs:1004-1014`) and first tile `Store` CB for `out0`, so pasted `(0, 1)` is the intended pair. Still worth confirming on the failing run.
3. Per-tile `copy_tile_init(0)` inside the loop interacting with `typecast_tile_init<0,1>` pre-loop. Canonical examples do this, but the exact tt-metal 0.72 init ordering for F32->F16 is the remaining unknown. Read-only check against 0.72 `typecast.h` / `eltwise_unary.h` needed.
4. Reader leftover `DEVICE_PRINT` at `codegen/tenstorrent.rs:342`. Should be removed as cleanup; not suspected for values.
5. `fake_jit_prelude.h` format arrays are all 255 in tt-metal 0.72 checkout. Read-only note: not to be edited, no recompile. If the root cause is proven to live there, the next step is GitHub issues, not a local patch.

## Next test-only diagnostics (no tt-metal edits)

- Remove codegen `DEVICE_PRINT` TEMP at `codegen/tenstorrent.rs:342` (test-only cleanup, still zyx-side).
- Add temporary `eprintln!` in `tests/pad.rs` passthrough path dumping first mismatching tile index and raw F16 bits vs expected bits, to see whether errors cluster per tile or per row.
- Capture runtime CB config log for this exact launch (`cb_idx/cb_fmt/cb_tb` lines) alongside ZYX_DEBUG=16 output, then compare `cb_tb` against the `TensorAccessor` page sizes in reader/writer.

## 2026-09-10 — bits step (test-only diagnostic added)

- Changed `tests/pad.rs` passthrough mismatch print to include tile and offset (`tile = i / 1024`, `off = i % 1024`) alongside got/expected f32. No kernel or backend change.
- Raw F16 bit dump still open: need the preferred accessor for `zyx::f16` bits before printing hex. Pending user guidance on which accessor to use.
- Hardware run not executed here (board-touching runs stay with the user). Next: user runs the passthrough test and pastes the first mismatch lines.

## 2026-09-10 — bits step run output

- Command (agent-run, board reset untouched): `AGENT=1 TT_METAL_ROOT=/home/x/Dev/cpp/tt-metal cargo test -p qwen3-8-27b --test pad pad_passthrough_tt_run -- --nocapture` from `examples/`.
- `roundtrip bad: 0 / 81920`. Host tilize/untilize/pool path exact.
- `compile_program`: `n_cbs=2`, `n_params=2`, reader 1 param, compute 0, writer 1. Out alloc: `size=327688 tile_bytes=2048` (160 tiles * 2048 = 327680, +8 header).
- `run`: `gd0=1,gd1=1, src0=0, dst0=1`. `reader_rt: 1048704 0 0 reader_ct: 2 4096`. `writer_rt: 1142912 0 0 writer_ct: 2 4096`.
- First mismatches all in tile 0, offsets 0-9. Got values cluster near +/-1.93..1.99 regardless of expected sign/magnitude:
  - `[0] tile=0 off=0 got=-1.9873047 expected=-1.2109375`
  - `[1] tile=0 off=1 got=1.9726563 expected=0.6303711`
  - `[5] tile=0 off=5 got=1.9375 expected=0.13171387`
- `passthrough bad: 81722 / 81920` this run (earlier noted 80954 — count varies run to run; flag as possible nondeterminism, or different binary after diagnostic edit).
- Note: `RUST_SEND compile_program` log truncates JSON at 200 chars, so `cb_fmt`/`cb_tb` values are NOT visible in this output. That motivates the CB step below.
- Raw F16 hex bits still not dumped (no accessor chosen yet).

## 2026-09-10 — CB step run output

- Temporary `TT_CB_CONFIG` print in `backend/tenstorrent.rs` compile path showed: `cb0:fmt=0:tb=4096 cb1:fmt=1:tb=2048`.
- That matches the intended mapping (cdata F32 -> 0/4096, cout F16 -> 1/2048) and the pasted `TensorAccessor` page sizes (reader 4096, writer 2048).
- Conclusion: CB `fmt`/`tb` sent to the runtime is NOT the culprit on this run. Temporary backend print removed after capture; test tile print in `tests/pad.rs` retained.
- Test still panics at `tests/pad.rs:209` (`assert_eq!(bad, 0)`, left 81722 right 0). Same pre-existing failure mode, no new panic.

## 2026-09-10 — SUBSTANTIAL: DRAM accessor page size must equal buffer page size

- Mechanism (read-only, `tensor_accessor.h:315`): NOC address = `bank_start + base + bank_page_offset * aligned_page_size + offset`. The stride is the ACCESSOR's page size, so it must equal the DRAM buffer page size or every tile past tile 0 misaddresses (page 0 offset 0 is correct under any stride).
- Zyx was emitting dtype tile size instead: F16 accessors used 2048 (`codegen/tenstorrent.rs` reader/writer creation + page/offset arithmetic), while `tt_runtime.cpp` `alloc_buf` always allocates 4096-paged DRAM (`PAGE_SIZE`). Tile 0 passed by luck; tiles 1+ landed in wrong banks/offsets. F32 traffic (accessor 4096 = buffer 4096) always worked — hence move/copy passing and only F16 output broken.
- Fix (zyx-side only, no protocol change): new `TT_DRAM_PAGE_BYTES = 4096` const in codegen with cross-reference comment to `tt_runtime.cpp` `PAGE_SIZE`; all five DRAM `TensorAccessor`/arithmetic sites (reader Global + GlobalMut creation, reader arithmetic, writer creation, writer arithmetic) stride by it. CB-side (L1) paging untouched. Unused dtype bindings dropped (no new warnings).
- Result: `passthrough bad: 81722 -> 716`. Every remaining mismatch is exactly 1 ulp of truncation (device truncates F32->F16, host `f16::from_f32` rounds; e.g. `got=0x4118 exp=0x4119`). `bad small/large: 8/708` — fails concentrate at |exp| >= 1.0 where one fp16 ulp exceeds the 1e-3 test tol.
- No regressions: copy unchanged (30421 bad at 1e-6 tol, same ~3e-4 tracking), move passes, zyx lib warning-free.
- Open: remaining 716 are test-tolerance vs device-truncation, not pipeline error. Test prints (tile/bits/row histograms) retained for now.

## 2026-09-10 — SUBSTANTIAL: F32->F16(1) has no SFPU kernel in 0.72 (read-only)

- `typecast.h` doc list (`tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:17-42`): every half-precision entry names `Float16_b`, never plain `Float16`. F32 targets listed: `Float16_b`, `Int32`, `UInt16`, `UInt32`, `UInt8`, `Bfp8_b`, `Bfp4_b`. No `Float32 <-> Float16` entry.
- Blackhole SFPU impl (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h`): `llk_math_eltwise_unary_sfpu_typecast` (lines 18-155) and `_init` (lines 163-217) contain zero branches mentioning `DataFormat::Float16` (1). All half branches use `Float16_b` (5); the implemented F32 downcast is `Float32 -> Float16_b` (`calculate_typecast_fp32_to_fp16b`, `init_typecast_fp32_to_fp16b`).
- Our kernel emits `typecast_tile_init<0, 1>()` + `typecast_tile<0, 1>(0)` (zyx `tt_dtype_format`: F32=0, F16=1 at `zyx/src/codegen/tenstorrent.rs:29-36`). That pair matches NO branch: the per-tile op compiles to an empty body (DST keeps `copy_tile` F32 bits), `_init` falls to the default `else` init. `pack_tile(0, 1)` then packs those untouched bits as F16. That predicts exactly the observed symptom: values near +/-1.93..1.99 independent of input, from tile 0 onward.
- Consistency check: `pad_copy_tt` (no typecast, copy+pack to F32/F16 CBs) passes with only small errors; `pad_move_tt` passes. Only the SFPU typecast path is broken. CB `fmt`/`tb` verified correct (`cb0:fmt=0:tb=4096 cb1:fmt=1:tb=2048`), so CB creation is ruled out.
- Fix direction (zyx-side, no tt-metal edits): emit the implemented combo for zyx F16 casts, i.e. `typecast_tile[_init]<0, 5>` (Float32 -> Float16_b), and confirm pack-to-F16-CB still applies. Open design point: whether `tt_dtype_format` F16 should become 5 for the typecast params only, or CB fmt mapping changes too. No code changed yet; awaiting user call.

## 2026-09-10 — fix attempt: typecast params <0,1> -> <0,5> (CB mapping kept)

- User approved trying the fix; CB question answered: CB mapping kept (CBs hold IEEE-half bits the host reads; copy path packs to F16 CB fine; `tt_dtype_format` feeds only typecast template params at codegen lines 874-875 and 1293-1294, separate from backend CB fmt mapping).
- Edit: `zyx/src/codegen/tenstorrent.rs:29-36` `tt_dtype_format` F16 `1 -> 5` with comment citing 0.72 `typecast.h` supported list.
- ZYX_DEBUG=16 confirms the new emission: `typecast_tile_init<0, 5>();` + `typecast_tile<0, 5>(0);` (reader 866 B / compute 1291 B / writer 895 B, same lengths as before since only one digit changed).
- Result: output byte-identical (`passthrough[0] tile=0 off=0 got=-1.9873047 expected=-1.2109375`, bad 81720/81920). The template-param change had no visible effect.
- Read: either (a) the device ran a stale cached kernel binary, or (b) the SFPU op still does not execute and the corruption sits downstream (pack/DST-mode config) or upstream (unpacker input to DST). Next diagnostics: confirm recompile vs cache for `compile_program id:0`, then inspect pack path (`pack_tile(0,1)` config, `DST_ACCUM_MODE=false` vs compute `fp32_dest_acc_en=true` in `tt_runtime.cpp` compute config).

## 2026-09-10 — SUBSTANTIAL: uniform 1.0 -> uniform 0x3FF0, stuck exponent

- Experiment (temporary, since reverted): fed uniform `1.0f` input (`in=0x3f800000` everywhere). Result: uniform `got=0x3ff0` (1.984375), `exp=0x3c00` (1.0), `passthrough bad: 81920/81920`. Fully deterministic, position-independent.
- Combined with real-data runs: mapping is a deterministic per-bit-pattern function with preserved sign, stuck fp16 exponent 15 (values in [1,2)), data-dependent low mantissa bits (~0x3C0-0x3F7). Zero inputs also corrupt (only ~200/81920 pass with real data, far fewer than the 66560 zero elements), so corruption has a data-independent component.
- `pad_copy_tt` re-run: values track expected to ~3e-4 (`copy[0] = -1.2109375, expected -1.2113045`, bad 30421/81920 at 1e-6 tol). Reader/unpack/copy/pack-to-F32/writer leg is sound.
- Pack-isolation attempt (skip cast, store F32 to F16 CB) correctly rejected by the balance checker (`tenstorrent.rs:291`: compute pushes 655360 vs writer 327680). Reverted. Side benefit: confirms byte accounting is dtype-aware and coherent.
- Ranked suspects now: (1) pack DST->F16-CB / DST-mode config, (2) SFPU executing on misformatted DST input, (3) stale device binary (not falsified yet).
- Test bits print (`tile/off/in/got/exp` hex) retained in `tests/pad.rs`; uniform-input hack reverted; `lib.rs` builder restored verbatim.

## 2026-09-10 — SUBSTANTIAL: fp32_dest_acc_en=false flips garbage to mostly-working

- Observation from tt-metal's own tree: passing float SFPU tests/examples use `fp32_dest_acc_en=false` (e.g. `test_sfpu_compute.cpp:748` sets it `= is_int8_op`; programming examples comment "We don't need the destination accumulator to be FP32 as input and output are BFP16"). Zyx `tt_runtime.cpp:666` forces `true` unconditionally for every compute kernel.
- Diagnostic flip (TEMP, in zyx's own shim, quick rebuild, no tt-metal changes): `fp32_dest_acc_en` true -> false.
- Result: `passthrough bad: 81722 -> 26404`. First mismatch moved from index 0 to index 10, and index 10 is a 1-ulp truncation artifact (`in=0x40231ab9`, got `0x4118` vs exp `0x4119`; at magnitude ~2.5 one fp16 ulp ~0.002 exceeds the 1e-3 test tol). Values now in-family (signs/magnitudes track input).
- Residual pattern under false: tile-0 row 0 passes (modulo truncation ulps); failures start at offset 32 (tile row 1) with large errors (e.g. `off=32 in=0x3ffa1786 got=0x3959 exp=0x3fd1`; sign flips at off 37/38). Row-dependent within a single tile op — points at face/vector-chunk handling or DST-lane layout in the remaining path, NOT at CB fmt (verified) or template params (verified emitted).
- NOT reverted yet: flag still false in worktree. Global flip would endanger F32 matmul accumulation (needs 32-bit DST), so the durable fix is per-kernel dest-acc selection. Pending user call on that design.

## 2026-09-10 — per-kernel dest-acc selection implemented (output dtypes)

- Rule (dtype-based, no op checks): 32-bit DST iff any output dtype is F32; else 16-bit. Passthrough (F16 out) -> false; copy/move (F32 out) -> true.
- Plumbing: `backend/tenstorrent.rs` computes the flag from `output_dtypes` next to the `TTProgram` push, threads it through private `compile_program` into new `fp32_dest_acc` JSON field; `tt_runtime.cpp` stores it in `ProgramConfig` (default 1) and uses it in `ComputeConfig` instead of hardcoded `true`.
- Verification run: passthrough 26404 bad (same improved state as manual flip), copy unchanged tiny errors, move ok. No regressions from the plumbing itself.
- Remaining: residual 26k row-pattern errors under 16-bit DST; index-10 1-ulp truncation (tol too tight at magnitude ~2.5).

## 2026-09-10 — histogram: zeros pass, real rows fail uniformly

- `bad per row: [4403, 4401, 4395, 4401, 4404, 4400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`. Rows 0-5 (real data) fail ~4400/5120 each, nearly identical counts; rows 6-15 (zero padding) have ZERO failures. Earlier "zeros fail" reading was from the old dest_acc=true runs; under 16-bit DST, f(0)=0 exactly.
- `bad small/large: 17961/8443` (|exp| below/above 1.0). Small-magnitude failures rule out pure truncation (fp16 has ample precision below 1.0; device truncation worst case ~1 ulp).
- Revised model: deterministic per-pattern mapping, sign-preserving, stuck exponent for the old mode; under 16-bit DST the flow is right for zeros + ~14% of real elements, wrong for the rest with in-family values.
- `copy_tile_init(cbid)` body verified identical to canonical `copy_tile_to_dst_init_short(cbid, 0, false)`; canonical SFPU test kernel (`eltwise_sfpu_2_0.cpp`) pairs `<op>_tile_init(); <op>_tile(0);` inside the loop — zyx now matches that shape (inline init verified in ZYX_DEBUG=16 output), still identical output.
- Next: face-row-mapped error distribution within tile 0 to pin the geometry.

## 2026-09-11 — any-CB dest-acc rule REGRESSED (81909), then doc-contract fixes (no effect yet)

- Rule changed per user (commit "fix typecast tenstorrent kernel" era): 32-bit DST iff ANY CB (in/out/circular) is F32. Result: passthrough 716 -> 81909, pad_input_tt 26514 -> 81911. Copy unchanged (was already 32-bit), move passes.
- DST flag path verified clean, no inversion: Rust computes `true` (`[TT_DIAG] in=[F32] out=[F16] cb=[F32,F16]`), JSON `"fp32_dest_acc":1`, `extract_u32` parses exact key, `.fp32_dest_acc_en != 0` applied, device prints `compute fp32_dest_acc_en=1`, CBs `idx0 fmt0 tb4096 / idx1 fmt1 tb2048`. JIT derives unpack/pack arrays + DST_ACCUM_MODE from the flag.
- `fp32_accuracy.rst` (the doc): flag=true = 32-bit DST lanes; false (default) = 16-bit DST with FP32 auto-converted on unpack. Kernel contract: `copy_tile_init()` per input CB + `pack_reconfig_data_format()` per output CB, re-called on format switch; skipping either -> "data treated as 16-bit". Canonical order: commit -> reserve -> wait -> reconfig -> pack; tail: release -> push -> pop.
- Applied, all verified live in emitted source, ZERO effect (still 81909): pack_reconfig emission (compute 1293->1373 B), doc-exact pack order, typecast init hoisted pre-loop (CUDA style: collect ops, emit inits up front), tail release->push->pop. Supersedes the old "inline init is canonical" note above.
- One CB = one config shared by reader/compute/writer (CircularBufferConfig has single page_sizes_/format map; kernel doc: single producer/consumer FIFO). Different CBs have own configs. Accessor page (4096, DRAM buffer geometry) vs CB page (tile bytes) independent by design; transfers match CB pages both sides; writer byte-identical to the 716 era. Addressing ruled out (uniform 1.0 -> 0x3FF0 falsifies all permutation-class causes).

## 2026-09-11 — BIT FORENSICS: unpack converts F32->Tf32; typecast is a no-op on it

- Copy-bit probe (F32 compute copy, 32-bit DST): `0xbf9b0c07 -> 0xbf9b0000`, `0x3f215f2e -> 0x3f214000` — low 13 mantissa bits zeroed = **Tf32**. Matches programmed `unpack_conditional_dst_format = Tf32` (`genfiles.cpp:511-513`, fp32 flag + exp_prec==B for [F32,F16]). Unpack-as-programmed + F32 pack work, lossy by design.
- Typecast removal probe: deleting `typecast_tile<0,5>` -> byte-identical stuck-exponent garbage (81912). Typecast neither helps nor corrupts: pack maps Tf32-held DST to garbage identically with/without it.
- Net: mixed [F32-in, F16-out] SFPU chain under 32-bit DST broken at typecast/pack-of-Tf32. Open: is mixed-format SFPU off the supported path, or does the typecast op need different programming for Tf32 input? (TEMP copy bit-print + Cast-arm skip both reverted.)
- All three kernels verified doc-exact (reader reserve/read/barrier/push; compute init-hoisted + doc order; writer wait/write/barrier/pop). Leftover TEMP noise to remove at cleanup: reader `DEVICE_PRINT("TEMP src0=...")` in codegen, `[TT_DIAG]` prints in backend + tt_runtime.cpp, test tile/bits/row histograms in tests/pad.rs.

## 2026-09-11 — TYPECAST SFPU AUDIT: op is mode-unaware, 16-bit-addressed writes miss 32-bit read lanes

- Chain: `typecast_tile<0,5>` -> `llk_math_eltwise_unary_sfpu_typecast` -> `calculate_typecast_fp32_to_fp16b` (tt-llk `ckernel_sfpu_typecast.h:221-268`); init `:594-643` programs only SFPU config regs + LREG12/13 rounding consts. No LUTs, no CB/SRC, no unpacker/packer touches, no dependency on prior-op state. Nothing checks for missing state (would mis-write, not skip).
- The op+init take NO DST_ACCUM_MODE parameter anywhere (contrast tanh/topk which switch on `is_fp32_dest_acc_en`). FP16B stores use 16-bit DST addressing (ADDR_MOD_6 incr 2). Under 32-bit DST the writes land where the BF16-positioned packer doesn't read -> pack sees untouched F32 halves -> byte-identical output with/without the op. No init call fixes it: the addressing has no mode term.
- `typecast.h:44`: 32-bit DST required only when Int32/UInt32/Float32 is an input/output — F32->BF16 nominally runs in 16-BIT DST. Our fused mul(F32, needs 32-bit)+cast+pack in one kernel is off every sanctioned path.
- Sanctioned options: (A) two kernels — F32 compute/pack-F32 in 32-bit DST, then a separate F32->F16 typecast kernel in 16-bit DST (ttnn-style decomposition); (B) one kernel, 32-bit DST, drop SFPU cast, let the packer convert F32->F16 directly (early conversion supported, `cpack_common.h:211-217`) — needs pack_src=Float32 override for an F16 CB, API unknown; (C) exploit exact lane geometry (fragile, LLK-internals-dependent).

## 2026-09-11 — UnpackToDestFp32 FIXES unpack (copy now bit-exact); pack-to-F16 isolated as the breaker

- `unpack_to_dest_mode` vector must be 64 entries (max CB count), not n_cbs — JIT `TT_FATAL`s otherwise (`data_format.cpp:178`). Padded with Default. Derived per-CB in `tt_runtime.cpp` (F32 CB -> UnpackToDestFp32): placement questioned, stays in shim for now (no protocol change); moving to Rust backend is open.
- Result: **copy bad 30421 -> 0** (bit-exact F32 compute copy), zeros rows pass, passthrough 81909 -> 30712, pad_input_tt 81911 -> 30711. Remaining fails are rows 0-5 only, same stuck-exponent signature.
- Typecast removal probe REPEATED with exact-F32 unpack: output byte-identical again (30712). Typecast confirmed no-op twice; the breaker is the pack to the F16 CB under 32-bit DST (pack_src programmed F16b, DST holds F32). No bit model fits (not halves, not rebias, not saturation: 0.13 -> 1.94 boosts the exponent).
- (TEMP DIAG2 skip reverted; typecast op emission restored.)

## 2026-09-11 — PACK LLK AUDIT: packer innocent, typecast guilty; LLK mistake list

- Bit proof: `0x3F80 -> 0x3FF0` = FP32 bits [28:19] leaking into the F16 mantissa — a converter fed a full FP32 word while positioned for a BF16-origin word. Neither clean conversion (would give 1.0) nor identity (would give 1.875).
- 32-bit path assumes every DST word is BF16-origin (upper valid, low zero/don't-care): `Round_10b_mant=0`, `in_data_format/Dstacc=Float16_b` with `Read_32b=1`, late-only BF16->FP16 leg (`cpack_common.h`). The JIT *designs* FP32-DST -> BF16 intermediate -> FP16 L1 (`data_format.cpp:260-261`) — so the SFPU typecast is supposed to leave BF16-origin data, and ours doesn't (proven no-op twice). Pack works as designed; the typecast is the breaker.
- `DST_ACCUM_MODE` is a JIT compile-time const (`genfiles.cpp:616`); runtime toggle `_llk_pack_set_fp32_dest_acc_` RMWs only Read_32b, leaving formats/Dstacc/strides stale. 2-arg `pack_reconfig_data_format` compares only pack_dst — src-only changes silently skip. `llk_pack_init` programs addrmod+MOP+strides but NOT formats/Read_32b/Dstacc (those come from hw-configure via op inits). `are_packers_configured_correctly` checks formats + face_r_dim only — blind to Read_32b/Dstacc/strides/relu/threshold/L1acc.
- LLK mistakes for future direct-RISC emission: (1) template/runtime mode split, (2) reconfig programs a subset (never resets L1-acc/relu/edge/MOP), (3) blind-spot asserts, (4) hidden GPR/counter/ping-pong state, (5) semaphore+clear ordering, (6) STALL pairing + t6 mutex, (7) 4-bit format aliasing (UInt8->Int8, Fp8->Lf8 in-field), (8) dual capacity sources of truth, (9) trailing SETADCZW reset is part of the op.

## 2026-09-11 — LLK TAX: forced splits are software, not hardware; launch-count prediction

- The fused mul(F32)+cast+pack kernel is not hardware-impossible: the packer itself supports F32->BF16 early conversion, and DST can hold F32. What blocks it is two software layers: the mode-unaware SFPU typecast op (no DST_ACCUM_MODE term in its addressing) and the JIT's rigid `pack_src` programming (F16 CB -> F16b src under fp32, no override). A direct-RISC emitter recovers the fusion.
- Cost model: every mixed-format boundary forced into ttnn-style decomposition pays a full reader/compute/writer triplet plus DRAM round-trips for intermediates. qwen3.8-27b fuses ~600 launches (CUDA) vs vLLM ~300 (flash attention); on TT the LLKs push it toward 1000+ per forward pass. At least ~130 forced triplets from mixed-format splits alone, likely more.
- Decision (user-approved): split (A) — F32 compute kernels pack F32 (32-bit DST), standalone F32->F16 typecast kernels run 16-bit DST. Dest-acc rule back to output-based (F32 out -> 32-bit), which gives every split kernel the right mode. New builders: `pad_cast_tt` + `pad_mul_tt`; tests go two-stage with F32 tilized intermediates.

## 2026-09-11 — SPLIT (A) WORKS: 707/716, residual is 1-ulp truncation

- `pad_cast_tt` (F32->F16, 16-bit DST) + `pad_mul_tt` (F32 mask-mul, 32-bit DST) builders; tests two-stage via F32 tilized intermediates (`pad_copy_tt` for passthrough stage 1). Dest-acc rule back to output-based. `unpack_to_dest_mode` DST-aware (Fp32 straight-copy only under 32-bit DST; converting Default in 16-bit DST — straight copy overflows 2KB tile space, caused bit14-stuck + ulp noise).
- Result: passthrough 716, pad_input_tt 707 — every residual is exactly exp-minus-1-ulp (`0x4118` vs `0x4119`), rows 6-15 zero fails, small/large 8/708. Same shape as the original fused 716: unpack-side F32->F16b truncation vs host rounding at 1e-3 tol. Copy/move/roundtrip bit-exact.
- Journey: 81722 -> 716 -> 81909 (any-CB regression) -> 30712 (Fp32 unpack) -> 22310 (split, stuck-bit) -> 716/707 (DST-aware unpack). Open: tolerance decision on the truncation residual. (Resolved below: 1-ulp + FTZ carve-out, all green.)

## 2026-09-11 — HOST f16 BUG, 1-ULP TOLERANCE, DEVICE FTZ: all 6 pad tests pass

- `zyx::f16::from_f32` (`zyx/src/scalar.rs`) rounded mantissa with `& 0x3ff`, dropping the carry: any value rounding up across a power of two halved (e.g. 0.99999 -> 0.5, 12 fails). Fix: increment-then-split (`(mant+1) >> 10`, exponent carry), same for the subnormal path (carry -> 0x0400 min-normal). Passthrough 12 -> 3, pad_input 707 -> 2.
- Second host bug found via outlier `in=0xb8a52ebe (~-7.9e-5) exp=0x8400`: the code treated `exp == -14` as subnormal, but -14 is F16's min-NORMAL exponent (field 1); x in [2^-14, 2^-13) took the subnormal path and collapsed to 0x0400. Condition is now `exp < -14` (true subnormals, x < 2^-14); -14 flows the normal path.
- Tolerance: device truncates F32->F16b on unpack, host rounds — tests allow exactly 1 ulp (`(a_bits - b_bits).abs() > 1`, ±0.0 equal).
- FTZ (device semantics, not a bug): the F32->F16 conversion path flushes F16 subnormals to zero. Goldens like `in=0x374962f1 -> exp=0x00c9` come back `0x0000`. Standard accelerator behavior (subnormals cost area/power; nothing in a transformer cares about 1e-5 vs 0). Tests carve out exactly this: golden with F16 exp-field 0 + nonzero mantissa accepts device ±0.0; anything else still goes through the 1-ulp check.
- Result: all 6 pad tests pass on the P100A (`pad_input`, `pad_normed`, `pad_copy_tt_run`, `pad_move_tt_run`, `pad_passthrough_tt_run`, `pad_input_tt`). All TEMP forensics prints reverted.
- Standing rule for future TT kernels: scope goldens to normal range, or carry the same FTZ carve-out (±0.0 accepted iff the golden is F16-subnormal).

## 2026-09-11 — AUTOMATIC TT OPTIMIZATION IS OFF: `tenstorrent_local` broken, manual kernels first

- `opt_local` (`zyx/tests/06_realize.rs:487`, TT-only failure): backtrace names `split_dim` <- `tenstorrent_local` <- `opt_tenstorrent_tile`. `tenstorrent_local` (`kernel/tenstorrent.rs:152`) creates the group-trip const with `const_idx` (= `push_back`, tail-append) while `split_dim` inserts `Range(Group(len))` before the split point — use-before-declaration, caught by `split_dim`'s own trailing verify. One-line fix (`insert_const_idx_before(op_id, f1)`), user's to apply.
- Decision: no automatic kernel->TT conversion work until manual TT kernels + TT codegen stand on their own. The auto tiling/opt path (`opt_tenstorrent_tile`, `tenstorrent_local`, `split_dim` callers) is untrusted until then; every pass must leave valid IR (each calls verify) and this one doesn't.
- Lesson for pass authors: `const_idx` (tail-append) inside a pass is only safe if a later ordering sweep is guaranteed; positioned inserts need positioned consts.

## 2026-09-11 — GEMM WORKS: single-kernel official-structure matmul, review-then-launch validated

- `gemm_tt(32, 5120, 10240)` passes on the P100A: max_err 0.0082 (< 1e-2 vs torch golden), zero-rows exactly 0. F16 tilized in, F32 tilized out, DST accumulation over Kt=160 in one kernel.
- Representation (user-designed): the running sum is an explicit acc CB — loop-carried values flow through named storage (verify-clean, passes can reason), codegen lowers it to nothing. `init acc` = free (outer acquire zeroes DST); no seed traffic.
- Codegen changes (all in `zyx/src/codegen/tenstorrent.rs`, single/nested-loop emission byte-identical for existing kernels — all 6 pad tests still pass):
  - Outermost-only acquire (`acquire_stack`): nested loops inherit DST state; mirrors official `bmm.cpp:323`.
  - `body_cbs` direct-only: outer loops don't wait/pop inner-loop CBs (was double-pop + wait-deadlock).
  - CB-scratch-seed fold extension: `add(acc_load, matmul)` folds onto DST like const-seed (guarded: CB must be compute-stored and never writer-read).
  - `roundtrip_loads` + `scratch_chains`: outer load-acc/store-cout emits pack of the chain DST slot directly (a `copy_tile` there stalls UNPACK under `mm_init` — wedge guard).
  - Full scratch suppression (`scratch_cbs` set): reader/compute stores, waits, pops, and the end-of-reader drain emit nothing for scratch CBs (two `continue`-hazard fixes: op advance sits at loop end, so guards not `continue`s).
  - Balance: single-output rule = exactly one writer-read CB; compute-local scratch exempt from push/consume byte equality; compute store byte count uses dst CB dtype (matmul typed F16, packs F32).
  - `mm_out_cb` = last circular store (was first; identical for single-store kernels).
- `verify.rs`: C-like F16xF32->F32 promotion in `Binary` (matmul typed F16, holds F32 under 32-bit DST; emission is F32-correct).
- `Tensor::tilize/untilize` are now chainable methods (`from_vec(..)?.tilize()?`; old UFCS call form still compiles).
- REVIEW-THEN-LAUNCH validated end to end: dump-only run (`ZYX_TT_DUMP_ONLY=1`, no board execution) -> line-by-line match vs official (only deviations: sanctioned `pack_reconfig_data_format`, since-fixed phantom `cb0.push_back(4)`) -> launch green. The g6 hang (fixed by the above) wedged the board; user reset. Rule now in AGENTS.md.
- Cost note: 200MB-partials two-kernel split was designed then abandoned — single kernel won once acquire hoisting removed the need for any SFPU add (no mixed `mm_init`+SFPU inits at all).

- The fused mul(F32)+cast+pack kernel is not hardware-impossible: the packer itself supports F32->BF16 early conversion, and DST can hold F32. What blocks it is two software layers: the mode-unaware SFPU typecast op (no DST_ACCUM_MODE term in its addressing) and the JIT's rigid `pack_src` programming (F16 CB -> F16b src under fp32, no override). A direct-RISC emitter recovers the fusion. [tail section]
- Cost model: every mixed-format boundary forced into ttnn-style decomposition pays a full reader/compute/writer triplet plus DRAM round-trips for intermediates. qwen3.8-27b fuses ~600 launches (CUDA) vs vLLM ~300 (flash attention); on TT the LLKs push it toward 1000+ per forward pass. At least ~130 forced triplets from mixed-format splits alone, likely more.
- Decision (user-approved): split (A) — F32 compute kernels pack F32 (32-bit DST), standalone F32->F16 typecast kernels run 16-bit DST. Dest-acc rule back to output-based (F32 out -> 32-bit), which gives every split kernel the right mode. New builders: `pad_cast_tt` + `pad_mul_tt`; tests go two-stage with F32 tilized intermediates.

## 2026-09-11 — dequant bring-up: header bug, U16 gaps, loop-discipline design

- tt-metal 0.72 `ckernel_sfpu_bitwise_and.h:20` does NOT compile for blackhole: `input & value` with 64-bit `uint` is an ambiguous overload (only `vInt`/`int32_t` candidates). trisc1 build dies including it. Read-only headers, so the INT-mask extraction path is dead until TT fixes it. Pivot: float nibble extraction (`n = x - trunc(x/16)*16`, bit-exact in F32 for 16-bit inputs — exact scaling, exact sub) with mainstream ops only (mul/trunc/sub, all proven headers).
- U16 support gaps closed (all mechanical): `tt_dtype_format` U16=>9 (UInt16), U32=>24, I32=>8; backend `dtype_to_tt_fmt` U16=>3 (was silently defaulting to Float32!) + `tile_bytes_of` U16=>2048 (was defaulting to 4096!); `tt_runtime.cpp` `case 3: DataFormat::UInt16`. Both silent defaults would have corrupted without error.
- `typecast.h` sanctioned pairs used: Float32<->UInt16, Float32<->UInt32, Float16_b<->Float32. BF16 sidecar == Float16_b bits (e8m7), so no landmine; zyx F16 values ride Float16_b through SFPU per the existing workaround (pad/gemm green).
- Reader is DRAM->CB only (`tenstorrent.rs:463` panics on anything else) — reader-side scale replication is impossible. Sidecar streams as full tiles in v1 (test expands dense->[T,1024] host-side); dense+bcast is the scheduled production upgrade (needs new IR ops + audit, deferred deliberately).
- compute `dst_slots` duplication is per STATIC use-site: one inner-loop use = one slot reused across iterations (nested DST inherit, gemm precedent). Binary results alias `slot_x` except shift (fresh slot — the shared u16 tile must survive 4 planes). Static slot budget: U,U2,C0,C16,S,M = 6 of 8 F32 DST slots.
- `verify.rs` rejects F32xBF16 binary (only F16xF32 promotes): sidecars cast to F32 after load.
- F32-out forced: 32-bit DST mode (needed for U16->F32 typecast) auto-enables only with F32 output; fused F32-compute/F16-pack is the documented off-path shape. F32->F16 cast is a later standalone kernel.
- `load_gguf` + safetensors loader gap: loader `todo!()`d Q4_K (added raw-U8 arms for 8/11/12/13/14/20/21/23, sizes verified against llama `static_assert`s AND file byte ratios); safetensors `from_safetensors` lacked "U16" (saver had it).
- C backend `Bitcast` emitted raw `memcpy` sized by storage width — broken whenever F16/BF16 is on either side (their compute regs are `float`). Fixed to round-trip storage bits through `f16tof32`/`f32tof16`/bf16 pair (+ cross-half double-hop). CUDA shares the `memcpy` pattern — unchecked.
- C disk cache (`~/.config/zyx/cache/c`) keys on IR hash and is skipped under `ZYX_DEBUG=16`: a codegen-only fix reuses the stale pre-fix `.so` on normal runs (pass/fail alternation with zero code change). Clear the cache after any codegen fix; the key should arguably include a codegen version.

- The fused mul(F32)+cast+pack kernel is not hardware-impossible: the packer itself supports F32->BF16 early conversion, and DST can hold F32. What blocks it is two software layers: the mode-unaware SFPU typecast op (no DST_ACCUM_MODE term in its addressing) and the JIT's rigid `pack_src` programming (F16 CB -> F16b src under fp32, no override). A direct-RISC emitter recovers the fusion.
- Cost model: every mixed-format boundary forced into ttnn-style decomposition pays a full reader/compute/writer triplet plus DRAM round-trips for intermediates. qwen3.8-27b fuses ~600 launches (CUDA) vs vLLM ~300 (flash attention); on TT the LLKs push it toward 1000+ per forward pass. At least ~130 forced triplets from mixed-format splits alone, likely more.
- Decision (user-approved): split (A) — F32 compute kernels pack F32 (32-bit DST), standalone F32->F16 typecast kernels run 16-bit DST. Dest-acc rule back to output-based (F32 out -> 32-bit), which gives every split kernel the right mode. New builders: `pad_cast_tt` + `pad_mul_tt`; tests go two-stage with F32 tilized intermediates.

## 2026-09-13 — v2 codegen bring-up (pad_move / pad_copy / pad_cast)

- New codegen (`zyx/src/codegen/tenstorrent.rs` `generate_tenstorrent`, v1 saved as untracked `tenstorrent_old.rs`): shared `Compiler` (one `CBId` map, `check_ir_tt` validity gate, per-section `var_map` scalar registers), one emitter per section. Old `Compiler`/backend launch path untouched and re-verified (output-based dest-acc, 64-entry DST-aware `unpack_to_dest`, CB fmt/tb) — all match the proven values above.
- `pad_move_tt` first: dump showed two invalid-but-compiling kernels (host `device.compile` does not JIT C++). (1) Reader/writer DRAM indices formatted the raw `OpId` (`r10`/`r17`, undeclared) instead of the emitted `r{reg}` slot — fixed by resolving every index through the section `var_map` (consts inline, Variable params keep `r{op_id}`, else compilation error). (2) Writer emitted a pre-loop `wait_front` + cached `wbase` (161 waits for 160 pushes) — removed; writer is now per-iteration `wait/write/barrier/pop` exactly like official `write_tile.cpp`. New `check_ir_tt` gates the input IR (2 barriers, reader/writer tile-store shapes, loop/group lengths via `resolve_const`, then balance). Move green: `move bad: 0 / 81920`.
- `pad_copy_tt` compute hung the board twice. Dump matched official `tiles_add.cpp` structurally, but v2 was missing `init_sfpu(in0, out0)` — the only unpack config on non-matmul kernels (`tenstorrent_old.rs:1274-1292`). Unprogrammed unpacker fits the hang (no values, `Finish` never completes). Fix: emit `init_sfpu` (first load CB in, first store CB out) after the CB declarations; empty compute stays silent. Also aligned to v1's proven shape: broad header set (minimal includes are unproven on-device since host compile never JITs) and EndLoop tail `release -> push -> pop` (doc-contract order) instead of per-store `push -> pop -> release`. Copy green: `copy bad: 0 / 81920` bit-exact.
- `pad_cast_tt` (standalone F32->F16, split-A shape): tile `Cast` lowers to the loaded tile's DST slot aliased through, `typecast_tile<0, 5>` (F32->Float16_b, no plain-F16 SFPU kernel in 0.72) with `typecast_tile_init<0, 5>` hoisted pre-loop (per-iteration init reprograms live packer state). Fused mixed-format math stays a loud halt. Compute `Store` accepts any DST-slotted tile value (copy or cast); scalar-math tiles have no slot and fail compilation instead of emitting traffic.
- Standing rules confirmed again: host compile passing proves nothing about device validity (undeclared ids, missing inits all pass); every launch needs the dump-only eye review vs official examples plus the per-CB push/wait/pop table first. Board resets stay manual; after any hung run, no further device access until reset.
- Plain `cargo build -p zyx` does NOT compile the TT codegen (behind the `tenstorrent` feature, off by default). Verify with `cargo build -p zyx --features tenstorrent`; the example test builds caught what the plain build silently skipped.

## 2026-09-13 — v2 refactor: OpEmitter + unified registers + mul

- `OpEmitter` (`zyx/src/codegen/tenstorrent.rs`): owns the kernel ref, `src`, indent/level, section params, the unified register file, and the init sets. Constructed identically in all three generators; `emit_op` is a method on it dispatched on `MemLayout` (scalar in → scalar C++, tiled in → tile op on DST slots). No second machinery: the vars index IS the id for both (`r{id}` scalars, bare-id DST slots), tiled entries never decrement.
- Init discipline: `Set<UOp>` + `Set<BOp>` + typecast pairs fill during the walk; `prepend_compute_inits` inserts the collected `*_init` block at the anchor after `init_sfpu`, ahead of all loops. Copy inits stay per-load.
- `pad_mul_tt` (two F32 tiles → `mul_binary_tile`, result folds onto the lhs slot, `mul_binary_tile_init` hoisted): `pad_input_tt` green, `bad: 0 / 81920` bit-exact over the full mask-mul + cast pipeline. Fused mixed-format and non-Mul tile ALU stay loud halts.

## 2026-09-13 — v2 closed GEMM (lazy matmul, SFPU ban)

- `gemm_qkv_tt` green on first launch: `max_err 0.00817` (< 1e-2 vs torch golden), zero-rows exactly 0 — matches the v1 result. No hang, no reset.
- Model (user-designed): `MatmulTile` emits nothing; it tags the value with its two input CBs. The consuming store does the traffic: acc-fold stores emit `wait/wait/matmul_tiles/pop/pop` (add rides DST), pack stores emit `commit/reserve/wait/pack` + EndLoop push. Fewest locals: matmul values never get C++ names.
- `check_matmul_closed` (IR gate, all `Err`, no asserts): SFPU (`Unary`/`Cast`/`Bitcast`/non-fold `Binary`/`Mad`) sharing a kernel with matmul is rejected up front — `mm_init` + SFPU inits hung the board in v1, so the combination never reaches codegen. Folds must be `add(acc-load, matmul)` stored to their own acc; packs must read a folded chain; matmul sides must be 32x32 F16 CB tiles. Scratch accs exempt from balance.
- Emission notes: `mm_init(in0, in1, out)` once per distinct triple at the anchor (transpose default 0); outermost-only acquire, no hoisted waits; fold CBs + scratch skip blanket pops; no `copy_tile` under `mm_init` (UNPACK-stall wedge guard); no `pack_reconfig` (mm_init owns the packer, official carries none); `mm_out` = last circular store. Reader/writer untouched (seed suppression already existed).
- Dump matched official `matmul_single_core` line-for-line in structure; per-CB push/wait/pop counts exact (51200/51200/320, acc phantom). Sanctioned deviations: tail `release -> push` (v1 order), unused `zero` accessor declared (device JIT has `-Wno-unused-variable`, proven harmless).

## 2026-09-13 — v2 finish: If control flow, pack-scope acquire, Mt=2

- `If`/`EndIf` emit in all three sections (`if (cond) {`, condition = prior boolean scalar, const-inline or register). `loop_level` renamed `scope_level` (ifs and loops share it); acquire/push/pop stay paired with loops only. `Warp` is `unreachable!` (gpu-only). `Index`/`Stack` stay `todo!()`. Closed-mm check rejects `If` in mm compute. No kernel uses `If` yet: arms are compile-verified only.
- Pack-scope acquire replaces outermost acquire: pre-scan collects loop-depths of non-scratch tile stores; acquire/release fire iff the loop depth is a pack depth; multiple depths are a loud `Err` (nested acquires illegal DST); depth-0 packs acquire up front / release at close. Emission byte-identical on all existing kernels (every pack sits at depth 1 = outermost).
- `gemm_tt` generalized to Mt rows (mt/nt/kt, A indexed `mti*Kt + kti`, writer tile-row-major). `gemm_qkv_tt_mt2` (r=64, 16 golden + 48 zero rows) green first launch: `max_err 0.00817`, zero-rows 0 across rows 16..63 — per-output-tile acquire proven (the old code would have acquired once for all 640 tiles). Mt=1 re-verified green on the generalized builder. Counts: cb1/cb2 102400 pushes/pops, cb3 640.

## 2026-09-13 — v2 multi-core: grid query, bounds, 4-core pad green

- First multi-core launch green: `pad_move_tt_mc_run` (160 F32 tiles over 4 cores, 40 contiguous each) `bad: 0 / 81920`, clean `Finish`. All 7 pad tests pass (no single-core regression).
- Grid is queried, not hardcoded: C++ `grid` command (`MeshDevice::compute_with_storage_grid_size`), Rust feeds it into `max_global_work_dims` = [rows, cols]. This board reports 10x11 (2x harvested — hardcoding the unharvested 13 would have launched dead cores). Const over-grid fails at compile (existing `gws_from_kernel` max check); dynamic sizes fail at launch (`TTProgram.max_grid`). `check_ir_tt` allows unresolvable (Variable/symbolic) group lengths; bounds live in the gws/launch path.
- Shard model: `group_range` var reads the trailing per-core coord arg (`arg_pos.len() + axis`, already the emitted shape); shard math (`base = gidx * shard`) is normal builder ops. No builder migration for single-core (`group_range(0, 1)` untouched); no inter-core traffic (per-core L1 CBs + own DRAM tiles by construction).
- Backend doc comment corrected (was 10x12/120 cores).

## 2026-09-13 — v2 symbolic dims: Variable trips green

- `pad_move_tt_sym_run` green first launch: `bad: 0 / 81920` with the 160-trip loop bound as a launch-time Variable. Symbolic bounds + index math proven on-board; mc still green after the rework.
- CUDA model mirrored: Variables are section runtime args (ordinal plumbing already existed), usable in any scalar position through the one register file; grid sizes resolve at launch (GwsDim) with bounds vs max.
- Codegen plugs: `check_ir_tt` Loop arm allows unresolvable trips (negativity only when const); `check_balance` compares per-CB enclosing-trip multisets (const by value, dynamic by op identity) instead of trip products.
- Two fixes from review: `Variable` section arms declared `r{id}` but never registered (later `get_var` failed) — and the declaration naming itself: CUDA keeps params as `p{id}` (pointer) with per-load registers, but TT declares Variables as locals, so they are slot-named `r{slot}` like every other register; the old `r{op_id}` special cases in reader/writer index resolution are deleted (unregistered Variables now fail loudly instead of mis-resolving).

## 2026-09-13 — reduce bring-up: explicit acc + scaler, first hang

- `Op::MatmulTile{x,y}` / `Op::ReduceTile{x}` gained explicit `acc` (SSA threading replaces fold-marker `add`/`max` + fold detection; ~200 lines of detector deleted). `ReduceTile` also gained explicit `scaler` (LLK-mandated ones tile for MAX): check_balance caught the implicit version (pushed-never-popped scaler CB = reader block = wedged board) before silicon did.
- First `tenstorrent_row_max_reduce` launch hung in `run` (no response, board wedged). Traffic was balanced and CB-correspondent; suspects are init/config: our loop runs `init_sfpu` startup + per-iter `copy_tile`s + `reduce_init`, vs ref (`reduce_w_neg`) `compute_kernel_hw_startup` + bare `reduce_init`/`reduce_tile`/`reduce_uninit`.
- Certain bug found in review: missing `reduce_uninit()` before the consuming pack (header: packer keeps reduce edge masks → incorrect packing). Added to the ReduceTile arm (phase bracketing, moreh_softmax precedent).
- Reference findings: `mm_uninit` does not exist anywhere in tt-metal (matmul needs no teardown); fused SFPU+reduce (`moreh_softmax_w`) and SFPU+matmul (`bmm_large_block_zm_fused_bias_activation`) are standard, phased by uninit/reconfig — the coexistence ban stands in for missing phase transitions.
- Process failure: launched without reading the ZYX_DEBUG=16 dump first; ZYX_TT_DUMP_ONLY is inert (nothing reads it) so every run was live. Next: real compile-only dump gate before any new-kernel launch.
- Hang cause (prime suspect): `init_sfpu(icb, ocb)` programs unpack for ONE CB (datacopy, SrcA); the scaler tile arrives via SrcB whose format stays unconfigured → unpack stall on first `reduce_tile`. Reference calls `compute_kernel_hw_startup(input, scaler, acc)` (two-CB unpack). Fix: reduce-only kernels (`phases == {Reduce}`) emit the reference startup with the triple resolved from the first `ReduceTile` (input, scaler, acc); mixed Reduce+Sfpu keeps `init_sfpu` until softmax-fused designs it. `common.h` (already included) pulls the startup header.

## 2026-09-14 — reduce result layout: Row writes row0+col0, Col writes row0

- `REDUCE_ROW` does NOT pack results into the first row. Observed 32x32 tile: live faces 0 and 2 each hold their 16 maxima in row 0 AND column 0 (interior zero); faces 1 and 3 are all zero. All 32 maxima present and correct — math + accumulation proven, pure placement mismatch. The `TileReduceKind::Row` doc ("carried in the result tile's first row") is wrong: a row-0 reader sees faces 0/1 = 16 values + 16 zeros ("bad 16/32"), while column 0 alone holds all 32 maxima top to bottom.
- `REDUCE_COL` is textbook: row 0 across the full 32-wide row (faces 0+1) = 32 column-maxes, rows 1-31 zero. `tenstorrent_row_max_reduce` flipped to Col + column-max expectations went green first launch (`reduce bad: 0/32`; adjacent columns share values — F16 ulp at 16.25 is 2x the 2^-7 step).
- Rule: row-wise consumers read column 0 (or row 0 of even faces only); column-wise consumers read row 0.
- Face-order suspect exonerated: the bit-exact copy test anchors our `0 1 / 2 3` to hardware (packer scans DST in hw order, untilize inverts ours — a mismatch would scramble passthrough, not pass). Matmul passing proves nothing (face relabeling is permutation conjugation, commutes with matmul). Col corroborates: clean faces 0+1 in our frame would be split 0+2 under the swapped order.
- Conclusion: the Row dual-fill is genuine LLK behavior, likely deliberate broadcast for fused-softmax subtract reuse — not our mapping.

## 2026-09-14 — third state machine: `Programmed` phase cursor

- Header inits are "first-call or switch-from-another-op" reconfigurations (`add_tiles_bcast` docs say it outright) — hoist-once is correct only for single-kind kernels. `Programmed` tracks the configured kind along the walk. SUPERSEDED 2026-09-14 (anchor redesign, below): the anchor (first kind) alone hoists; every switch inits inline exactly once; the per-kind seen-sets are deleted.
- Single-kind output proven byte-identical: transpose compute-section diff before/after shows only the 3 new includes, zero init changes. All 8 board tests green.
- Open: per-CB stickiness across alternating copies assumed, unproven; CB-granularity switches inside one kind (two mm pairs alternating) re-init inline — safe direction.

## 2026-09-14 — transpose bring-up: green first launch

- `transpose_wh_init(icb, ocb)` hoisted + `transpose_wh_tile(cb, 0, slot)` under MATH lock: `transpose bad: 0/1024` first board launch, loop-less IR (entry states need no loops — the earlier `wait_front` panic was NOT loop-related).
- Real bug on the way there: the compute `Load` arm treats loads feeding only fused ops as lazy, and `TransposeTile` was missing from that list — the load copied first, then transpose's `wait_front` popped air. Any new fused CB→DST op must join that list.
- Face note: transpose output reads back correctly through our tilize/untilize, further corroborating `0 1 / 2 3`.

## 2026-09-14 — salvaged from v1 (`tenstorrent_old.rs`, dead, uncompiled)

- No hoisted reserves: every tile store reserves its own page per execution. A hoisted reserve pins a page on 1-page CBs and stalls the first iteration forever. V2 obeys (`reserve_back` per store); written down so nobody "optimizes" it later.
- `copy_tile` under `mm_init`'s unpacker config stalls UNPACK (v1 wedge): matmul inputs must be read from CBs directly, never copied first. V2 immune by construction (matmul arm takes CB ids, rejects non-CB loads).
- Explicitly NOT salvaged: the v1 "mixed matmul+SFPU ban" was a misdiagnosis of missing phase transitions. Do not re-add.

## Why we need our own driver (running list)

1. Inter-core traffic does not work on Blackhole (driver, not hardware). All multi-core kernels must be embarrassingly parallel — no core-to-core communication.
2. The JIT programs `pack_src` rigidly from the CB format (F16 CB → F16b src under fp32, no override) and the SFPU typecast op is mode-unaware (no `DST_ACCUM_MODE` term) — the stack, not the silicon, forbids fused mixed-format kernels.
3. `are_packers_configured_correctly` is blind to Read_32b/Dstacc/strides/relu/threshold/L1acc; the 2-arg `pack_reconfig_data_format` silently skips src-only changes; the runtime DST toggle RMWs only Read_32b, leaving formats stale. Asserts that can't see the state they claim to check.
4. Docs describe APIs the pinned SDK doesn't have: latest docs say `transpose_tile`/`transpose_init` (and the split `matmul_init`); 0.72 has neither — the only working spellings are marked deprecated. Every new op starts with version archaeology in the installed headers because the docs can't be trusted.
5. Result geometry is unspecified: no header or doc says `REDUCE_ROW` lands in row 0 *and* column 0 of the even faces — found on silicon with a full-tile print. A contract that doesn't state where results land isn't a contract; board-discovery per op is the tax.
6. `log_tile` is ln, not log2: the base arrives via a separate `log_with_base_tile(idst, base_scale)` taking the bits of 1/ln(base) (`0x3fb8aa3b` for log2). The name says nothing; our `Log2 → log_tile` mapping would have computed ln silently. Caught in review, never ran. Fallout: `UOp::Ln` deleted entirely (`ln = log2·ln2` at the tensor/custom/autograd construction points, lowering passes removed) — it existed only for TT and was pure surface area.
7. Vendor LLK computes wrong results: `sin_tile` (+init, verified in dump, init sane on read) returns inputs unchanged on this board/SDK; a second `exp` per episode zeroes or constant-garbages the tile. Our driver can't fix their microcode — but it must verify every op's MATH on silicon (no op is innocent until a board test says so) and be able to ship our own implementations (LUT/polynomial trig, phase-split cones) through custom kernels + `k.asm` instead of trusting vendor LLK. The per-op board matrix above is the start of that ledger.

## 2026-09-14 — anchor redesign + the second-exp silicon rule

- Cursor v1 hoisted ALL seen kinds and inlined every switch, so mixed cones got each later init twice (hoisted + inline). Mixed `exp→add` came back all-zeros; the duplicate looked guilty. Redesign (user's call): the seen-sets are deleted, replaced by `anchor: Programmed` (starts `Entry`) beside `cursor`. First transition records the anchor and trusts the hoist; repeats are free; every switch inlines exactly once, including switch-backs (A→B→A re-inits — no stale-hoist hole). `cursor` stays `Entry` iff a single kind appears. Prepend renders the hoist straight from the anchor value (it carries its params); startup emission re-keyed from "any matmul" to "anchor is matmul". Single-kind dumps byte-identical; mixed dumps match the official `sfpu_eltwise_chain` shape line for line.
- The duplicate was NOT the cause: anchor-clean mixed still zeros. Bisect series (one board launch each, all finishing clean — no wedge):
  - `exp,exp` same slot → zeros. `exp,exp,add` → zeros. `exp,neg,exp` → CONSTANT 24240 for all inputs.
  - `exp` alone (loop ×4, one/episode) → green. `add` alone → green. `exp,neg` → green. `neg,neg` → green. `exp→pack→reload→reduce` (two episodes) → green 0/32.
- Rule: at most ONE `exp` per acquire→commit episode. Every green case has ≤1 exp; every red case has 2. Distinct-op chains are fine; the second `exp` poisons the episode (zeros, or deterministic garbage with a mid re-init). `init_sfpu` tested instead-of-startup (chain-exact order) and after per-op inits — no effect either way. Mechanism unknown (SFPU microcode state consumed without restore? sequencer hazard?); loop-iteration traffic (commit/pack/re-acquire) re-arms whatever it is, since 4 looped exps share one hoisted init and pass.
- Consequence: single-episode multi-exp cones (`exp(a)+exp(b)`, `exp(exp(x))`) cannot go green on this silicon/SDK — restructure at CB boundaries (the exp_reduce two-phase shape) or keep one exp per episode. Model patterns mostly comply already (softmax loops rows, SiLU/GELU carry one exp); the failing shapes were test constructs. Fair cursor proof rewritten as `exp(a)+b` (one exp, genuine Unary→Binary switch).
- `k.asm` now works in compute: DCE roots it (opaque side effect), section splitter keeps it lexical, the arm substitutes `{i}` with CB indices (Circular Storage) or DST slots (live tiles), everything else errors loudly. Scheduler lesson: `instruction_schedule` floated the side-effect-free-looking Asm across a barrier into the reader (operands are top-defined Storages, no edges) — Asm is now structural (never crosses barriers/scopes/stores) with load-vs-asm order preservation. Any reorderable-but-side-effecting op needs the same treatment.
- `ZYX_TT_INIT_SFPU` experiment hook (env-gated `init_sfpu` instead of startup in prepend): decided the init question with zero code churn. No effect either way — removed if unused... (still in code at time of writing; delete once the second-exp rule below is confirmed final).

## 2026-09-14 — silicon rules, refined: second-exp poison + dead trig

- Bisect matrix, one board launch each, every launch finishing clean (no wedge anywhere in this series — the earlier wedge was the pre-anchor duplicate era):
  - GREEN: `exp` alone (loop ×4), `add` alone, `add,add`, `neg,neg`, `exp,neg`, `neg,exp`, `exp(a)+b` (fair Unary→Binary switch, `mixed bad: 0/1024`), `exp→pack→reload→reduce` (`0/32`).
  - RED: `exp,exp` → all zeros. `exp,neg,exp` → CONSTANT 24240 for all inputs. `sin` alone → identity passthrough (`z[10] = 0.625` for expected `0.585`; 704/1024 bad, all identity-form).
- Refined rule: at most ONE `exp` per acquire→commit episode. It is not "second op" (add/neg repeat and chain freely) and not "any microcoded op twice" (only exp proven so far — sin never gets that far). Loop-iteration traffic (commit/pack/re-acquire) re-arms whatever the second exp consumes: 4 looped exps share one hoisted init and pass. `init_sfpu` tested instead-of-startup (chain-exact order) and after per-op inits: no effect. Mechanism unknown.
- `sin_tile` is broken singly: textbook source (`sin_tile_init()` + `sin_tile(0)`, both present in `trigonometry.h`) returns inputs unchanged. LLK read (`ckernel_sfpu_trigonometry.h`, BH+WH copies): `sine_init` loads Cody-Waite P2/P3 + 1/π into `vConstFloatPrgm0/1/2`, `calculate_sine` does range reduction + odd-poly — sane on paper, identity on silicon. Below our codegen layer: SFPU trig microcode (or its 0.72 programming) doesn't compute on this board. `cos` shares the family/file, guilty by association, untested. Consequence: RoPE needs sin+cos per position — plan LUT/polynomial trig or a firmware retest; do not trust vendor trig.
- Practical fallout: single-episode multi-exp cones (`exp(a)+exp(b)`, `exp(exp(x))`) cannot go green here — split at CB boundaries (the exp_reduce two-phase shape) or keep one exp per episode. Real-model patterns mostly comply (softmax loops rows, SiLU/GELU carry one exp); the failing shapes were test constructs. Probes for answered questions were removed; the suite keeps one fair test per proven shape.
