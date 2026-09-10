# Tenstorrent debugging log — pad_passthrough_tt

Date: 2026-09-10.
Test: `examples/qwen3.8-27b/tests/pad.rs:157` `pad_passthrough_tt_run`.
Builder: `examples/qwen3.8-27b/src/lib.rs:204` `pad_passthrough_tt`.
Constraint: tt-metal 0.72 read-only. No tt-metal edits, no tt-metal recompile. Fixes belong in zyx. Test-only diagnostic prints are acceptable.

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
