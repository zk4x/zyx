//! Configuring the unpacker and the packer for a single flat run of FP32 datums.
//!
//! Shared by every datapath gate, because the configuration is long, fiddly and
//! entirely determined by the functional models -- so a second copy would be a
//! second thing to keep in step with `UNPACR_Regular.md` and the `Packers/` pages.
//!
//! # What this deliberately does not do
//!
//! It configures **one flat run**, not a tiled 32x32 image: `XDim` datums in one
//! row of one plane. That is enough to exercise the datapath and the numerics, and
//! it keeps the address arithmetic small enough to check against the specification
//! by hand. Driving a real `tt_layout::Layout` through it is Phase 6 work, and it
//! is where the `Z`-plane-to-face convention finally gets settled.
//!
//! Everything from `WormholeB0/.../Packers/` is `UNVERIFIED`: that directory has no
//! Blackhole counterpart. Blackhole's own `PACR.md` is self-labelled "basic".

use tt_isa::backend::ConfigWords;
use tt_isa::backend::ThreadConfigEntry;
use tt_isa::cfg::generated::{alu, global, pack0, thcon, thread, unpack1};
use tt_isa::isa::Instruction;
use tt_isa::isa::generated::{defs, encode};
use tt_isa::tile::{L1Format, TileDescriptor, TileImage};

/// Where the input tile image is staged, and where the packer writes.
///
/// Both clear of the firmware at `0x6000` and the mailbox at `0x10_0000`, and
/// 16-byte aligned: the packer writes L1 in 16-byte units (`PACR.md`), and a tile
/// base must be 16-byte aligned anyway (`UNPACR_Regular.md:113`).
pub const STAGE: u64 = 0x2_0000;
pub const OUT: u64 = 0x3_0000;

pub const SCRATCH_GPR: u32 = 8;
/// See `probe_unpack.rs`: the smallest `REG5_Dest_cntx0_address` the unpacker
/// accepts, which is `Dst` row 0.
pub const DST_BASE: u32 = 64;

pub const UNPACR_LAST: u32 = 1;

pub fn flat_descriptor(datums: u32) -> TileDescriptor {
    TileDescriptor::zeroed()
        .with_x_dim(datums)
        .with_y_dim(1)
        // Zero means one (`UNPACR_Regular.md:70-71`), and it keeps words 2 and 3 of
        // the descriptor zero -- which ttsim will not let us write at all.
        .with_z_dim(0)
        .with_w_dim(0)
        .with_is_uncompressed(true)
        .with_in_data_format_raw(L1Format::Fp32.code().expect("FP32's code is measured"))
}

pub fn staged_image(descriptor: TileDescriptor, datums: u32) -> Vec<u8> {
    let image = TileImage::new(descriptor, L1Format::Fp32).unwrap();
    let mut staged = vec![0u8; image.total_bytes()];
    for i in 0..datums as usize {
        let v = 1.0f32 + i as f32;
        let off = image.datum_bit_offset(i) / 8;
        staged[off..off + 4].copy_from_slice(&v.to_bits().to_le_bytes());
    }
    staged
}

pub fn thread_config() -> Vec<Instruction> {
    vec![
        // `SETC16.md`: after coming out of reset this must be executed for some
        // value before anything else relies on the configuration bank.
        ThreadConfigEntry::zeroed(thread::CFG_STATE_ID_StateID.addr32())
            .set(thread::CFG_STATE_ID_StateID, 0)
            .unwrap()
            .encode()
            .unwrap(),
        // `UnpackToDst` with this set is `UndefinedBehavior` (`UNPACR_Regular.md:313`).
        ThreadConfigEntry::zeroed(thread::SRCA_SET_SetOvrdWithAddr.addr32())
            .set(thread::SRCA_SET_SetOvrdWithAddr, 0)
            .unwrap()
            .encode()
            .unwrap(),
        ThreadConfigEntry::zeroed(thread::UNPACK_MISC_CFG_CfgContextOffset_0.addr32())
            .set(thread::UNPACK_MISC_CFG_CfgContextOffset_0, 0)
            .unwrap()
            .encode()
            .unwrap(),
    ]
}

/// `REG3_Base_address` for a tile image staged at `l1_base`.
///
/// `InAddr = (Base_address + 1 + DigestSize) * 16` (`UNPACR_Regular.md:112-115`):
/// the `+ 1` is the unpacker stepping over the tile header, so `Base_address`
/// names the *start of the image, header included* -- which is where
/// [`TileImage`] puts the header. This was once `l1_base / 16 - 1`, subtracting
/// the header a second time: the unpacker then read the header's 16 zero bytes
/// as the first datums and dropped as many from the end. That is the whole of
/// what divergence rows 30 and 35 recorded as a "hidden output base" scaling
/// with the input width -- ttsim, silicon and the specification all agreed, and
/// the error was here.
pub fn tile_base_units(l1_base: u64) -> u32 {
    (l1_base / TileImage::ALIGNMENT as u64) as u32
}

/// Unpacker configuration, as established by `probe_unpack.rs`.
pub fn unpack_config(words: &mut ConfigWords, descriptor: TileDescriptor, l1_base: u64) {
    let base_units = tile_base_units(l1_base);
    words
        .set(thcon::THCON_SEC0_REG3_Base_address, base_units)
        .unwrap()
        .set(thcon::THCON_SEC0_REG7_Offset_address, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, descriptor.x_dim())
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Out_data_format, L1Format::Fp32.code().unwrap())
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Ovrd_data_format, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0, 1)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0, 1)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Haloize_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Tileize_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Throttle_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Upsample_rate, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Upsample_and_interleave, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Shift_amount_cntx0, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Force_shared_exp, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG5_Dest_cntx0_address, DST_BASE)
        .unwrap()
        .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 0)
        .unwrap()
        .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
        .unwrap();

    // Words 0 and 1 always -- `Config` outlives the kernel that wrote it, and a
    // matmul's `ZDim` of 4 in word 1 left under a flat run's zero changes what
    // the unpacker reads -- and 2 and 3 only if they carry something: ttsim
    // models words 0 and 1 of the span and refuses 2 and 3 (divergence row 29).
    for (i, word) in descriptor.words().iter().enumerate() {
        if i < 2 || *word != 0 {
            words.seed(thcon::THCON_SEC0_REG0_TileDescriptor.addr32() + i as u16, *word).unwrap();
        }
    }
}

/// Which unpacker: 0 feeds `SrcA` (or `Dst`), 1 feeds `SrcB` only.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Unpacker {
    SrcA = 0,
    SrcB = 1,
}

/// The `ThreadConfig` the `Src` path needs, in place of [`thread_config`].
///
/// `SRCA_SET_SetOvrdWithAddr` is the reverse of the `Dst` path's setting: with it
/// clear, a `Src` unpack is `UnsupportedFunctionality` (`UNPACR_Regular.md:316`),
/// and ttsim says so by name (`!unpack_to_dst: SRCA_SET_SetOvrdWithAddr=0`). It
/// shares entry 5 with `SRCA_SET_Base`, which stays zero.
pub fn src_thread_config() -> Vec<Instruction> {
    vec![
        ThreadConfigEntry::zeroed(thread::CFG_STATE_ID_StateID.addr32())
            .set(thread::CFG_STATE_ID_StateID, 0)
            .unwrap()
            .encode()
            .unwrap(),
        ThreadConfigEntry::zeroed(thread::SRCA_SET_SetOvrdWithAddr.addr32())
            .set(thread::SRCA_SET_SetOvrdWithAddr, 1)
            .unwrap()
            .encode()
            .unwrap(),
        ThreadConfigEntry::zeroed(thread::UNPACK_MISC_CFG_CfgContextOffset_0.addr32())
            .set(thread::UNPACK_MISC_CFG_CfgContextOffset_0, 0)
            .unwrap()
            .encode()
            .unwrap(),
    ]
}

/// Configure `unpacker` to move FP32 datums from `l1_base` into `SrcA`/`SrcB`,
/// converting to `out` (`UNPACR_Regular.md:495-520`: FP32 in, TF32 or a 16-bit
/// type out; FP32 *into* `Src` is `UndefinedBehavior`).
///
/// Unpacker 1 is configured through far fewer fields than unpacker 0, because ttsim
/// refuses most of `THCON_SEC1_REG5`/`REG7` and `UNP1_ADDR_*` outright
/// (`probe_src::map_the_src_path_configuration_surface`). None of the refused ones
/// is read on this path: unpacker 1 takes `XDim` from the descriptor rather than
/// from `Tile_x_dim_cntx` (`UNPACR_Regular.md:73`), has no `Dest_cntx` term, and a
/// single flat row needs no `Ystride`. `REG7_Offset_address` *is* read and cannot
/// be written, so this relies on its reset value being zero.
pub fn unpack_src_config(words: &mut ConfigWords, unpacker: Unpacker, descriptor: TileDescriptor, l1_base: u64, out: u32) {
    let base_units = tile_base_units(l1_base);
    let descriptor_span = match unpacker {
        Unpacker::SrcA => {
            words
                .set(thcon::THCON_SEC0_REG3_Base_address, base_units)
                .unwrap()
                .set(thcon::THCON_SEC0_REG7_Offset_address, 0)
                .unwrap()
                .set(thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, descriptor.x_dim())
                .unwrap()
                .set(thcon::THCON_SEC0_REG2_Out_data_format, out)
                .unwrap()
                // `Src`, not `Dst`.
                .set(thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0, 0)
                .unwrap()
                .set(thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0, 1)
                .unwrap()
                // With `UnpackToDst` clear this *replaces* the output address rather
                // than adding to it (`UNPACR_Regular.md:265-270`); `SrcA` row 0 is
                // `OutAddr / 16 - 4`, so 64 is row 0.
                .set(thcon::THCON_SEC0_REG5_Dest_cntx0_address, DST_BASE)
                .unwrap()
                .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 0)
                .unwrap()
                .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
                .unwrap();
            thcon::THCON_SEC0_REG0_TileDescriptor
        }
        Unpacker::SrcB => {
            words
                .set(thcon::THCON_SEC1_REG3_Base_address, base_units)
                .unwrap()
                .set(thcon::THCON_SEC1_REG2_Out_data_format, out)
                .unwrap()
                .set(thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx0, 1)
                .unwrap()
                .set(unpack1::UNP1_ADDR_CTRL_ZW_REG_1_Zstride, 0)
                .unwrap();
            thcon::THCON_SEC1_REG0_TileDescriptor
        }
    };
    // Every other field of the words touched above is zero by construction, which
    // is what `Ovrd_data_format`, `Haloize_mode`, `Tileize_mode`, `Throttle_mode`,
    // upsampling, `Shift_amount` and `Unpack_Src_Reg_Set_Upd` must be here.

    // Only the non-zero descriptor words: row 29.
    for (i, word) in descriptor.words().iter().enumerate() {
        if *word != 0 {
            words.seed(descriptor_span.addr32() + i as u16, *word).unwrap();
        }
    }
}

/// `SETADCXX` for either unpacker's channels.
pub fn set_adc_x(unpacker: Unpacker, first: u32, last: u32) -> Instruction {
    let adc = encode::Setadcxx::ZERO.x0_val(first).x1_val(last);
    match unpacker {
        Unpacker::SrcA => adc.u0(1),
        Unpacker::SrcB => adc.u1(1),
    }
    .encode()
    .unwrap()
}

/// `UNPACR` into `SrcA`/`SrcB`, handing the bank to the Matrix Unit if `flip`.
pub fn unpack_src_instruction(unpacker: Unpacker, flip: bool) -> Instruction {
    let base = encode::UnpacrRegular::ZERO
        .which_unpacker(unpacker as u32)
        .multi_context_mode(1)
        .flip_src(u32::from(flip))
        .encode()
        .unwrap();
    Instruction::new(base.word() | UNPACR_LAST, &defs::UNPACR_Regular)
}

/// Packer configuration: read `Dst` as 32-bit data, write FP32 to `l1_dest`.
pub fn pack_config(words: &mut ConfigWords, l1_dest: u64) {
    // `Packers/OutputAddressGenerator.md`: `Addr = L1_Dest_addr +
    // !Sub_l1_tile_header_size`, and on Blackhole the `YZW_Addr` term is in bytes
    // and gets `>>= 4`, so the whole address is in 16-byte units.
    // The address is `L1_Dest_addr + !Sub_l1_tile_header_size`, so with that field
    // at zero the packer always skips one 16-byte unit. ttsim refuses to let it be
    // written -- `tensix_cfg_wr32: THCON_SEC0_REG1_Sub_l1_tile_header_size`, a
    // named field refusal rather than the `reg=N` the unpacker path gives -- so the
    // `+1` is unavoidable here and gets compensated in the base instead.
    let dest_units = l1_dest / 16 - 1;
    words
        .set(thcon::THCON_SEC0_REG1_L1_Dest_addr, dest_units as u32)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_In_data_format, L1Format::Fp32.code().unwrap())
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Out_data_format, L1Format::Fp32.code().unwrap())
        .unwrap()
        // Input comes from `Dst`, not from L1 (`Packers/InputAddressGenerator.md`:
        // `Source_interface_selection == 1` is the L1 path).
        .set(thcon::THCON_SEC0_REG1_Source_interface_selection, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Disable_zero_compress, 1)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Add_l1_dest_addr_offset, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Enable_out_fifo, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Exp_section_size, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Row_start_section_size, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Downsample_rate, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Downsample_mask, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Exp_threshold_en, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Pack_L1_Acc, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_pack_start_intf_pos, 0)
        .unwrap();

    // The output address generator's Y/Z/W term, all zero for one contiguous run.
    words
        .set(pack0::PCK0_ADDR_BASE_REG_1_Base, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_XY_REG_1_Ystride, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
        .unwrap()
        // The input (`Dst`) side's strides. `Ystride` is in bytes and divided
        // by the datum size (`Packers/InputAddressGenerator.md`), so one FP32
        // `Dst` row is 64: ADC Y then counts rows, which is how [`pack_rows`]
        // walks `Dst` a group of four at a time. With Y at zero -- every single
        // `PACR` here -- it contributes nothing. `PCK0_ADDR_BASE_REG_0_Base`
        // is left at its reset zero rather than written: it is register 16,
        // which ttsim does not model (`tensix_cfg_wr32: reg=16`, divergence
        // row 28), and the silicon per-thread reset zeroes all of `Config`.
        .set(pack0::PCK0_ADDR_CTRL_XY_REG_0_Xstride, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_XY_REG_0_Ystride, DST_ROW_BYTES)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_ZW_REG_0_Zstride, 0)
        .unwrap()
        // 32-bit reads out of `Dst`, matching the FP32 format everywhere else.
        .set(pack0::PCK_DEST_RD_CTRL_Read_32b_data, 1)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_unsigned, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_int8, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Round_10b_mant, 0)
        .unwrap()
        // `PACR.md`: `TilePosition` returns to zero only when advancing it lands
        // exactly on `pack_reads_per_xy_plane`, so it must match the number of read
        // interfaces each `PACR` enables.
        //
        // Only this one of the three counters is set. ttsim refuses a non-zero
        // `pack_per_xy_plane` or `pack_xys_per_tile` at *configuration* time
        // (`tensix_cfg_wr32: PACK_COUNTERS_SEC0_pack_per_xy_plane` -- a
        // value-dependent refusal on a register whose zero write it accepts), and
        // refuses a *zero* `pack_reads_per_xy_plane` at *execution* time
        // (`tensix_pacr: pack_reads_per_xy_plane=0`). The intersection is exactly
        // this.
        .set(pack0::PACK_COUNTERS_SEC0_pack_reads_per_xy_plane, 4)
        .unwrap()
        // The `Dst` read offset, added by the input address generator.
        .set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, 0)
        .unwrap()
        // The packer edge-masking path, off.
        .set(pack0::PCK_EDGE_MODE_mode, 0)
        .unwrap()
        .set(pack0::PCK_EDGE_OFFSET_SEC0_mask, 0xFFFF)
        .unwrap();

    // `Dst` is read as FP32 by the packer too; `ALU_ACC_CTRL_Fp32_enabled` is what
    // tells the rest of the backend that.
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
}

pub fn set_adc_x_unpack(first: u32, last: u32) -> Instruction {
    encode::Setadcxx::ZERO.u0(1).x0_val(first).x1_val(last).encode().unwrap()
}

/// `SETADCXX` for the packer channels (`PK`), which set the `Dst` datum range.
pub fn set_adc_x_pack(first: u32, last: u32) -> Instruction {
    encode::Setadcxx::ZERO.pk(1).x0_val(first).x1_val(last).encode().unwrap()
}

pub fn unpack_instruction() -> Instruction {
    // Bit 0 is undocumented and ttsim refuses the instruction without it; see
    // `probe_unpack::unpacr_without_the_undocumented_last_bit_is_refused`.
    let base = encode::UnpacrRegular::ZERO.which_unpacker(0).multi_context_mode(1).encode().unwrap();
    Instruction::new(base.word() | UNPACR_LAST, &defs::UNPACR_Regular)
}

/// Bytes in one row of FP32 `Dst`: the packer's input `Ystride` per row.
pub const DST_ROW_BYTES: u32 = 16 * 4;

/// The pack `AddrMod` entry [`pack_rows`] uses to step ADC Y by four rows.
pub const PACK_ADDR_MOD_NEXT_GROUP: u32 = 1;

/// Pack `rows` rows of FP32 `Dst`, from row 0, to the L1 run [`pack_config`]
/// set up, contiguously.
///
/// One `PACR` reads one aligned group of four rows (`PACR.md`), so this issues
/// one per group, with `AddrMod` stepping the input ADC's Y by four between
/// them (`Packers/InputAddressGenerator.md`) and `Last` only on the final one:
/// without `Last` the output address generator keeps appending to the same run
/// rather than starting at `L1_Dest_addr` again (`OutputAddressGenerator.md`).
/// A final partial group gets the mask `(1 << remaining) - 1`, as `PACR.md`
/// recommends. Runs on the pack thread, whose `ThreadConfig` it sets.
pub fn pack_rows(rows: u32) -> Vec<Instruction> {
    super::loops::Item::unrolled(&pack_rows_items(rows))
}

/// [`pack_rows`] as loop items: every group's `PACR` but the last is the same
/// word, so the groups are one `Repeat` (`super::loops`), which a `MOP` takes
/// in two words.
pub fn pack_rows_items(rows: u32) -> Vec<super::loops::Item> {
    use super::loops::Item;
    assert!(rows > 0, "nothing to pack");
    let groups = rows.div_ceil(4);
    let mut p: Vec<Item> = [
        state_id(),
        thread_entry(thread::ADDR_MOD_PACK_SEC0_YsrcIncr, 0),
        thread_entry(thread::ADDR_MOD_PACK_SEC1_YsrcIncr, 4),
        set_adc_x_pack(0, 15),
        // From row 0, whatever an earlier `pack_rows` on this thread left Y at.
        encode::Setadcxy::ZERO.pk(1).y0(1).y0_val(0).encode().unwrap(),
    ]
    .into_iter()
    .map(Item::I)
    .collect();
    let pacr = |mask: u32, last: bool| {
        encode::Pacr::ZERO
            .read_intf_sel(mask)
            .addr_mod(if last { 0 } else { PACK_ADDR_MOD_NEXT_GROUP })
            .last(u32::from(last))
            .encode()
            .unwrap()
    };
    // Every group but the last is whole (rows are packed from row 0).
    if groups > 1 {
        p.push(Item::Repeat { times: groups - 1, body: vec![Item::I(pacr(0b1111, false))] });
    }
    let remaining = rows - 4 * (groups - 1);
    let mask = if remaining >= 4 { 0b1111 } else { (1 << remaining) - 1 };
    p.push(Item::I(pacr(mask, true)));
    p
}

/// One `PACR` enabling `read_intf_sel` read interfaces.
pub fn pack_instruction(read_intf_sel: u32, last: bool) -> Instruction {
    encode::Pacr::ZERO.read_intf_sel(read_intf_sel).last(u32::from(last)).encode().unwrap()
}

/// Release every Tensix semaphore: `SEMINIT` each to `Value` 1, `Max` 2.
///
/// A thread blocked in `SEMWAIT` survives the backend soft-reset pulse, and
/// every later program on that thread queues behind it -- `MEASURED` on silicon,
/// where a kernel whose math role waited on a semaphore nothing posted left the
/// tile wedged across processes (divergence row 65). One `SEMWAIT` waits while a
/// semaphore is zero, the other while it is at its max; 1 of 2 satisfies
/// neither, so whatever is blocked runs on. Pushed from thread 0, which the
/// reset runs first. Every kernel initialises the semaphores it uses in its own
/// setup, so nothing depends on these values.
pub fn release_semaphores() -> Vec<Instruction> {
    (0..8)
        .map(|i| {
            let sem = tt_isa::sync::Semaphore::new(i).expect("eight semaphores");
            tt_isa::sync::init(sem, 1, 2).expect("four-bit fields")
        })
        .collect()
}

/// Hand both banks of `SrcA` and of `SrcB` to the Matrix Unit, by four plain
/// `UNPACR`s (the matmul's own encoding, `unpack_src_instruction`) of whatever
/// is at [`STAGE`]: the wedged-tile recovery (`session::unwedge_tile`, X5b).
///
/// A Matrix Unit instruction that reads `Src` waits, by itself, until its bank's
/// `AllowedClient` is the Matrix Unit (`STALLWAIT.md`, C7/C8), and the backend
/// pulse hands every bank back to the unpackers (`SoftReset.md`, bits 15-16):
/// one caught waiting by the pulse waits for good, and its thread takes nothing
/// more. These unpacks give it its banks. Two per unpacker, one into each bank:
/// the pulse leaves both banks the unpackers' and each unpacker on bank 0, so
/// none of them waits, whatever the stuck thread does with what it is given.
/// What lands in `Src` is not data anyone reads. The pulse after it gives the
/// banks back.
pub fn src_feeder() -> Vec<Instruction> {
    const DATUMS: u32 = 16;
    let descriptor = flat_descriptor(DATUMS);
    let tf32 = L1Format::Tf32.code().expect("TF32 has a code");
    let mut p = src_thread_config();
    let mut words = ConfigWords::new();
    unpack_src_config(&mut words, Unpacker::SrcA, descriptor, STAGE, tf32);
    unpack_src_config(&mut words, Unpacker::SrcB, descriptor, STAGE, tf32);
    let mut buf = [Instruction::new(0, &defs::NOP); 96];
    let n = words.program(SCRATCH_GPR, &mut buf).expect("two unpackers' configuration fits");
    p.extend_from_slice(&buf[..n]);
    for u in [Unpacker::SrcA, Unpacker::SrcB] {
        for _bank in 0..2 {
            p.push(set_adc_x(u, 0, DATUMS - 1));
            p.push(unpack_src_instruction(u, true));
        }
    }
    p.push(tt_isa::backend::wait_for_unpacker0(tt_isa::backend::Before::EVERYTHING).unwrap());
    p.push(tt_isa::backend::wait_for_unpacker1(tt_isa::backend::Before::EVERYTHING).unwrap());
    p
}

/// Put the tile's configuration and the issuing thread's own Tensix state back to
/// what ttsim starts with: `Config` below the global block zero, every
/// `ThreadConfig` entry zero, every RWC zero, every ADC counter zero.
///
/// None of it is touched by anything the host can do from outside -- the
/// backend soft-reset pulse resets the units, not the per-thread state -- so on
/// silicon one gate inherits the last one's. The first silicon run of the `Src`
/// probes after `step9_matmul` had configured thread 0's address modifiers and
/// RWCs moved nothing into the rows they dumped. ttsim starts every run from
/// zero, which is what every gate is written against.
///
/// **Silicon only.** ttsim refuses `SETC16` to some entries at any value
/// (`SRCB_SET_Base`, divergence row 36), and has nothing to reset.
pub fn thread_state_reset() -> Vec<Instruction> {
    let entries: std::collections::BTreeSet<u16> =
        tt_isa::cfg::generated::ALL_THREAD_CONFIG_FIELDS.iter().map(|(_, f)| f.addr32()).collect();
    // `Config` first: it is shared, and tt-metal (or an earlier gate) may have
    // left any of it set (`backend::reset_config`).
    let mut p: Vec<Instruction> = tt_isa::backend::reset_config(SCRATCH_GPR).unwrap().to_vec();
    p.extend(entries.into_iter().map(|addr32| ThreadConfigEntry::zeroed(addr32).encode().unwrap()));
    // RWCs: set each counter (and its carry register) to zero, and the fidelity
    // phase; no bank flips (`SETRWC.md`).
    p.push(encode::Setrwc::ZERO.src_a(1).src_b(1).dst(1).fidelity(1).encode().unwrap());
    // ADCs: X, Y, Z, W of both channels, for both unpackers and the packers.
    p.push(encode::Setadcxy::ZERO.u0(1).u1(1).pk(1).x0(1).y0(1).x1(1).y1(1).encode().unwrap());
    p.push(encode::Setadczw::ZERO.u0(1).u1(1).pk(1).z0(1).w0(1).z1(1).w1(1).encode().unwrap());
    p
}

/// The first `ThreadConfig` write any thread must make (`SETC16.md`): its
/// configuration state ID. Every role program starts with it, because each
/// thread has its own.
pub fn state_id() -> Instruction {
    ThreadConfigEntry::zeroed(thread::CFG_STATE_ID_StateID.addr32())
        .set(thread::CFG_STATE_ID_StateID, 0)
        .unwrap()
        .encode()
        .unwrap()
}

/// The `ThreadConfig` increments of address modifier `entry` (0..8): what an
/// instruction's `AddrMod` selects. Blackhole has eight entries and a three-bit
/// `AddrMod` to reach them (divergence row 42).
pub struct AddrModEntry {
    pub src_a_incr: tt_isa::cfg::ThreadConfigField,
    pub src_b_incr: tt_isa::cfg::ThreadConfigField,
    pub dst_incr: tt_isa::cfg::ThreadConfigField,
}

pub fn addr_mod_entry(entry: usize) -> AddrModEntry {
    use thread::*;
    let src_a = [
        ADDR_MOD_AB_SEC0_SrcAIncr,
        ADDR_MOD_AB_SEC1_SrcAIncr,
        ADDR_MOD_AB_SEC2_SrcAIncr,
        ADDR_MOD_AB_SEC3_SrcAIncr,
        ADDR_MOD_AB_SEC4_SrcAIncr,
        ADDR_MOD_AB_SEC5_SrcAIncr,
        ADDR_MOD_AB_SEC6_SrcAIncr,
        ADDR_MOD_AB_SEC7_SrcAIncr,
    ];
    let src_b = [
        ADDR_MOD_AB_SEC0_SrcBIncr,
        ADDR_MOD_AB_SEC1_SrcBIncr,
        ADDR_MOD_AB_SEC2_SrcBIncr,
        ADDR_MOD_AB_SEC3_SrcBIncr,
        ADDR_MOD_AB_SEC4_SrcBIncr,
        ADDR_MOD_AB_SEC5_SrcBIncr,
        ADDR_MOD_AB_SEC6_SrcBIncr,
        ADDR_MOD_AB_SEC7_SrcBIncr,
    ];
    let dst = [
        ADDR_MOD_DST_SEC0_DestIncr,
        ADDR_MOD_DST_SEC1_DestIncr,
        ADDR_MOD_DST_SEC2_DestIncr,
        ADDR_MOD_DST_SEC3_DestIncr,
        ADDR_MOD_DST_SEC4_DestIncr,
        ADDR_MOD_DST_SEC5_DestIncr,
        ADDR_MOD_DST_SEC6_DestIncr,
        ADDR_MOD_DST_SEC7_DestIncr,
    ];
    AddrModEntry { src_a_incr: src_a[entry], src_b_incr: src_b[entry], dst_incr: dst[entry] }
}

/// A `SETC16` giving one `ThreadConfig` field `value`, every other field of its
/// word zero.
pub fn thread_entry(field: tt_isa::cfg::ThreadConfigField, value: u16) -> Instruction {
    ThreadConfigEntry::zeroed(field.addr32()).set(field, value).unwrap().encode().unwrap()
}

/// The instructions that write `words` into `Config`, sized to fit.
pub fn config_program(words: &ConfigWords) -> Vec<Instruction> {
    let mut buf = vec![tt_isa::sfpu::nop(); words.program_len()];
    let n = words.program(SCRATCH_GPR, &mut buf).unwrap();
    buf.truncate(n);
    buf
}

/// The three role programs of an L1 -> `Dst` -> L1 round trip, with `kernel`
/// running on the math thread between the unpack and the pack.
///
/// Split the way LLK splits it (`harness::Roles`): thread 0 configures both
/// halves of the datapath and unpacks `datums` of `descriptor` from [`STAGE`]
/// into `Dst`; thread 1 runs `kernel`; thread 2 packs four `Dst` rows through
/// the read interfaces in `read_intf_sel` to [`OUT`]. Each role ends by waiting
/// for the unit it drove, so its work is complete when its firmware reports
/// `DONE` and the next role starts.
pub fn dst_round_trip_roles(
    descriptor: TileDescriptor,
    datums: u32,
    kernel: &[Instruction],
    read_intf_sel: u32,
) -> [Vec<Instruction>; 3] {
    use tt_isa::backend::{self, Before};

    let mut unpack = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, descriptor, STAGE);
    pack_config(&mut words, OUT);
    unpack.extend(config_program(&words));
    unpack.push(set_adc_x_unpack(0, datums - 1));
    unpack.push(unpack_instruction());
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());

    let mut math = vec![state_id()];
    if !kernel.is_empty() {
        math.extend_from_slice(kernel);
        math.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
    }

    let pack = vec![
        state_id(),
        set_adc_x_pack(0, 15),
        pack_instruction(read_intf_sel, true),
        // Without this the host can read L1 before the packer has drained: the
        // thread unblocks once the packer has *accepted* the work, not finished
        // it (`Packers/README.md`).
        backend::wait_for_packer(Before::EVERYTHING).unwrap(),
    ];
    [unpack, math, pack]
}

/// Unpacker 0's address counters -- X, Y, Z and W of both channels -- back to
/// zero, for the issuing thread (`SETADCXY`, `SETADCZW`).
pub fn clear_unpacker0_adcs() -> [Instruction; 2] {
    [
        encode::Setadcxy::ZERO.u0(1).x0(1).y0(1).x1(1).y1(1).encode().unwrap(),
        encode::Setadczw::ZERO.u0(1).z0(1).w0(1).z1(1).w1(1).encode().unwrap(),
    ]
}

/// Datums in a 32x32 FP32 tile.
pub const TILE_DATUMS: u32 = 1024;
/// `Dst` rows one FP32 tile fills, sixteen datums to a row: its four faces in
/// order, each sixteen rows of one face row apiece.
pub const TILE_DST_ROWS: u32 = TILE_DATUMS / 16;

/// The `REG5_Dest_cntx0_address` that lands an unpack's first datum in `Dst`
/// row `row`: `Dst` row is `OutAddr / 16 - 4` (`UNPACR_Regular.md:430-432`),
/// so [`DST_BASE`] is row 0.
pub const fn dst_address(row: u32) -> u32 {
    (row + 4) * 16
}

/// A tile image's descriptor for a whole-tile unpack into `Dst`: its 1024
/// datums as one flat run. The image is `tt_layout`'s face order, so the flat
/// run lays the four faces down `Dst` one after another -- `Dst` row `16f + r`
/// is face `f`'s row `r` -- which is the order the packer writes back and the
/// SFPU's row groups walk (`tt_isa::sfpu::program`).
pub fn tile_descriptor() -> TileDescriptor {
    flat_descriptor(TILE_DATUMS)
}

/// Unpacker 0's configuration for whole FP32 tiles into `Dst`, starting with
/// the tile at `l1` into row 0 (`ThreadConfig` from [`thread_config`]).
pub fn tile_unpack_config(words: &mut ConfigWords, l1: u64) {
    unpack_config(words, tile_descriptor(), l1);
}

/// Unpack the FP32 tile image at `l1` (header included, 16-byte aligned) into
/// `Dst` rows `row..row + 64`, after whatever unpack came before: the
/// previous one drains before the base and destination are rewritten, and
/// the rewrite lands before this one starts (`ConfigurationUnit.md`).
pub fn unpack_tile_to_dst(l1: u64, row: u32) -> Vec<Instruction> {
    unpack_datums_to_dst(l1, 0, TILE_DATUMS, row)
}

/// `unpack_tile_to_dst`, for a tile loop whose base advances one `step` bytes
/// per iteration: the config still seeds slot 0, and the address the unpacker
/// actually uses comes from a register the translator stores before each unpack
/// and advances at the end of the body (`ttinsn::STEP_STORE`). One copy of the
/// body then serves the whole loop, which is what a block repeat needs.
pub fn unpack_tile_stepped(l1: u64, step: u32, row: u32) -> Vec<u32> {
    assert!(step > 0 && step % TileImage::ALIGNMENT as u32 == 0, "a stepped base moves in 16-byte units");
    let units = step / TileImage::ALIGNMENT as u32;
    assert!(units < 1 << 12, "a per-iteration step must fit the advance's 12 bits");
    let mut words: Vec<u32> = unpack_tile_to_dst(l1, row).iter().map(|i| i.word()).collect();
    // After the config and its stallwait, before the unpack instruction.
    let at = words.len() - 1;
    words.insert(at, crate::codegen::tenstorrent::ttinsn_step_store(l1));
    words.push(crate::codegen::tenstorrent::ttinsn_step_advance(units));
    words
}

/// Unpack `count` datums of the FP32 tile image at `l1`, from datum `first`
/// (a multiple of four, so the run starts on a 16-byte unit), into `Dst` from
/// row `row`. An uncompressed unpack always starts at the image's first datum
/// (`UNPACR_Regular.md`: `XPos = 0`), so the run is reached by moving the base
/// instead: the unpacker reads from `(Base_address + 1) * 16`, one header on.
pub fn unpack_datums_to_dst(l1: u64, first: u32, count: u32, row: u32) -> Vec<Instruction> {
    use tt_isa::backend::{self, Before};
    assert!(first % 4 == 0 && count > 0 && first + count <= TILE_DATUMS);
    let mut p = vec![backend::wait_for_unpacker0(Before::CONFIG).unwrap()];
    let mut words = ConfigWords::new();
    // The unpacker's context (descriptor words included) outlives the kernel
    // that wrote it, so every copy seeds the whole context: a flat copy that
    // inherits a matmul's `ZDim` or its clear `is_uncompressed` flag reads
    // the wrong thing (ttsim: `tensix_unpacr: is_uncompressed=0`).
    let base = l1 + 4 * first as u64;
    unpack_config(&mut words, tile_descriptor(), base);
    words
        .set(thcon::THCON_SEC0_REG3_Base_address, tile_base_units(base))
        .unwrap()
        .set(thcon::THCON_SEC0_REG5_Dest_cntx0_address, dst_address(row))
        .unwrap();
    p.extend(config_program(&words));
    p.push(backend::stallwait(backend::block::UNPACKER | backend::block::CONFIG, backend::cond::CONFIG_BUSY).unwrap());
    p.push(set_adc_x_unpack(0, count - 1));
    p.push(unpack_instruction());
    p
}

/// Pack `Dst` rows `row..row + 64` -- one FP32 tile's datums, faces in order,
/// no header -- to `l1_dest`, after whatever pack came before. The packer's
/// configuration is rewritten whole (as the matmul's pack role does per
/// tile), with the `Dst` read offset at `row`: the input address generator
/// adds `DEST_TARGET_REG_CFG_PACK_SEC0_Offset << 4` datums, a row each
/// (`Packers/InputAddressGenerator.md`).
pub fn pack_tile_from_dst(l1_dest: u64, row: u32) -> Vec<Instruction> {
    use tt_isa::backend::{self, Before};
    let mut p = vec![backend::wait_for_packer(Before::CONFIG).unwrap()];
    let mut words = ConfigWords::new();
    pack_config(&mut words, l1_dest);
    words.set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, row).unwrap();
    p.extend(config_program(&words));
    p.push(backend::stallwait(backend::block::PACKER | backend::block::CONFIG, backend::cond::CONFIG_BUSY).unwrap());
    p.extend(pack_rows(TILE_DST_ROWS));
    p
}

/// The optional packer stages of [`tt_isa::packer`], staged together with
/// the shared `Config` words they live in.
///
/// [`pack_config`] leaves the stages as `Config`'s reset state (ReLU off,
/// every datum kept); this is the explicit, checked way to turn one on.
/// **UNVERIFIED on Blackhole** (`Packers/{ReLU,EdgeMasking}.md` are
/// Wormhole-only pages): the gate is `step112_packer_relu_edge`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PackStages {
    pub relu: tt_isa::packer::PackerRelu,
    pub edge: tt_isa::packer::EdgeMasking,
    /// `ALU_ACC_CTRL_Zero_Flag_disabled_src`, which shares `Config` word 2 with
    /// ReLU, so staging ReLU must say what it should be.
    pub zero_flag_disabled_src: bool,
    /// Write every word of the stages, including the ones ttsim refuses
    /// (`EdgeMasking::apply_all`): silicon, so nothing is left to the last program.
    pub exhaustive: bool,
}

impl PackStages {
    /// Both stages off: what a whole-word write must restore after a gate.
    pub const OFF: Self = Self {
        relu: tt_isa::packer::PackerRelu::OFF,
        edge: tt_isa::packer::EdgeMasking::OFF,
        zero_flag_disabled_src: false,
        exhaustive: false,
    };

    pub fn apply(&self, words: &mut ConfigWords) -> Result<(), tt_isa::packer::PackModeError> {
        self.relu.apply(words, self.zero_flag_disabled_src)?;
        if self.exhaustive {
            self.edge.apply_all(words)
        } else {
            self.edge.apply(words)
        }
    }
}

/// [`pack_tile_from_dst`] with `stages` on: the packer's configuration is
/// rewritten whole, so the stages ride in the same `WRCFG` run.
pub fn pack_tile_from_dst_staged(
    l1_dest: u64,
    row: u32,
    stages: &PackStages,
) -> Result<Vec<Instruction>, tt_isa::packer::PackModeError> {
    use tt_isa::backend::{self, Before};
    let mut p = vec![backend::wait_for_packer(Before::CONFIG).unwrap()];
    let mut words = ConfigWords::new();
    pack_config(&mut words, l1_dest);
    words.set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, row).unwrap();
    stages.apply(&mut words)?;
    p.extend(config_program(&words));
    p.push(backend::stallwait(backend::block::PACKER | backend::block::CONFIG, backend::cond::CONFIG_BUSY).unwrap());
    p.extend(pack_rows(TILE_DST_ROWS));
    Ok(p)
}

// ---- BF16 into `Dst` (`UnpackToDst`, D1) -----------------------------------
//
// `UNPACR_Regular.md`'s `FormatConversion`: with `UnpackToDst`, a BF16 input
// converted to BF16 is `DstEncodeBF16(DatumBits)` and goes to `Dst16b[Row][Col]`
// (`Dst32b` is only for FP32/TF32/INT32 outputs). ttsim refuses it (divergence
// row 31); silicon accepts it (`silicon_measure::m08_bf16_unpack_to_dst`), and
// `step114_bf16_unpack_to_dst` is the gate.

/// The `THCON_SEC*_REG*_*_data_format` code of BF16 (`L1Format::Bf16.code()`).
pub fn bf16_code() -> u32 {
    L1Format::Bf16.code().expect("BF16's code is measured")
}

/// A flat run of `datums` BF16 datums, as [`flat_descriptor`] is for FP32.
pub fn bf16_flat_descriptor(datums: u32) -> TileDescriptor {
    flat_descriptor(datums).with_in_data_format_raw(bf16_code())
}

/// A BF16 tile image of raw 16-bit patterns, header included, the way
/// [`staged_image`] stages FP32.
pub fn staged_bf16_image(descriptor: TileDescriptor, patterns: &[u16]) -> Vec<u8> {
    let image = TileImage::new(descriptor, L1Format::Bf16).unwrap();
    let mut staged = vec![0u8; image.total_bytes()];
    for (i, &p) in patterns.iter().enumerate() {
        let off = image.datum_bit_offset(i) / 8;
        staged[off..off + 2].copy_from_slice(&p.to_le_bytes());
    }
    staged
}

/// Unpacker 0 for BF16 in, BF16 out, to `Dst` ([`unpack_config`] with the output
/// format retargeted; `descriptor` must be a [`bf16_flat_descriptor`]).
///
/// The result is `Dst16b`: BF16 rows of sixteen, `Dst` row `OutAddr / 16 - 4`.
/// A [`unpack_datums_to_dst`] after it retargets base and row as for FP32. Only
/// the first eight of every sixteen `Dst16b` rows are visible through the
/// firmware's FP32 dump (the 32-bit view pairs row `r` with `r + 8`).
pub fn unpack_bf16_config(words: &mut ConfigWords, descriptor: TileDescriptor, l1_base: u64) {
    assert_eq!(descriptor.in_data_format_raw(), bf16_code(), "a BF16 descriptor");
    unpack_config(words, descriptor, l1_base);
    words.set(thcon::THCON_SEC0_REG2_Out_data_format, bf16_code()).unwrap();
}

/// The packer reading `Dst16b` (BF16) and writing BF16 to `l1_dest`: [`pack_config`]
/// with the 16-bit read path -- `Read_32b_data` clear, a 32-byte `Dst` row stride,
/// BF16 both ways, and `Fp32_enabled` clear. **UNVERIFIED on Blackhole**
/// (`Packers/InputAddressGenerator.md` is a Wormhole page).
pub fn pack_bf16_dst16_config(words: &mut ConfigWords, l1_dest: u64) {
    pack_config(words, l1_dest);
    words
        .set(thcon::THCON_SEC0_REG1_In_data_format, bf16_code())
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Out_data_format, bf16_code())
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_32b_data, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_XY_REG_0_Ystride, DST_ROW_BYTES / 2)
        .unwrap()
        .set(alu::ALU_ACC_CTRL_Fp32_enabled, 0)
        .unwrap();
}

/// [`pack_tile_from_dst`] for BF16 in `Dst16b`: `Dst` rows `row..row + 64` to
/// `l1_dest` as 1024 BF16 datums, no header.
pub fn pack_bf16_tile_from_dst16(l1_dest: u64, row: u32) -> Vec<Instruction> {
    use tt_isa::backend::{self, Before};
    let mut p = vec![backend::wait_for_packer(Before::CONFIG).unwrap()];
    let mut words = ConfigWords::new();
    pack_bf16_dst16_config(&mut words, l1_dest);
    words.set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, row).unwrap();
    p.extend(config_program(&words));
    p.push(backend::stallwait(backend::block::PACKER | backend::block::CONFIG, backend::cond::CONFIG_BUSY).unwrap());
    p.extend(pack_rows(TILE_DST_ROWS));
    p
}

// ---- unpacker input modes: tileize and transpose (M3, D5) --------------------
//
// `tt_isa::unpacker` has the checked field staging; these are the program
// builders `step113_unpacker_modes` gates. Both are UNVERIFIED on Blackhole
// silicon. ttsim models both (tileize with Blackhole's 32-datum input rows;
// transpose on unpacker 0 into `SrcA`) and refuses transpose with
// `UnpackToDst` by name.

/// Unpacker 0 for FP32 into `Dst` reading `datums` datums of a row-major
/// matrix: 32 datums from each input row, the rows `row_stride_bytes` apart
/// (`Tileize_mode`). [`unpack_config`] with the mode staged over it.
pub fn unpack_tileize_config(
    words: &mut ConfigWords,
    descriptor: TileDescriptor,
    l1_base: u64,
    row_stride_bytes: u32,
) -> Result<(), tt_isa::unpacker::UnpackModeError> {
    unpack_config(words, descriptor, l1_base);
    tt_isa::unpacker::stage_tileize(words, row_stride_bytes, 32)
}

/// Unpacker 0 into `SrcA` ([`unpack_src_config`]) with transpose staged:
/// each 16x16 block of `SrcA` rows is written transposed.
pub fn unpack_src_transposed_config(
    words: &mut ConfigWords,
    descriptor: TileDescriptor,
    l1_base: u64,
    out: u32,
) -> Result<(), tt_isa::unpacker::UnpackModeError> {
    unpack_src_config(words, Unpacker::SrcA, descriptor, l1_base, out);
    tt_isa::unpacker::stage_transpose(words, false)
}
