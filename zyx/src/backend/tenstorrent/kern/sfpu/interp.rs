//! What an SFPU program computes, by the specification's functional models.
//!
//! [`Vector`] holds the state those models name -- `LReg[17][32]`, each lane's
//! `LaneFlags`, `UseLaneFlagsForLaneEnable` and flag stack (`VectorUnit.md`),
//! the issuing thread's `Dst` row counter and address modifiers (`RWCs.md`),
//! its replay buffer (`REPLAY.md`) and `Dst` itself -- and [`Vector::run`]
//! steps through a math thread's instruction stream, one functional model per
//! instruction, transcribed from each page. Anything it has no model for is
//! refused by name ([`InterpError::Unmodelled`]) rather than skipped: an
//! oracle that silently ignores an instruction would agree with a device that
//! did the same.
//!
//! Two things are modelled as the hardware's defaults rather than in full:
//! `LaneConfig` is all zero (no row masks, no column exchange), and the `Dst`
//! address adjustments other than the row counter (`DEST_TARGET_REG_CFG_MATH`,
//! `DEST_REGW_BASE`, the stack pointer) are zero -- which is what every
//! program here runs with; a program that changed them would need them
//! modelled first.
//!
//! The arithmetic is `tt_isa::numerics::fma_bh`, itself held bit for bit to
//! the specification's `fma.c` (`fma_oracle`).

use tt_isa::isa::Instruction;
use tt_isa::isa::generated::defs;
use tt_isa::numerics::fma_bh;
use tt_isa::numerics::sfpu::sign_mag_is_smaller;

/// `Dst` rows the model holds: the ten bits of a `Dst` row address.
pub const DST_ROWS: usize = 1024;

/// Why a program could not be run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterpError {
    /// An instruction, or a mode of one, with no model here.
    Unmodelled { at: usize, what: String },
    /// A read of an `LReg` whose contents nothing has established: `LReg[8]`
    /// (a constant documented only as "0.8373"), or `LReg[11..15]` before an
    /// `SFPCONFIG`.
    Unknown { at: usize, lreg: u32 },
    /// Something the page calls `UndefinedBehavior`.
    Undefined { at: usize, what: &'static str },
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpError::Unmodelled { at, what } => {
                write!(f, "instruction {at}: no model for {what}")
            }
            InterpError::Unknown { at, lreg } => {
                write!(f, "instruction {at}: LReg[{lreg}] holds nothing established")
            }
            InterpError::Undefined { at, what } => write!(f, "instruction {at}: undefined: {what}"),
        }
    }
}

impl std::error::Error for InterpError {}

/// The Vector Unit, its thread's counters, and `Dst`, as the models see them.
#[derive(Clone)]
pub struct Vector {
    /// `None` where nothing has established a register's contents.
    pub lreg: [Option<[u32; 32]>; 17],
    pub lane_flags: [bool; 32],
    pub use_flags: [bool; 32],
    pub stack: [Vec<(bool, bool)>; 32],
    /// `Dst32b`, `[row][column]`.
    pub dst: Vec<[u32; 16]>,
    /// `RWCs[thread].Dst`.
    pub rwc_dst: u32,
    /// `ADDR_MOD_DST_SEC[i].DestIncr`.
    pub dst_incr: [u32; 8],
    replay: [Option<Instruction>; 32],
    /// Per-lane hardware PRNG state. Unknown until a caller supplies a seed
    /// or an observed snapshot; software cannot assume silicon reset seeds.
    pub prng: Option<[u32; 32]>,
}

impl Default for Vector {
    fn default() -> Self {
        Self::new()
    }
}

impl Vector {
    /// The state a program starts from: general registers zero (the models'
    /// `0` -- a program must not read one it has not written, and the gates
    /// do not), the constants, every lane enabled by default, `Dst` zero.
    pub fn new() -> Self {
        let mut lreg: [Option<[u32; 32]>; 17] = [None; 17];
        for r in lreg.iter_mut().take(8) {
            *r = Some([0; 32]);
        }
        lreg[9] = Some([0; 32]);
        lreg[10] = Some([1.0f32.to_bits(); 32]);
        lreg[15] = Some(std::array::from_fn(|i| 2 * i as u32));
        Vector {
            lreg,
            lane_flags: [false; 32],
            use_flags: [false; 32],
            stack: std::array::from_fn(|_| Vec::new()),
            dst: vec![[0; 16]; DST_ROWS],
            rwc_dst: 0,
            dst_incr: [0; 8],
            replay: [None; 32],
            prng: None,
        }
    }

    fn lane_enabled(&self, lane: usize) -> bool {
        !self.use_flags[lane] || self.lane_flags[lane]
    }

    fn read(&self, at: usize, r: u32) -> Result<[u32; 32], InterpError> {
        self.lreg[r as usize].ok_or(InterpError::Unknown { at, lreg: r })
    }

    /// Write lane-wise where enabled (`LaneEnabled`), and only to registers an
    /// instruction can write (`VD < 8`; the rest are dropped, as on the device).
    fn write(&mut self, vd: u32, v: [u32; 32], all_lanes: bool) {
        if vd >= 8 {
            return;
        }
        let en: [bool; 32] = std::array::from_fn(|l| all_lanes || self.lane_enabled(l));
        let r = self.lreg[vd as usize].get_or_insert([0; 32]);
        for l in 0..32 {
            if en[l] {
                r[l] = v[l];
            }
        }
    }

    /// The `Dst` row and column lane `lane` of an `SFPLOAD`/`SFPSTORE` at
    /// `imm10` reaches, and the counter stepped by `addr_mod`.
    fn dst_at(&self, imm10: u32, lane: usize) -> (usize, usize) {
        let addr = (imm10 + self.rwc_dst) & 0x3ff;
        let row = (addr & !3) as usize + lane / 8;
        let col = (lane & 7) * 2 + usize::from(addr & 2 != 0);
        (row, col)
    }

    fn step(&mut self, addr_mod: u32) {
        self.rwc_dst = (self.rwc_dst + self.dst_incr[addr_mod as usize]) & 0x3ff;
    }

    /// Run `program`, the math thread's stream. Instructions for other units
    /// that an SFPU program carries -- waits, semaphores -- have no effect on
    /// this state and are passed over by name.
    pub fn run(&mut self, program: &[Instruction]) -> Result<(), InterpError> {
        let mut i = 0;
        while i < program.len() {
            let ins = program[i];
            if core::ptr::eq(ins.def(), &defs::REPLAY) {
                let op = |n| ins.operand(n).unwrap();
                let (index, count) = (op("Index") as usize, op("Count") as usize);
                let count = if count == 0 { 64 } else { count };
                if op("Load") != 0 {
                    for k in 0..count {
                        let Some(&b) = program.get(i + 1 + k) else {
                            return Err(InterpError::Undefined { at: i, what: "REPLAY records past the end of the program" });
                        };
                        self.replay[(index + k) % 32] = Some(b);
                        if op("Exec") != 0 {
                            self.exec(i + 1 + k, b)?;
                        }
                    }
                    i += 1 + count;
                } else {
                    for k in 0..count {
                        let b = self.replay[(index + k) % 32]
                            .ok_or(InterpError::Undefined { at: i, what: "REPLAY of a slot nothing recorded" })?;
                        self.exec(i, b)?;
                    }
                    i += 1;
                }
                continue;
            }
            self.exec(i, ins)?;
            i += 1;
        }
        Ok(())
    }

    fn exec(&mut self, at: usize, ins: Instruction) -> Result<(), InterpError> {
        let op = |n: &str| ins.operand(n).unwrap_or(0);
        let unmodelled = |what: String| Err(InterpError::Unmodelled { at, what });
        // `VD >= 12` writes the instruction into an `SFPLOADMACRO` template
        // while `LaneConfig.DISABLE_BACKDOOR_LOAD` is clear, as it is here.
        let backdoor = |vd: u32| {
            if vd >= 12 {
                Err(InterpError::Unmodelled {
                    at,
                    what: format!("{} with VD {vd} (a backdoor template load)", ins.def().mnemonic()),
                })
            } else {
                Ok(())
            }
        };
        match ins.def().mnemonic() {
            "SFPNOP" | "STALLWAIT" | "SEMWAIT" | "SEMPOST" | "SEMGET" | "NOP" => {}
            "SFPLOAD" | "SFPSTORE" => {
                let (vd, mod0, imm) = (op("VD"), op("Mod0"), op("Imm10"));
                // `MOD0_FMT_FP32` and `MOD0_FMT_INT32`: `Dst32b` as is on a
                // load; on a store FP32 flushes denormals (`ToFP32`).
                if mod0 != 3 && mod0 != 4 {
                    return unmodelled(format!("{} with Mod0 {mod0}", ins.def().mnemonic()));
                }
                if ins.def().mnemonic() == "SFPLOAD" {
                    if vd < 8 {
                        let mut v = [0; 32];
                        for (l, x) in v.iter_mut().enumerate() {
                            let (r, c) = self.dst_at(imm, l);
                            *x = self.dst[r][c];
                        }
                        self.write(vd, v, false);
                    }
                } else {
                    if vd >= 12 {
                        return unmodelled(format!("SFPSTORE from LReg[{vd}]"));
                    }
                    let v = self.read(at, vd)?;
                    for (l, &x) in v.iter().enumerate() {
                        if self.lane_enabled(l) {
                            let (r, c) = self.dst_at(imm, l);
                            self.dst[r][c] = if mod0 == 3 && x & 0x7f80_0000 == 0 {
                                x & 0x8000_0000
                            } else {
                                x
                            };
                        }
                    }
                }
                self.step(op("AddrMod"));
            }
            "SFPLOADI" => {
                let (vd, mod0, imm) = (op("VD"), op("Mod0"), op("Imm16"));
                let old = self.lreg[vd.min(16) as usize].unwrap_or([0; 32]);
                let v: [u32; 32] = std::array::from_fn(|l| match mod0 {
                    0 => imm << 16,
                    1 => {
                        let (s, e, m) = (imm >> 15, (imm >> 10) & 0x1f, imm & 0x3ff);
                        (s << 31) | ((e + 112) << 23) | (m << 13)
                    }
                    2 => imm,
                    4 => imm as u16 as i16 as i32 as u32,
                    8 => (imm << 16) | (old[l] & 0xffff),
                    10 => (old[l] & 0xffff_0000) | imm,
                    _ => 0,
                });
                if ![0, 1, 2, 4, 8, 10].contains(&mod0) {
                    return Err(InterpError::Undefined { at, what: "SFPLOADI with an undefined Mod0" });
                }
                self.write(vd, v, false);
            }
            "SFPMAD" | "SFPMUL" | "SFPADD" => {
                let (va, vb, vc, vd, mod1) = (op("VA"), op("VB"), op("VC"), op("VD"), op("Mod1"));
                if mod1 & 12 != 0 {
                    return unmodelled(format!("{} with indirect registers", ins.def().mnemonic()));
                }
                let (a, b, c) = (self.read(at, va)?, self.read(at, vb)?, self.read(at, vc)?);
                let v: [u32; 32] = std::array::from_fn(|l| {
                    let a = a[l] ^ if mod1 & 1 != 0 { 0x8000_0000 } else { 0 };
                    let c = c[l] ^ if mod1 & 2 != 0 { 0x8000_0000 } else { 0 };
                    fma_bh(a, b[l], c)
                });
                self.write(vd, v, false);
            }
            "SFPMOV" => {
                let (vc, vd, mod1) = (op("VC"), op("VD"), op("Mod1") & !4);
                if mod1 & 8 != 0 && vc == 9 {
                    let mut state = self.prng.ok_or_else(|| InterpError::Unmodelled { at, what: "unknown PRNG state".into() })?;
                    let old = state;
                    for (lane, value) in state.iter_mut().enumerate() {
                        if self.lane_enabled(lane) {
                            *value = tt_isa::numerics::stochastic::advance(*value);
                        }
                    }
                    self.prng = Some(state);
                    self.write(vd, old, false);
                    return Ok(());
                }
                if mod1 & 8 != 0 {
                    return unmodelled("SFPMOV from a special register".into());
                }
                let x = self.read(at, vc)?;
                let v = x.map(|x| if mod1 & 1 != 0 { x ^ 0x8000_0000 } else { x });
                self.write(vd, v, mod1 == 2);
            }
            "SFP_STOCH_RND" => {
                use tt_isa::numerics::stochastic::{self, Precision, Rounding};
                let precision = match op("Mod1") {
                    0 => Precision::Tf32,
                    1 => Precision::Bf16,
                    _ => return unmodelled("SFPSTOCHRND integer conversion".into()),
                };
                let rounding = match op("RoundingMode") {
                    0 => Rounding::Nearest,
                    1 => Rounding::Stochastic,
                    2 => Rounding::TowardZero,
                    _ => return unmodelled("SFPSTOCHRND rounding mode".into()),
                };
                if rounding == Rounding::Stochastic && self.prng.is_none() {
                    return unmodelled("unknown PRNG state".into());
                }
                let input = self.read(at, op("VC"))?;
                let state = self.prng.unwrap_or([0; 32]);
                let result = std::array::from_fn(|l| stochastic::round(input[l], state[l], precision, rounding));
                if self.prng.is_some() {
                    let mut next = state;
                    for l in 0..32 {
                        if self.lane_enabled(l) {
                            next[l] = stochastic::advance(state[l]);
                        }
                    }
                    self.prng = Some(next);
                }
                self.write(op("VD"), result, false);
            }
            "SFPABS" => {
                let (vc, vd, mod1) = (op("VC"), op("VD"), op("Mod1"));
                let v = self.read(at, vc)?.map(|x| {
                    if x < 0x8000_0000 {
                        x
                    } else if mod1 & 1 != 0 {
                        if x > 0xff80_0000 { x } else { x & 0x7fff_ffff }
                    } else {
                        x.wrapping_neg()
                    }
                });
                self.write(vd, v, false);
            }
            "SFPSETSGN" => {
                let (vc, vd, mod1, imm1) = (op("VC"), op("VD"), op("Mod1"), op("Imm1"));
                let c = self.read(at, vc)?;
                let b = if mod1 & 1 != 0 { [0; 32] } else { self.read(at, vd)? };
                let v: [u32; 32] = std::array::from_fn(|l| {
                    let sign = if mod1 & 1 != 0 { imm1 & 1 } else { b[l] >> 31 };
                    (sign << 31) | (c[l] & 0x7fff_ffff)
                });
                self.write(vd, v, false);
            }
            "SFPSETCC" => {
                let (vc, mod1, imm1) = (op("VC"), op("Mod1"), op("Imm1"));
                let c = self.read(at, vc)?;
                for l in 0..32 {
                    if !self.lane_enabled(l) {
                        continue;
                    }
                    self.lane_flags[l] = if !self.use_flags[l] || mod1 & 8 != 0 {
                        false
                    } else if mod1 & 1 != 0 {
                        imm1 != 0
                    } else {
                        let c = c[l] as i32;
                        match mod1 {
                            0 => c < 0,
                            2 => c != 0,
                            4 => c >= 0,
                            6 => c == 0,
                            _ => return Err(InterpError::Undefined { at, what: "SFPSETCC with an undefined Mod1" }),
                        }
                    };
                }
            }
            "SFPAND" | "SFPOR" => {
                let (vb, vc, vd, mod1) = (op("VB"), op("VC"), op("VD"), op("Mod1"));
                let b = self.read(at, if mod1 & 1 != 0 { vb } else { vd })?;
                let c = self.read(at, vc)?;
                let and = ins.def().mnemonic() == "SFPAND";
                let v: [u32; 32] = std::array::from_fn(|l| if and { b[l] & c[l] } else { b[l] | c[l] });
                self.write(vd, v, false);
            }
            "SFPSHFT2" => {
                let (vc, vd, mod1) = (op("VC"), op("VD"), op("Mod1"));
                // `SFPSHFT2_MOD1_SUBVEC_SHFLROR1`, and its zero-filling twin.
                if mod1 != 3 && mod1 != 4 {
                    return unmodelled(format!("SFPSHFT2 with Mod1 {mod1}"));
                }
                let c = self.read(at, vc)?;
                let v: [u32; 32] = std::array::from_fn(|l| {
                    if l & 7 != 0 {
                        c[l - 1]
                    } else if mod1 == 3 {
                        c[l + 7]
                    } else {
                        0
                    }
                });
                self.write(vd, v, false);
            }
            "SFPTRANSP" => {
                for base in [0usize, 4] {
                    let mut r: [[u32; 32]; 4] = [[0; 32]; 4];
                    for (i, ri) in r.iter_mut().enumerate() {
                        *ri = self.read(at, (base + i) as u32)?;
                    }
                    let old = r;
                    for col in 0..8 {
                        for i in 0..4 {
                            for j in 0..i {
                                let (ij, ji) = (old[i][j * 8 + col], old[j][i * 8 + col]);
                                if self.lane_enabled(j * 8 + col) {
                                    r[i][j * 8 + col] = ji;
                                }
                                if self.lane_enabled(i * 8 + col) {
                                    r[j][i * 8 + col] = ij;
                                }
                            }
                        }
                    }
                    for (i, ri) in r.iter().enumerate() {
                        self.lreg[base + i] = Some(*ri);
                    }
                }
            }
            "SFPIADD" => {
                let (imm, vc, vd, mod1) = (op("Imm12"), op("VC"), op("VD"), op("Mod1"));
                let c = self.read(at, vc)?;
                let b = self.read(at, vd)?;
                let sext = ((imm << 20) as i32 >> 20) as u32;
                let v: [u32; 32] = std::array::from_fn(|l| {
                    if mod1 & 1 != 0 {
                        c[l].wrapping_add(sext)
                    } else if mod1 & 2 != 0 {
                        c[l].wrapping_sub(b[l])
                    } else {
                        c[l].wrapping_add(b[l])
                    }
                });
                let en: [bool; 32] = std::array::from_fn(|l| self.lane_enabled(l));
                self.write(vd, v, false);
                if vd < 8 {
                    for l in 0..32 {
                        if !en[l] {
                            continue;
                        }
                        if mod1 & 4 == 0 {
                            self.lane_flags[l] = (v[l] as i32) < 0;
                        }
                        if mod1 & 8 != 0 {
                            self.lane_flags[l] = !self.lane_flags[l];
                        }
                    }
                }
            }
            "SFPSHFT" => {
                let (imm, vc, vd, mod1) = (op("Imm12"), op("VC"), op("VD"), op("Mod1"));
                if mod1 > 7 || mod1 == 4 || mod1 == 6 {
                    return unmodelled(format!("SFPSHFT with reserved Mod1 {mod1}"));
                }
                let c = self.read(at, vc)?;
                let b = self.read(at, vd)?;
                let v: [u32; 32] = std::array::from_fn(|l| {
                    let mut x = b[l];
                    let mut amount = c[l] as i32;
                    if mod1 & 1 != 0 {
                        if mod1 & 4 != 0 {
                            x = c[l];
                        }
                        amount = imm as i32;
                    }
                    if amount >= 0 {
                        x << (amount & 31)
                    } else if mod1 & 2 != 0 {
                        ((x as i32) >> ((-amount) & 31)) as u32
                    } else {
                        x >> ((-amount) & 31)
                    }
                });
                self.write(vd, v, false);
            }
            "SFPEXEXP" => {
                let (vc, vd, mod1) = (op("VC"), op("VD"), op("Mod1"));
                let bias = if mod1 & 1 != 0 { 0 } else { 127 };
                let c = self.read(at, vc)?;
                let v = c.map(|c| (((c >> 23) & 0xff) as i32 - bias) as u32);
                let en: [bool; 32] = std::array::from_fn(|l| self.lane_enabled(l));
                self.write(vd, v, false);
                if vd < 8 && mod1 & 10 != 0 {
                    for l in 0..32 {
                        if en[l] {
                            if mod1 & 2 != 0 {
                                self.lane_flags[l] = (v[l] as i32) < 0;
                            }
                            if mod1 & 8 != 0 {
                                self.lane_flags[l] = !self.lane_flags[l];
                            }
                        }
                    }
                }
            }
            "SFPEXMAN" => {
                let (vc, vd, mod1) = (op("VC"), op("VD"), op("Mod1"));
                let hidden = if mod1 & 1 != 0 { 0 } else { 1 << 23 };
                let v = self.read(at, vc)?.map(|c| hidden + (c & 0x7f_ffff));
                self.write(vd, v, false);
            }
            "SFPSETEXP" => {
                let (imm, vc, vd, mod1) = (op("Imm8"), op("VC"), op("VD"), op("Mod1"));
                let c = self.read(at, vc)?;
                let b = if mod1 & 1 != 0 { [0; 32] } else { self.read(at, vd)? };
                let v: [u32; 32] = std::array::from_fn(|l| {
                    let e = if mod1 & 1 != 0 {
                        imm
                    } else if mod1 & 2 != 0 {
                        (b[l] >> 23) & 0xff
                    } else {
                        b[l] & 0xff
                    };
                    (c[l] & 0x807f_ffff) | (e << 23)
                });
                self.write(vd, v, false);
            }
            "SFPSETMAN" => {
                let (imm, vc, vd, mod1) = (op("Imm12"), op("VC"), op("VD"), op("Mod1"));
                let c = self.read(at, vc)?;
                let b = if mod1 & 1 != 0 { [0; 32] } else { self.read(at, vd)? };
                let v: [u32; 32] = std::array::from_fn(|l| {
                    let m = if mod1 & 1 != 0 { imm << 11 } else { b[l] & 0x7f_ffff };
                    (c[l] & 0xff80_0000) | m
                });
                self.write(vd, v, false);
            }
            "SFPDIVP2" => {
                let (imm, vc, vd, mod1) = (op("Imm8"), op("VC"), op("VD"), op("Mod1"));
                let v = self.read(at, vc)?.map(|c| {
                    let e = (c >> 23) & 0xff;
                    let e = if mod1 & 1 != 0 {
                        if e == 255 { e } else { (e + imm) & 0xff }
                    } else {
                        imm
                    };
                    (c & 0x807f_ffff) | (e << 23)
                });
                self.write(vd, v, false);
            }
            "SFPARECIP" => {
                let (vb, vc, vd, mod1) = (op("VB"), op("VC"), op("VD"), op("Mod1"));
                if vd >= 8 {
                    return unmodelled(format!("SFPARECIP into LReg[{vd}]"));
                }
                let c = self.read(at, vc)?;
                let b = if mod1 == 1 { self.read(at, vb)? } else { [0; 32] };
                let d = self.read(at, vd)?;
                let v: [u32; 32] = std::array::from_fn(|l| tt_isa::numerics::sfpu::arecip(mod1, b[l], c[l], d[l]));
                self.write(vd, v, false);
            }
            "SFPGT" | "SFPLE" => {
                let m = ins.def().mnemonic();
                let (vc, vd, mod1) = (op("VC"), op("VD"), op("Mod1"));
                if mod1 & 2 != 0 {
                    return unmodelled(format!("{m} mutating the flag stack"));
                }
                backdoor(vd)?;
                // `SFPGT`: `SignMagIsSmaller(C, D)`, `D > C`; `SFPLE`: its
                // negation, `D <= C`.
                let le = m == "SFPLE";
                let (c, d) = (self.read(at, vc)?, self.read(at, vd)?);
                let holds: [bool; 32] = std::array::from_fn(|l| sign_mag_is_smaller(c[l], d[l]) != le);
                if mod1 & 8 != 0 {
                    let v = holds.map(|s| if s { u32::MAX } else { 0 });
                    self.write(vd, v, false);
                }
                if mod1 & 1 != 0 {
                    for (l, &h) in holds.iter().enumerate() {
                        if self.lane_enabled(l) {
                            self.lane_flags[l] = h;
                        }
                    }
                }
            }
            "SFPSWAP" => {
                let (vc, vd, mod1) = (op("VC"), op("VD"), op("Mod1"));
                backdoor(vd)?;
                // `VDGetsMin`, one bit a lane; `LaneConfig` is zero here, so
                // neither `EXCHANGE_SRCB_SRCC` nor `ENABLE_DEST_INDEX` applies.
                let gets_min: u32 = match mod1 {
                    1 => 0xffff_ffff,
                    2 => 0x0000_ffff,
                    3 => 0x00ff_00ff,
                    4 => 0xff00_00ff,
                    5 => 0x0000_00ff,
                    6 => 0x0000_ff00,
                    7 => 0x00ff_0000,
                    8 => 0xff00_0000,
                    _ => 0,
                };
                if mod1 > 9 {
                    return unmodelled(format!("SFPSWAP with Mod1 {mod1} (NonContractualBehavior)"));
                }
                let (c, d) = (self.read(at, vc)?, self.read(at, vd)?);
                let (mut nc, mut nd) = (c, d);
                for l in 0..32 {
                    if !self.lane_enabled(l) {
                        continue;
                    }
                    let swap = mod1 == 0 || {
                        let s = sign_mag_is_smaller(c[l], d[l]) || (c[l] == d[l] && c[l] & 0x8000_0000 != 0);
                        s == (gets_min >> l & 1 != 0)
                    };
                    if swap {
                        nc[l] = d[l];
                        nd[l] = c[l];
                    }
                }
                if vc < 8 {
                    self.lreg[vc as usize] = Some(nc);
                }
                if vd < 8 {
                    self.lreg[vd as usize] = Some(nd);
                }
            }
            "SFPMULI" | "SFPADDI" => {
                let m = ins.def().mnemonic();
                let (imm, vd, mod1) = (op("Imm16"), op("VD"), op("Mod1"));
                if mod1 & 8 != 0 {
                    return unmodelled(format!("{m} with an indirect VD"));
                }
                backdoor(vd)?;
                let c = self.read(at, vd)?;
                let v = c.map(|c| {
                    let c = if mod1 & 2 != 0 { c ^ 0x8000_0000 } else { c };
                    if m == "SFPMULI" {
                        fma_bh(imm << 16, c, 0)
                    } else {
                        fma_bh(imm << 16, 1.0f32.to_bits(), c)
                    }
                });
                self.write(vd, v, false);
            }
            "SFPXOR" => {
                let (vc, vd) = (op("VC"), op("VD"));
                let (c, b) = (self.read(at, vc)?, self.read(at, vd)?);
                let v: [u32; 32] = std::array::from_fn(|l| b[l] ^ c[l]);
                self.write(vd, v, false);
            }
            "SFPNOT" => {
                let (vc, vd) = (op("VC"), op("VD"));
                let v = self.read(at, vc)?.map(|c| !c);
                self.write(vd, v, false);
            }
            "SFPLZ" => {
                let (vc, vd, mod1) = (op("VC"), op("VD"), op("Mod1"));
                if mod1 & 1 != 0 {
                    return unmodelled("SFPLZ with its reserved Mod1 bit".into());
                }
                let c = self.read(at, vc)?.map(|c| if mod1 & 4 != 0 { c & 0x7fff_ffff } else { c });
                let v = c.map(|c| c.leading_zeros());
                let en: [bool; 32] = std::array::from_fn(|l| self.lane_enabled(l));
                self.write(vd, v, false);
                if vd < 8 {
                    for l in 0..32 {
                        if en[l] {
                            if mod1 & 2 != 0 {
                                self.lane_flags[l] = c[l] != 0;
                            }
                            if mod1 & 8 != 0 {
                                self.lane_flags[l] = !self.lane_flags[l];
                            }
                        }
                    }
                }
            }
            "SFPMUL24" => {
                let (va, vb, vc, vd, mod1) = (op("VA"), op("VB"), op("VC"), op("VD"), op("Mod1"));
                if mod1 & 12 != 0 {
                    return unmodelled("SFPMUL24 with indirect registers".into());
                }
                backdoor(vd)?;
                if self.read(at, vc)?.iter().any(|&c| c != 0) {
                    return unmodelled("SFPMUL24 with a non-zero VC (its shift-add is NonContractualBehavior)".into());
                }
                let (a, b) = (self.read(at, va)?, self.read(at, vb)?);
                let v: [u32; 32] = std::array::from_fn(|l| tt_isa::numerics::sfpu::mul24(a[l], b[l], mod1 & 1 != 0));
                self.write(vd, v, false);
            }
            "SFPCAST" => {
                let (vc, vd, mod1) = (op("VC"), op("VD"), op("Mod1"));
                // Only `SFPCAST_IntFloat` rounding to nearest; the stochastic
                // mode draws on the PRNG, and the other two flavours are other
                // pages (one of them the Tier 2 `IntAbs` bug).
                if mod1 != 0 {
                    return unmodelled(format!("SFPCAST with Mod1 {mod1}"));
                }
                backdoor(vd)?;
                let v = self.read(at, vc)?.map(tt_isa::numerics::sfpu::cast_sm32_to_fp32_rne);
                self.write(vd, v, false);
            }
            "SFPCONFIG" => {
                let (imm, vd, mod1) = (op("Imm16"), op("VD"), op("Mod1"));
                // Only the writes of `LReg[11..15]`: the `SFPLOADMACRO`
                // templates, `Misc` and `LaneConfig` are not modelled.
                if !(11..15).contains(&vd) {
                    return unmodelled(format!("SFPCONFIG of target {vd}"));
                }
                let fixed = [0xbf80_0000u32, 0x3b00_0000, 0xbf2c_c4c7, 0xbeb0_8ff9];
                let src = if mod1 & 1 != 0 {
                    [fixed[vd as usize - 11]; 32]
                } else {
                    self.read(at, 0)?
                };
                let mut r = self.lreg[vd as usize].unwrap_or([0; 32]);
                let mut all = true;
                for (l, x) in r.iter_mut().enumerate() {
                    // Both the lane mask and the predication are lane `l & 7`'s.
                    let masked = mod1 & 8 != 0 && imm >> ((l & 7) * 2) & 1 == 0;
                    let off = self.use_flags[l & 7] && !self.lane_flags[l & 7];
                    if masked || off {
                        all = false;
                        continue;
                    }
                    *x = src[l & 7];
                }
                if !all && self.lreg[vd as usize].is_none() {
                    return unmodelled(format!("SFPCONFIG writing some lanes of LReg[{vd}], which holds nothing established"));
                }
                self.lreg[vd as usize] = Some(r);
            }
            "SFPLUT" => {
                let (vd, mod0) = (op("VD"), op("Mod0"));
                if mod0 & 8 != 0 {
                    return unmodelled("SFPLUT with an indirect VD".into());
                }
                backdoor(vd)?;
                let l3 = self.read(at, 3)?;
                let t = [self.read(at, 0)?, self.read(at, 1)?, self.read(at, 2)?];
                let v: [u32; 32] = std::array::from_fn(|l| {
                    let i = tt_isa::numerics::sfpu::lut_index(l3[l]);
                    tt_isa::numerics::sfpu::lut(t[i][l], l3[l], mod0 & 4 != 0)
                });
                self.write(vd, v, false);
            }
            "SFPLUTFP32" => {
                use tt_isa::numerics::sfpu::{lutfp32, lutfp32_mod1 as m};
                let (vd, mod1) = (op("VD"), op("Mod1"));
                backdoor(vd)?;
                let fp16_3 = mod1 & m::FP16_3ENTRY_TABLE == m::FP16_3ENTRY_TABLE;
                let mut regs = vec![0, 1, 2, 3];
                if !fp16_3 {
                    regs.extend([4, 5, 6]);
                }
                if mod1 & m::INDIRECT_VD != 0 {
                    regs.push(7);
                }
                let mut l = [[0u32; 32]; 8];
                for r in regs {
                    l[r] = self.read(at, r as u32)?;
                }
                let en: [bool; 32] = std::array::from_fn(|lane| self.lane_enabled(lane));
                for lane in 0..32 {
                    if !en[lane] {
                        continue;
                    }
                    let lanes: [u32; 8] = std::array::from_fn(|r| l[r][lane]);
                    let d = lutfp32(mod1, &lanes);
                    // The Tier 2 hazard, modelled: with `INDIRECT_VD` set --
                    // and `FP16_3ENTRY_TABLE` sets it -- the result goes to
                    // `LReg[LReg[7] & 15]`, whatever `VD` says.
                    let to = if mod1 & m::INDIRECT_VD != 0 { lanes[7] & 15 } else { vd };
                    if to < 8 {
                        self.lreg[to as usize].get_or_insert([0; 32])[lane] = d;
                    }
                }
            }
            "SFPENCC" => {
                let (mod1, imm2) = (op("Mod1"), op("Imm2"));
                for l in 0..32 {
                    if mod1 & 2 != 0 {
                        self.use_flags[l] = imm2 & 1 != 0;
                    } else if mod1 & 1 != 0 {
                        self.use_flags[l] = !self.use_flags[l];
                    }
                    self.lane_flags[l] = if mod1 & 8 != 0 { imm2 & 2 != 0 } else { true };
                }
            }
            "SFPPUSHC" => {
                let mod1 = op("Mod1");
                if mod1 != 0 {
                    return unmodelled(format!("SFPPUSHC with Mod1 {mod1}"));
                }
                for l in 0..32 {
                    if self.stack[l].len() >= 8 {
                        return Err(InterpError::Undefined { at, what: "SFPPUSHC on a full flag stack" });
                    }
                    self.stack[l].push((self.lane_flags[l], self.use_flags[l]));
                }
            }
            "SFPPOPC" => {
                let mod1 = op("Mod1");
                if mod1 != 0 {
                    return unmodelled(format!("SFPPOPC with Mod1 {mod1}"));
                }
                for l in 0..32 {
                    let (f, u) =
                        self.stack[l].pop().ok_or(InterpError::Undefined { at, what: "SFPPOPC on an empty flag stack" })?;
                    self.lane_flags[l] = f;
                    self.use_flags[l] = u;
                }
            }
            "SFPCOMPC" => {
                for l in 0..32 {
                    let (tf, tu) = self.stack[l].last().copied().unwrap_or((true, true));
                    self.lane_flags[l] = if tu && self.use_flags[l] {
                        tf && !self.lane_flags[l]
                    } else {
                        false
                    };
                }
            }
            "SETRWC" => {
                // Only the counter-clearing form the builder emits.
                let bits: u32 = [
                    "FlipSrcB", "FlipSrcA", "DstCtoCr", "DstCr", "SrcBCr", "SrcACr", "DstVal", "Fidelity", "SrcB",
                ]
                .iter()
                .map(|n| op(n))
                .sum();
                if bits != 0 || op("SrcA") != 0 || op("SrcAVal") != 0 || op("SrcBVal") != 0 {
                    return unmodelled("SETRWC other than clearing Dst".into());
                }
                if op("Dst") != 0 {
                    self.rwc_dst = 0;
                }
            }
            "SETC16" => {
                // An address modifier's `Dst` increment, and only that: the
                // field's word is written whole, and anything else in it set
                // would be a mode this model does not have.
                let (index, value) = (op("CfgIndex"), op("NewValue"));
                let entry = (0..8).find(|&e| super::super::datapath::addr_mod_entry(e).dst_incr.addr32() as u32 == index);
                match entry {
                    Some(e) if value & !0x3ff == 0 => self.dst_incr[e] = value,
                    Some(_) => return unmodelled("an address modifier's CR, clear or fidelity bits".into()),
                    None => return unmodelled(format!("SETC16 of ThreadConfig entry {index}")),
                }
            }
            m => return unmodelled(m.to_string()),
        }
        Ok(())
    }

    /// Write a 64-row FP32 tile -- face order, as the unpacker lays it down --
    /// into `Dst` rows `row..row + 64`.
    pub fn put_tile(&mut self, row: usize, datums: &[u32]) {
        assert_eq!(datums.len(), 1024);
        for (i, d) in datums.iter().enumerate() {
            self.dst[row + i / 16][i % 16] = *d;
        }
    }

    /// `Dst` rows `row..row + 64` as a tile's datums, as the packer reads them.
    pub fn tile(&self, row: usize) -> Vec<u32> {
        (0..1024).map(|i| self.dst[row + i / 16][i % 16]).collect()
    }
}

#[cfg(any())]
mod tests {
    use super::super::{Cond, Format, LoopPolicy, Program};
    use super::*;
    use tt_isa::sfpu::LReg;

    fn tiles() -> (Vec<u32>, Vec<u32>) {
        let a = (0..1024).map(|i| (i as f32 - 300.0).to_bits()).collect();
        let b = (0..1024).map(|i| (0.5 * i as f32).to_bits()).collect();
        (a, b)
    }

    fn add_program(policy: LoopPolicy) -> Vec<Instruction> {
        let mut p = Program::with_policy(policy);
        p.for_each_row_group(64, |p, o| {
            p.load(LReg::L0, Format::Fp32, o);
            p.load(LReg::L1, Format::Fp32, 64 + o);
            p.add(LReg::L0, LReg::L1, LReg::L2);
            p.store(LReg::L2, Format::Fp32, 128 + o);
        });
        p.finish()
    }

    /// The model's walk covers every datum once, and a replayed loop computes
    /// what its unrolled form does.
    #[test]
    fn a_row_loop_reaches_every_datum_replayed_or_not() {
        let (a, b) = tiles();
        for policy in [LoopPolicy::Replay, LoopPolicy::Unrolled] {
            let mut v = Vector::new();
            v.put_tile(0, &a);
            v.put_tile(64, &b);
            v.run(&add_program(policy)).unwrap();
            let want: Vec<u32> = a.iter().zip(&b).map(|(x, y)| (f32::from_bits(*x) + f32::from_bits(*y)).to_bits()).collect();
            assert_eq!(v.tile(128), want, "{policy:?}");
            assert_eq!(v.rwc_dst, 0, "the loop leaves the counter cleared");
        }
    }

    /// Conditional execution: a scope's lanes, its complement, and nothing
    /// outside it.
    #[test]
    fn scopes_predicate_lanes() {
        let (a, _) = tiles();
        let mut p = Program::with_policy(LoopPolicy::Unrolled);
        p.for_each_row_group(64, |p, o| {
            p.load(LReg::L0, Format::Fp32, o);
            p.if_else(Cond::Lt0(LReg::L0), |p| p.mov(LReg::ZERO, LReg::L0), |p| p.neg(LReg::L0, LReg::L0));
            p.store(LReg::L0, Format::Fp32, 128 + o);
        });
        let mut v = Vector::new();
        v.put_tile(0, &a);
        v.run(&p.finish()).unwrap();
        let want: Vec<u32> = a.iter().map(|&x| if (x as i32) < 0 { 0 } else { x ^ 0x8000_0000 }).collect();
        assert_eq!(v.tile(128), want);
    }

    /// A program reading a constant another program wrote is refused: the
    /// interpreter starts `LReg[11..15]` unknown, so each program must write
    /// the constants it reads (concepts review G11). Written, it reads back
    /// in every lane, `SFPCONFIG` having taken lanes 0..8's value.
    #[test]
    fn a_constant_is_known_only_to_the_program_that_writes_it() {
        use tt_isa::sfpu::ConfigLReg;
        let mut p = Program::new();
        p.mov(ConfigLReg::L13.lreg(), LReg::L2);
        assert_eq!(Vector::new().run(&p.finish()), Err(InterpError::Unknown { at: 3, lreg: 13 }));
        let mut p = Program::new();
        p.constant(ConfigLReg::L13, 0x4049_0fdb);
        p.mov(ConfigLReg::L13.lreg(), LReg::L2);
        let mut v = Vector::new();
        v.run(&p.finish()).unwrap();
        assert_eq!(v.lreg[2], Some([0x4049_0fdb; 32]));
    }

    #[test]
    fn what_has_no_model_is_refused_by_name() {
        let mut v = Vector::new();
        let p = [tt_isa::isa::generated::encode::sfpcast(1, 2, 1).unwrap()];
        let e = v.run(&p).unwrap_err();
        assert!(e.to_string().contains("SFPCAST"), "{e}");
        let read8 = [tt_isa::isa::generated::encode::sfpmov(8, 0, 0).unwrap()];
        assert_eq!(Vector::new().run(&read8), Err(InterpError::Unknown { at: 0, lreg: 8 }));
    }
}
