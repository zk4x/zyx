//! Host-to-device TLB windows.
//!
//! The PCIe tile does not have NoC request initiators (`NoC/MemoryMap.md:60`); it
//! reaches the rest of the chip exclusively through 210 configurable TLB windows.
//! Each window maps a slice of a BAR onto a 64-bit address in some tile.
//!
//! Spec: `BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md`, cross-checked against
//! the worked example in `BlackholeA0/EthernetTile/Samples/ethdump/ethdump.c`.

use tt_isa::noc::{NocCoord, NocId};

use super::{Bar, Transport, TransportError};

/// Total number of configurable windows (`HostToDeviceTLBs.md:3`).
pub const NUM_WINDOWS: u16 = 210;
/// Windows 0..=201 are 2 MiB and live in BAR0.
pub const NUM_2MIB_WINDOWS: u16 = 202;
/// Window 201 is reserved for the kernel driver (`HostToDeviceTLBs.md:9`).
pub const KERNEL_RESERVED_WINDOW: u16 = 201;

pub const WINDOW_2MIB_SIZE: u64 = 2 * 1024 * 1024;
pub const WINDOW_4GIB_SIZE: u64 = 4 * 1024 * 1024 * 1024;

/// Base of the window configuration array in BAR0 (`HostToDeviceTLBs.md:14`).
pub const CONFIG_BASE: u64 = 0x1FC0_0000;

/// Which of the two window geometries a given index has.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WindowKind {
    /// 2 MiB, in BAR0. Indices 0..=201.
    TwoMib,
    /// 4 GiB, in BAR4. Indices 202..=209.
    FourGib,
}

impl WindowKind {
    pub const fn size(self) -> u64 {
        match self {
            WindowKind::TwoMib => WINDOW_2MIB_SIZE,
            WindowKind::FourGib => WINDOW_4GIB_SIZE,
        }
    }

    pub const fn bar(self) -> Bar {
        match self {
            WindowKind::TwoMib => Bar::Bar0,
            WindowKind::FourGib => Bar::Bar4,
        }
    }

    /// Number of low address bits taken from the offset within the window, and
    /// therefore the shift applied to `local_offset`.
    const fn offset_bits(self) -> u32 {
        match self {
            WindowKind::TwoMib => 21,
            WindowKind::FourGib => 32,
        }
    }
}

/// Geometry of a window index.
pub const fn window_kind(index: u16) -> Option<WindowKind> {
    if index < NUM_2MIB_WINDOWS {
        Some(WindowKind::TwoMib)
    } else if index < NUM_WINDOWS {
        Some(WindowKind::FourGib)
    } else {
        None
    }
}

/// Byte offset of a window's aperture within its BAR.
pub const fn window_bar_offset(index: u16) -> Option<u64> {
    match window_kind(index) {
        Some(WindowKind::TwoMib) => Some(index as u64 * WINDOW_2MIB_SIZE),
        Some(WindowKind::FourGib) => Some((index - NUM_2MIB_WINDOWS) as u64 * WINDOW_4GIB_SIZE),
        None => None,
    }
}

/// Byte offset in BAR0 of a window's first configuration dword.
pub const fn window_config_offset(index: u16) -> Option<u64> {
    if index < NUM_WINDOWS {
        Some(CONFIG_BASE + index as u64 * 12)
    } else {
        None
    }
}

/// Response-ordering mode (`HostToDeviceTLBs.md:31`).
///
/// This is a 2-bit enum on Blackhole where Wormhole had separate bits, so Wormhole
/// assumptions about this field do not carry over.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum Ordering {
    #[default]
    Default = 0,
    /// What `ethdump.c:239` selects, and a reasonable default for correctness-first
    /// work: it is the most conservative of the four.
    StrictAxi = 1,
    PostedWrites = 2,
    CountedWrites = 3,
}

/// Static virtual-channel class (`HostToDeviceTLBs.md:36`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VcClass {
    Unicast0 = 0b00,
    Unicast1 = 0b01,
    Multicast = 0b10,
}

/// What a window points at.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Target<N: NocId> {
    /// A single tile.
    Unicast(NocCoord<N>),
    /// A rectangle of tiles. **Write-only**: a multicast read is meaningless (there
    /// would be many responses) and the hardware does not define one.
    Multicast { start: NocCoord<N>, end: NocCoord<N> },
}

/// Configuration of one TLB window.
///
/// Encodes to three dwords per `HostToDeviceTLBs.md:22-38`. Note that the 2 MiB and
/// 4 GiB layouts are *not* related by a shift — the field offsets differ
/// irregularly — so they are encoded by separate arms rather than a common formula.
#[derive(Copy, Clone, Debug)]
pub struct TlbConfig<N: NocId> {
    /// Device address this window maps, which must be aligned to the window size.
    /// The low bits come from the offset within the window.
    pub base_address: u64,
    pub target: Target<N>,
    pub ordering: Ordering,
    /// `NOC_CMD_VC_STATIC`.
    pub static_vc: bool,
    pub static_vc_buddy: bool,
    pub static_vc_class: Option<VcClass>,
}

/// Why a window configuration could not be encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlbConfigError {
    /// `base_address` is not a multiple of the window size.
    Unaligned { address: u64, window_size: u64 },
}

impl core::fmt::Display for TlbConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TlbConfigError::Unaligned { address, window_size } => {
                write!(f, "device address {address:#x} is not aligned to the {window_size:#x}-byte window size")
            }
        }
    }
}

impl std::error::Error for TlbConfigError {}

/// Set `count` bits of `value` starting at `first_bit` within a 96-bit little-endian
/// field, expressed as three dwords.
fn place(words: &mut [u32; 3], first_bit: u32, count: u32, value: u64) {
    debug_assert!(first_bit + count <= 96);
    debug_assert!(count == 64 || value < (1u64 << count), "value does not fit in {count} bits");
    let mut bits = value;
    let mut bit = first_bit;
    let mut remaining = count;
    while remaining > 0 {
        let word = (bit / 32) as usize;
        let shift = bit % 32;
        let here = remaining.min(32 - shift);
        let mask = if here == 32 { u32::MAX } else { ((1u32 << here) - 1) << shift };
        words[word] = (words[word] & !mask) | (((bits as u32) << shift) & mask);
        bits >>= here;
        bit += here;
        remaining -= here;
    }
}

impl<N: NocId> TlbConfig<N> {
    /// A minimal unicast window: strict ordering, no static VC.
    pub fn unicast(base_address: u64, target: NocCoord<N>) -> Self {
        TlbConfig {
            base_address,
            target: Target::Unicast(target),
            ordering: Ordering::StrictAxi,
            static_vc: false,
            static_vc_buddy: false,
            static_vc_class: None,
        }
    }

    /// Encode to the three configuration dwords.
    ///
    /// The `linked` field (bit 72 / 61) is deliberately never set. The
    /// specification states it is *never* safe to set from the host, because the
    /// kernel driver may use its own window at any time
    /// (`HostToDeviceTLBs.md:32`), so it is not exposed as an option.
    pub fn encode(&self, kind: WindowKind) -> Result<[u32; 3], TlbConfigError> {
        let window_size = kind.size();
        if self.base_address % window_size != 0 {
            return Err(TlbConfigError::Unaligned { address: self.base_address, window_size });
        }
        let shift = kind.offset_bits();
        let local_offset = self.base_address >> shift;

        // Field positions differ between the two geometries in ways that are not a
        // uniform shift, so each is spelled out against the spec table.
        let (offset_bits, f_x_end, f_y_end, f_x_start, f_y_start, f_noc, f_mcast, f_ord, f_svc, f_buddy, f_class) = match kind {
            //           local  x_end y_end x_st  y_st  noc  mc   ord  svc  buddy class
            WindowKind::TwoMib => (43u32, 43u32, 49u32, 55u32, 61u32, 67u32, 69u32, 70u32, 73u32, 75u32, 76u32),
            WindowKind::FourGib => (32, 32, 38, 44, 50, 56, 58, 59, 62, 64, 65),
        };

        // `local_offset` is exactly wide enough to span the 64-bit device address
        // space in both geometries -- 43 + 21 and 32 + 32 both make 64 -- so a
        // window-aligned address always fits and there is no overflow case.
        debug_assert_eq!(offset_bits + shift, 64);

        let mut words = [0u32; 3];
        place(&mut words, 0, offset_bits, local_offset);

        match self.target {
            Target::Unicast(c) => {
                place(&mut words, f_x_end, 6, c.x() as u64);
                place(&mut words, f_y_end, 6, c.y() as u64);
            }
            Target::Multicast { start, end } => {
                place(&mut words, f_x_end, 6, end.x() as u64);
                place(&mut words, f_y_end, 6, end.y() as u64);
                place(&mut words, f_x_start, 6, start.x() as u64);
                place(&mut words, f_y_start, 6, start.y() as u64);
                place(&mut words, f_mcast, 1, 1);
            }
        }

        place(&mut words, f_noc, 1, N::INDEX as u64);
        place(&mut words, f_ord, 2, self.ordering as u64);
        place(&mut words, f_svc, 1, self.static_vc as u64);
        place(&mut words, f_buddy, 1, self.static_vc_buddy as u64);
        if let Some(class) = self.static_vc_class {
            place(&mut words, f_class, 2, class as u64);
        }
        Ok(words)
    }

    /// Is this window readable?
    ///
    /// Multicast windows are write-only: a read would solicit a response from every
    /// tile in the rectangle.
    pub fn is_readable(&self) -> bool {
        matches!(self.target, Target::Unicast(_))
    }
}

/// Write a window's configuration, three dwords at a time.
///
/// The configuration registers are write-only — reading one back is an error on the
/// simulator and undefined on silicon — so callers that need to know the current
/// configuration must shadow it. [`super::Device`] does.
pub fn write_config<T: Transport, N: NocId>(transport: &mut T, index: u16, config: &TlbConfig<N>) -> super::Result<()> {
    let kind = window_kind(index).ok_or(TransportError::OutOfBounds { bar: Bar::Bar0, offset: index as u64, len: 0 })?;
    let words = config.encode(kind).map_err(|e| TransportError::Misaligned {
        bar: kind.bar(),
        offset: config.base_address,
        len: kind.size(),
        reason: match e {
            TlbConfigError::Unaligned { .. } => "device address is not window-aligned",
        },
    })?;
    let base = window_config_offset(index).expect("index was validated above");
    for (i, word) in words.iter().enumerate() {
        transport.bar_write32(Bar::Bar0, base + 4 * i as u64, *word)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tt_isa::noc::{Noc0, Noc1};

    fn c(x: u8, y: u8) -> NocCoord<Noc0> {
        NocCoord::new(x, y).unwrap()
    }

    #[test]
    fn geometry_matches_the_spec_inventory() {
        assert_eq!(window_kind(0), Some(WindowKind::TwoMib));
        assert_eq!(window_kind(201), Some(WindowKind::TwoMib));
        assert_eq!(window_kind(202), Some(WindowKind::FourGib));
        assert_eq!(window_kind(209), Some(WindowKind::FourGib));
        assert_eq!(window_kind(210), None);

        // 202 x 2 MiB = 404 MiB of BAR0, and 8 x 4 GiB = all 32 GiB of BAR4.
        assert_eq!(window_bar_offset(201).unwrap() + WINDOW_2MIB_SIZE, 404 * 1024 * 1024);
        assert_eq!(window_bar_offset(209).unwrap() + WINDOW_4GIB_SIZE, Bar::Bar4.size());
    }

    #[test]
    fn config_array_ends_where_the_strided_array_begins() {
        // windows[210] of 12 bytes each starts at 0x1FC0_0000, so `strided[32]`
        // begins at 0x1FC0_09D8 and the whole structure ends at 0x1FC0_0A58 --
        // exactly the bounds ethdump.c:145-147 hardcodes.
        assert_eq!(window_config_offset(0), Some(0x1FC0_0000));
        assert_eq!(CONFIG_BASE + NUM_WINDOWS as u64 * 12, 0x1FC0_09D8);
        assert_eq!(0x1FC0_09D8 + 32 * 4, 0x1FC0_0A58);
    }

    /// The decisive cross-check: reproduce ethdump's own bit manipulation.
    ///
    /// `set_tlb_addr` (ethdump.c:270-290) computes word0 = addr >> 21, and merges
    /// `addr >> 53` into word1 while preserving `c1 & 0xfffff800` -- i.e. the low 11
    /// bits of word1 are the top of a 43-bit local_offset. `set_tlb_xy` (`:260-268`)
    /// places x at word1 bit 11 and y at word1 bit 17.
    #[test]
    fn two_mib_encoding_matches_ethdump() {
        let addr = 0x0000_0012_3456_0000u64 & !(WINDOW_2MIB_SIZE - 1);
        let cfg = TlbConfig::unicast(addr, c(19, 24));
        let w = cfg.encode(WindowKind::TwoMib).unwrap();

        assert_eq!(w[0], (addr >> 21) as u32, "word0 is the low 32 bits of addr >> 21");
        assert_eq!(w[1] & 0x7FF, ((addr >> 53) & 0x7FF) as u32, "word1[10:0] continues it");
        assert_eq!((w[1] >> 11) & 0x3F, 19, "x_end at word1 bit 11");
        assert_eq!((w[1] >> 17) & 0x3F, 24, "y_end at word1 bit 17");
        // ethdump writes exactly (1 << 6) into word2 for TLB_CFG_STRICT_AXI.
        assert_eq!(w[2], 1 << 6, "ordering=StrictAxi and nothing else set");
    }

    #[test]
    fn ordering_field_is_two_bits_at_seventy() {
        let mk = |o| {
            let mut cfg = TlbConfig::unicast(0, c(1, 2));
            cfg.ordering = o;
            cfg.encode(WindowKind::TwoMib).unwrap()[2]
        };
        assert_eq!(mk(Ordering::Default) >> 6 & 3, 0);
        assert_eq!(mk(Ordering::StrictAxi) >> 6 & 3, 1);
        assert_eq!(mk(Ordering::PostedWrites) >> 6 & 3, 2);
        assert_eq!(mk(Ordering::CountedWrites) >> 6 & 3, 3);
    }

    #[test]
    fn noc_selection_is_encoded_from_the_coordinate_type() {
        // The NoC a window targets is carried by the coordinate's type parameter,
        // so it cannot disagree with the coordinate it was computed for.
        let n0 = TlbConfig::unicast(0, c(1, 2)).encode(WindowKind::TwoMib).unwrap();
        assert_eq!((n0[2] >> 3) & 1, 0, "noc_sel at bit 67 = word2 bit 3");

        let n1: TlbConfig<Noc1> = TlbConfig::unicast(0, NocCoord::<Noc1>::new(1, 2).unwrap());
        let w = n1.encode(WindowKind::TwoMib).unwrap();
        assert_eq!((w[2] >> 3) & 1, 1);
    }

    #[test]
    fn four_gib_layout_is_not_a_shifted_two_mib_layout() {
        // The two layouts differ irregularly: in the 4 GiB form `ordering` sits at
        // bit 59 and `linked` at 61, leaving bit 60 unaccounted for. Encoding one
        // with the other's field offsets would silently misplace every field, so
        // this pins the 4 GiB positions independently.
        let cfg = TlbConfig::unicast(0, c(0x2A, 0x15));
        let w = cfg.encode(WindowKind::FourGib).unwrap();
        assert_eq!(w[1] & 0x3F, 0x2A, "x_end at bit 32 = word1 bit 0");
        assert_eq!((w[1] >> 6) & 0x3F, 0x15, "y_end at bit 38 = word1 bit 6");
        assert_eq!((w[1] >> 27) & 3, Ordering::StrictAxi as u32, "ordering at bit 59");

        // And the same config encoded for a 2 MiB window must differ.
        let two = cfg.encode(WindowKind::TwoMib).unwrap();
        assert_ne!(w, two);
    }

    #[test]
    fn address_must_be_window_aligned() {
        let cfg = TlbConfig::unicast(0x1000, c(1, 1));
        assert!(matches!(cfg.encode(WindowKind::TwoMib), Err(TlbConfigError::Unaligned { .. })));
        // The same address is fine once aligned.
        assert!(TlbConfig::unicast(0, c(1, 1)).encode(WindowKind::TwoMib).is_ok());
    }

    #[test]
    fn local_offset_spans_the_whole_64_bit_address_space() {
        // 43 bits of local_offset plus 21 bits of window offset is 64; so is 32 plus
        // 32. Both geometries therefore reach any device address, and neither can
        // overflow. Worth pinning: the natural assumption is that the narrower
        // 32-bit field addresses less memory, and it does not.
        for (kind, size) in [
            (WindowKind::TwoMib, WINDOW_2MIB_SIZE),
            (WindowKind::FourGib, WINDOW_4GIB_SIZE),
        ] {
            let highest = u64::MAX - (size - 1);
            let w = TlbConfig::unicast(highest, c(1, 1)).encode(kind).unwrap();
            let shift = if kind == WindowKind::TwoMib { 21 } else { 32 };
            let bits = if kind == WindowKind::TwoMib { 43 } else { 32 };
            let recovered = (w[0] as u64) | (((w[1] as u64) << 32) & ((1u64 << bits) - 1));
            assert_eq!(recovered << shift, highest, "{kind:?} must reach the top of the space");
        }
    }

    #[test]
    fn multicast_sets_the_rectangle_and_is_not_readable() {
        let cfg = TlbConfig {
            base_address: 0,
            target: Target::Multicast { start: c(1, 2), end: c(4, 5) },
            ordering: Ordering::StrictAxi,
            static_vc: false,
            static_vc_buddy: false,
            static_vc_class: None,
        };
        let w = cfg.encode(WindowKind::TwoMib).unwrap();
        assert_eq!((w[1] >> 11) & 0x3F, 4, "x_end");
        assert_eq!((w[1] >> 17) & 0x3F, 5, "y_end");
        assert_eq!((w[1] >> 23) & 0x3F, 1, "x_start at bit 55 = word1 bit 23");
        // y_start spans the word1/word2 boundary: bits 61..66.
        let y_start = ((w[1] >> 29) & 0x7) | ((w[2] & 0x7) << 3);
        assert_eq!(y_start, 2, "y_start straddles words 1 and 2");
        assert_eq!((w[2] >> 5) & 1, 1, "mcast at bit 69 = word2 bit 5");
        assert!(!cfg.is_readable());
        assert!(TlbConfig::unicast(0, c(1, 1)).is_readable());
    }

    #[test]
    fn linked_is_never_set() {
        // It is not an option on the type, so the only way to check is that the bit
        // stays clear for every configuration we can express.
        let cfg = TlbConfig {
            base_address: 0,
            target: Target::Multicast { start: c(0, 0), end: c(63, 63) },
            ordering: Ordering::CountedWrites,
            static_vc: true,
            static_vc_buddy: true,
            static_vc_class: Some(VcClass::Multicast),
        };
        let w = cfg.encode(WindowKind::TwoMib).unwrap();
        assert_eq!((w[2] >> 8) & 1, 0, "linked (bit 72) must stay clear");
        let w = cfg.encode(WindowKind::FourGib).unwrap();
        assert_eq!((w[1] >> 29) & 1, 0, "linked (bit 61) must stay clear");
    }
}
