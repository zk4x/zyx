//! Posted-write fences as API.
//!
//! [`Device::write`], [`Device::l1_write`] and [`Device::write32`] are *posted*:
//! when they return, the bytes have left the CPU, not landed. A read through the
//! same window is the only thing that says they have (a read does not pass a
//! posted write on the same path). That is harmless until another agent -- an
//! Ethernet transfer landing in the same buffer, a mover told to read it, a core
//! released into it -- acts on what the host just wrote. Then the *host's* late
//! write can land after the other agent's, or the other agent can start before it
//! (divergence row AA: 3 failures in 40 runs with no read-back, 6 in 60 with one
//! on the sender only, 0 in 60 with both).
//!
//! These are the writes that carry the rule so a caller cannot forget it. Each
//! issues its posted writes and then **exactly one** read-back, through the same
//! window and the same path as the writes, and returns only after that read has
//! come back:
//!
//! - [`Device::write32_fenced`], [`Device::write_fenced`] and [`FencedWrite`]: the
//!   general path (registers or memory). Addresses and lengths are dword
//!   multiples, because the read-back is one dword and a register may take only
//!   dword accesses; anything else is refused *before any write*.
//! - [`Device::l1_write_fenced`]: the bulk Tensix-L1 path, byte ranges allowed.
//!
//! The read-back is of the dword holding the last byte written -- the widest
//! single location that every earlier write on the path is ordered before. A
//! register that cannot be read, or whose read has side effects, is fenced by a
//! readable word of the same tile instead: [`FencedWrite::fence_at`].
//!
//! Not every write needs this. A write the host follows with a read through the
//! same window before anything else could act on it is already ordered. A hot
//! path whose only observer is a core the host starts *afterwards* stays
//! un-fenced: the start is a later host access, and every silicon gate for those
//! paths passes without a fence (if one ever does not, that is a finding to fence
//! there, not something this API assumes). The fence is for a *concurrent*
//! observer that needs no start from the host: an Ethernet transfer landing, a
//! mover told to read the range.

use super::super::{Device, Window};
use super::super::{Result, Transport, TransportError};
use super::{refuse_local_ram_aperture, refuse_non_l1};
use tt_isa::noc::{NocCoord, NocId};

/// The dword that holds the last byte of `[address, address + len)`.
fn last_word(address: u64, len: usize) -> u64 {
    (address + len as u64 - 1) & !3
}

fn dword_only(window: &Window, address: u64, len: usize) -> Result<()> {
    if address % 4 != 0 || len % 4 != 0 {
        return Err(TransportError::Misaligned {
            bar: window.kind().bar(),
            offset: address,
            len: len as u64,
            reason: "a fenced write is whole dwords (its read-back is one dword, and a \
                     register may accept nothing narrower)",
        });
    }
    Ok(())
}

/// A batch of posted writes to one tile, through one window, fenced by one
/// read-back. Build it, then [`commit`](FencedWrite::commit).
///
/// Use it where several words must all be in place before another agent looks --
/// a descriptor followed by its doorbell -- and one read-back covers the lot:
/// writes through one window are ordered with each other, so the read that
/// follows the last of them follows all of them.
#[must_use = "nothing is written until `commit`"]
pub struct FencedWrite<'w, N: NocId> {
    window: &'w Window,
    coord: NocCoord<N>,
    writes: Vec<(u64, Vec<u8>)>,
    fence_at: Option<u64>,
}

impl<'w, N: NocId> FencedWrite<'w, N> {
    pub fn new(window: &'w Window, coord: NocCoord<N>) -> Self {
        Self { window, coord, writes: Vec::new(), fence_at: None }
    }

    /// Queue `data` at `address` (dword aligned, whole dwords).
    pub fn bytes(mut self, address: u64, data: &[u8]) -> Self {
        self.writes.push((address, data.to_vec()));
        self
    }

    /// Queue one dword.
    pub fn word(self, address: u64, value: u32) -> Self {
        self.bytes(address, &value.to_le_bytes())
    }

    /// Read back the dword at `address` instead of the last word written: for a
    /// last write that is to a register that cannot be read back, or that reads
    /// with side effects. It must be in the same tile, and is read through the
    /// same window.
    pub fn fence_at(mut self, address: u64) -> Self {
        self.fence_at = Some(address);
        self
    }

    /// Check everything, write everything in order, read back once.
    ///
    /// Every refusal -- a foreign window, the local-data-RAM aperture, a
    /// misaligned write or fence word -- happens before the first write, so a
    /// refused batch changes nothing. A failed write returns its error and reads
    /// nothing back: there is no ordering to claim. An empty batch is a no-op
    /// with no traffic.
    pub fn commit<T: Transport>(self, dev: &mut Device<T>) -> Result<()> {
        dev.check_owned(self.window)?;
        for (address, data) in &self.writes {
            dword_only(self.window, *address, data.len())?;
            refuse_local_ram_aperture(*address, data.len())?;
        }
        if let Some(at) = self.fence_at {
            dword_only(self.window, at, 4)?;
            refuse_local_ram_aperture(at, 4)?;
        }
        let Some((last_at, last)) = self.writes.iter().rfind(|(_, d)| !d.is_empty()) else {
            return Ok(());
        };
        let fence = self.fence_at.unwrap_or_else(|| last_word(*last_at, last.len()));
        for (address, data) in &self.writes {
            if !data.is_empty() {
                dev.write_unchecked(self.window, self.coord, *address, data)?;
            }
        }
        let mut word = [0u8; 4];
        dev.read_unchecked(self.window, self.coord, fence, &mut word)
    }
}

impl<T: Transport> Device<T> {
    /// [`Device::write32`], then one read-back of the same dword: returns once
    /// the write has landed. `address` must be dword aligned.
    pub fn write32_fenced<N: NocId>(&mut self, window: &Window, coord: NocCoord<N>, address: u64, value: u32) -> Result<()> {
        FencedWrite::new(window, coord).word(address, value).commit(self)
    }

    /// [`Device::write`], then one read-back of the last dword written: returns
    /// once every byte has landed. `address` and `data.len()` must be dword
    /// multiples. For several writes under one read-back, use [`FencedWrite`].
    pub fn write_fenced<N: NocId>(&mut self, window: &Window, coord: NocCoord<N>, address: u64, data: &[u8]) -> Result<()> {
        FencedWrite::new(window, coord).bytes(address, data).commit(self)
    }

    /// [`Device::l1_write`] (the bulk path), then one bulk read-back of the dword
    /// holding the last byte: returns once every byte has landed. Any byte range
    /// inside a Tensix tile's L1 is allowed; the read-back is of the whole dword
    /// that contains the last byte.
    pub fn l1_write_fenced<N: NocId>(&mut self, window: &Window, coord: NocCoord<N>, address: u64, data: &[u8]) -> Result<()> {
        refuse_non_l1(coord, address, data.len())?;
        self.check_owned(window)?;
        if data.is_empty() {
            return Ok(());
        }
        self.write_memory(window, coord, address, data)?;
        let mut word = [0u8; 4];
        self.read_memory(window, coord, last_word(address, data.len()), &mut word)
    }

    /// A posted write of *memory* at any tile -- not only Tensix L1 -- and then
    /// one read-back of the dword holding its last byte. Byte ranges allowed; the
    /// caller has established that the range is memory (a read of the containing
    /// dword has no side effect there). For the Ethernet tiles' L1.
    pub(crate) fn write_range_fenced<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        data: &[u8],
    ) -> Result<()> {
        refuse_local_ram_aperture(address, data.len())?;
        self.check_owned(window)?;
        if data.is_empty() {
            return Ok(());
        }
        self.write_unchecked(window, coord, address, data)?;
        let mut word = [0u8; 4];
        self.read_unchecked(window, coord, last_word(address, data.len()), &mut word)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the fenced writes, against the windowing fake transport
    //! (`device::tests::FakeTransport`), which logs every data access in order.

    use super::super::tests::{c, device};
    use super::super::tlb::WindowKind;
    use super::*;

    fn tile() -> NocCoord<tt_isa::noc::Noc0> {
        c(3, 4)
    }

    /// `(is_write, tile address, len)` of every data access so far.
    type Log = Vec<(bool, u64, usize)>;

    fn reads(log: &Log) -> Log {
        log.iter().copied().filter(|e| !e.0).collect()
    }

    #[test]
    fn write32_fenced_is_one_write_then_one_read_of_the_same_dword() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.write32_fenced(&w, tile(), 0x1004, 0xDEAD_BEEF).unwrap();
        assert_eq!(
            d.transport().log,
            vec![(true, 0x1004, 4), (false, 0x1004, 4)],
            "the write, then exactly one read of the word just written"
        );
        assert_eq!(d.read32(&w, tile(), 0x1004).unwrap(), 0xDEAD_BEEF);
    }

    #[test]
    fn write_fenced_reads_back_the_last_dword_after_the_last_write() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let data: Vec<u8> = (0..16).collect();
        d.write_fenced(&w, tile(), 0x2000, &data).unwrap();
        assert_eq!(d.transport().log, vec![(true, 0x2000, 16), (false, 0x200C, 4)]);
        let mut back = [0u8; 16];
        d.read(&w, tile(), 0x2000, &mut back).unwrap();
        assert_eq!(&back[..], &data[..]);
    }

    #[test]
    fn a_transfer_split_across_windows_still_reads_back_once() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let size = WindowKind::TwoMib.size();
        d.write_fenced(&w, tile(), size - 8, &[0x5A; 16]).unwrap();
        assert_eq!(
            d.transport().log,
            vec![(true, size - 8, 8), (true, size, 8), (false, size + 4, 4)],
            "two write pieces, one read of the last dword"
        );
    }

    #[test]
    fn l1_write_fenced_takes_ragged_ranges_and_reads_the_dword_with_the_last_byte() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        // Bytes 0x3002..0x3008: the last byte is in the dword at 0x3004.
        d.l1_write_fenced(&w, tile(), 0x3002, &[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(d.transport().log, vec![(true, 0x3002, 6), (false, 0x3004, 4)]);
        let mut back = [0u8; 6];
        d.l1_read(&w, tile(), 0x3002, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn the_bulk_fence_refuses_what_the_bulk_path_refuses_without_traffic() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let before = d.traffic();
        // The ARC tile, a Tensix register, and a range crossing the end of L1.
        let end = tt_isa::tensix::L1_SIZE - 2;
        for (coord, addr) in [(c(8, 0), 0x1000), (tile(), 0xFFB1_21B0), (tile(), end)] {
            let e = d.l1_write_fenced(&w, coord, addr, &[0; 4]).unwrap_err();
            assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        }
        assert_eq!(d.traffic(), before, "nothing may be sent");
        assert!(d.transport().log.is_empty());
    }

    #[test]
    fn misaligned_fenced_writes_are_refused_before_any_traffic() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let before = d.traffic();
        // Unaligned address, a length that is not whole dwords, a ragged fence
        // word, and the second write of a batch (the first is fine).
        let results: Vec<Result<()>> = vec![
            d.write32_fenced(&w, tile(), 0x1002, 1),
            d.write_fenced(&w, tile(), 0x1000, &[1, 2, 3, 4, 5, 6]),
            d.write_fenced(&w, tile(), 0x1001, &[1, 2, 3, 4]),
            FencedWrite::new(&w, tile()).word(0x1000, 1).fence_at(0x1002).commit(&mut d),
            FencedWrite::new(&w, tile()).word(0x1000, 1).bytes(0x1010, &[9; 3]).commit(&mut d),
        ];
        for (i, r) in results.into_iter().enumerate() {
            let e = r.unwrap_err();
            assert!(matches!(e, TransportError::Misaligned { .. }), "case {i}: {e}");
        }
        assert_eq!(d.traffic(), before, "a refused batch changes nothing");
        assert!(d.transport().log.is_empty());
    }

    #[test]
    fn the_local_ram_aperture_is_refused_before_any_traffic() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let before = d.traffic();
        let e = d.write32_fenced(&w, tile(), 0xFFB1_4000, 1).unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        let e = FencedWrite::new(&w, tile()).word(0x1000, 1).fence_at(0xFFB1_4000).commit(&mut d).unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        assert_eq!(d.traffic(), before);
    }

    #[test]
    fn a_failed_write_reads_nothing_back() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.transport().fail_data_writes = true;
        let before = d.traffic();
        assert!(d.write_fenced(&w, tile(), 0x1000, &[0; 8]).is_err());
        assert!(d.l1_write_fenced(&w, tile(), 0x1000, &[0; 8]).is_err());
        assert!(d.write_range_fenced(&w, tile(), 0x1001, &[0; 3]).is_err());
        assert_eq!(d.traffic().read_calls, before.read_calls, "no read-back");
        assert!(reads(&d.transport().log).is_empty());
    }

    #[test]
    fn a_batch_is_ordered_and_fenced_once_at_the_last_word_or_the_named_one() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        FencedWrite::new(&w, tile()).word(0x4000, 1).bytes(0x4010, &[7; 8]).word(0x4004, 2).commit(&mut d).unwrap();
        assert_eq!(
            d.transport().log,
            vec![(true, 0x4000, 4), (true, 0x4010, 8), (true, 0x4004, 4), (false, 0x4004, 4)],
            "writes in call order, then one read of the last one's word"
        );
        d.transport().log.clear();
        // A doorbell that cannot be read: fence on a readable word instead.
        FencedWrite::new(&w, tile()).word(0x4000, 3).word(0x4100, 1).fence_at(0x4000).commit(&mut d).unwrap();
        assert_eq!(d.transport().log, vec![(true, 0x4000, 4), (true, 0x4100, 4), (false, 0x4000, 4)]);
    }

    #[test]
    fn an_empty_batch_is_no_traffic() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let before = d.traffic();
        FencedWrite::new(&w, tile()).commit(&mut d).unwrap();
        d.write_fenced(&w, tile(), 0x1000, &[]).unwrap();
        d.l1_write_fenced(&w, tile(), 0x1001, &[]).unwrap();
        assert_eq!(d.traffic(), before);
    }

    #[test]
    fn a_window_from_another_device_is_refused_before_any_write() {
        let (mut a, mut b) = (device(), device());
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let _wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let before = b.traffic();
        let e = b.write32_fenced(&wa, tile(), 0x1000, 1).unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        assert_eq!(b.traffic(), before);
        assert!(b.transport().log.is_empty());
    }
}
