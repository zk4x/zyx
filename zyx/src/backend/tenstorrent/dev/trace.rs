//! Reading a tile's debug timestamper from the host.
//!
//! `TensixTile/DebugTimestamper.md` (identical to Wormhole's): a 64-bit counter
//! that advances every cycle, and an event stream that appends one record to a
//! buffer in L1 per store to `RISCV_DEBUG_REG_TIMESTAMP`. The firmware makes the
//! stores (`tt_firmware::corpus`); this configures the buffer, and decodes it.
//!
//! One store per event and no software timestamp read, which is what makes it
//! strictly better than per-core `mcycle` for lining up what three cores did:
//! every event is stamped by the same counter.

use tt_isa::noc::{NocCoord, NocId};
use tt_isa::tensix::timestamper as ts;

use super::{Device, Result, Transport, TransportError, Window};

/// One event from the stream: the 29-bit token the firmware wrote, and the
/// counter when it did.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TraceEvent {
    pub token: u32,
    pub cycles: u64,
}

impl<T: Transport> Device<T> {
    /// Point the tile's timestamper stream at `bytes` of L1 from `buffer`, and
    /// start it empty.
    ///
    /// Buffer 0 only; buffer 1 is disabled so a full buffer 0 is an overflow
    /// rather than a silent switch. The stream reset is sticky and has to be
    /// pulsed (`CntlWrite`), and only a status write with the `full` bit set
    /// returns the write position to zero (`StatusWrite`).
    pub fn configure_trace<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
        buffer: u64,
        bytes: u64,
    ) -> Result<()> {
        if buffer % ts::EVENT_BYTES != 0 || bytes < ts::EVENT_BYTES {
            return Err(TransportError::Hazard {
                address: buffer,
                reason: "a timestamper buffer is whole 16-byte units, and at least one",
            });
        }
        self.write32(window, tile, ts::CNTL, 1 << 31)?;
        self.write32(window, tile, ts::BUF0_START, (buffer / 16) as u32)?;
        self.write32(
            window,
            tile,
            ts::BUF0_END,
            ((buffer + bytes) / 16 - 1) as u32,
        )?;
        // Clear buffer 0's `full` (which also zeroes its position) and
        // `overflow`, then release the reset with buffer 0 alone valid.
        self.write32(window, tile, ts::STATUS, (1 << 0) | (1 << 4))?;
        self.write32(window, tile, ts::CNTL, 1)
    }

    /// The events recorded since [`Device::configure_trace`], in order.
    ///
    /// Refuses a stream that overflowed, rather than returning the part that
    /// fitted as though it were all of it.
    pub fn read_trace<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
        buffer: u64,
    ) -> Result<Vec<TraceEvent>> {
        let status = self.read32(window, tile, ts::STATUS)?;
        if ts::buf0_overflowed(status) {
            return Err(TransportError::Hazard {
                address: buffer,
                reason: "the timestamper buffer overflowed; events were dropped",
            });
        }
        let n = ts::buf0_position(status) as usize;
        let mut bytes = vec![0u8; n * ts::EVENT_BYTES as usize];
        self.read(window, tile, buffer, &mut bytes)?;
        Ok(bytes
            .chunks_exact(ts::EVENT_BYTES as usize)
            .map(|e| {
                let word = |i: usize| u32::from_le_bytes([e[i], e[i + 1], e[i + 2], e[i + 3]]);
                TraceEvent {
                    token: word(0) >> 3,
                    cycles: u64::from(word(4)) | (u64::from(word(8)) << 32),
                }
            })
            .collect())
    }

    /// The tile's 64-bit cycle counter, read with the documented retry loop
    /// for when other agents may be reading it too.
    pub fn wall_clock<N: NocId>(&mut self, window: &Window, tile: NocCoord<N>) -> Result<u64> {
        loop {
            let hi = self.read32(window, tile, ts::WALL_CLOCK_L_PLUS_4)?;
            let lo = self.read32(window, tile, ts::WALL_CLOCK_L)?;
            if self.read32(window, tile, ts::WALL_CLOCK_L_PLUS_4)? == hi {
                return Ok(u64::from(lo) | (u64::from(hi) << 32));
            }
        }
    }
}
