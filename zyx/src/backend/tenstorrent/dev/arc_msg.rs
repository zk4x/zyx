//! Sending messages to the ARC firmware.
//!
//! The protocol UMD uses (`blackhole_arc_message_queue.cpp`): the ARC publishes a
//! queue control block whose CSM address sits in `SCRATCH_RAM_11`; each queue is
//! a header, a ring of request entries and a ring of response entries. A request
//! is written at the request write pointer, the pointer is advanced (modulo twice
//! the ring size, so full and empty are distinguishable), the firmware interrupt
//! is raised, and the response is popped the same way from the other ring.
//!
//! Every address the firmware hands out is bounds-checked against the CSM before
//! it is used, exactly as the telemetry reader does: an ARC read outside the CSM
//! is a NoC read to an unmapped address, which is a hang, not an error.

use std::time::{Duration, Instant};

use tt_isa::arc::{self, queue};

use super::device::{Device, Window};
use super::transport::{Result, Transport, TransportError};

fn arc_error(what: String) -> TransportError {
    TransportError::Io(std::io::Error::other(format!("ARC message: {what}")))
}

impl<T: Transport> Device<T> {
    /// Send `code` with up to seven `args` to the ARC's application queue and
    /// wait for its answer. Returns the response's upper half-word (UMD's "exit
    /// code") and the seven return values.
    pub fn arc_message(
        &mut self,
        window: &Window,
        code: u32,
        args: &[u32],
        timeout: Duration,
    ) -> Result<(u32, [u32; 7])> {
        if args.len() > 7 {
            return Err(arc_error(format!(
                "{} arguments; the ARC takes at most 7",
                args.len()
            )));
        }
        let tile = arc::arc_tile();
        let control = self.read32(window, tile, queue::CONTROL_PTR)? as u64;
        if !arc::is_within_csm(control, 8) {
            return Err(arc_error(format!(
                "queue control block at {control:#x} is outside the CSM"
            )));
        }
        let base_addr = self.read32(window, tile, control)? as u64;
        let entries = (self.read32(window, tile, control + 4)? & 0xFF) as u64;
        if entries == 0 {
            return Err(arc_error(
                "the firmware publishes zero queue entries".into(),
            ));
        }
        let queue_bytes = (queue::HEADER_WORDS + 2 * entries * queue::ENTRY_WORDS) * 4;
        let base = base_addr + queue::APPLICATION * queue_bytes;
        if !arc::is_within_csm(base, queue_bytes) {
            return Err(arc_error(format!(
                "application queue at {base:#x} is outside the CSM"
            )));
        }
        let word = |i: u64| base + i * 4;
        let deadline = Instant::now() + timeout;

        // Wait for room, then write the request.
        let wptr = self.read32(window, tile, word(queue::REQUEST_WPTR))? as u64;
        loop {
            let rptr = self.read32(window, tile, word(queue::REQUEST_RPTR))? as u64;
            if (wptr as i64 - rptr as i64).unsigned_abs() % (2 * entries) != entries {
                break;
            }
            if Instant::now() >= deadline {
                return Err(arc_error("the request queue stayed full".into()));
            }
        }
        let mut request = [0u32; 8];
        request[0] = code;
        request[1..=args.len()].copy_from_slice(args);
        let slot = queue::HEADER_WORDS + (wptr % entries) * queue::ENTRY_WORDS;
        for (i, w) in request.iter().enumerate() {
            self.write32(window, tile, word(slot + i as u64), *w)?;
        }
        self.write32(
            window,
            tile,
            word(queue::REQUEST_WPTR),
            ((wptr + 1) % (2 * entries)) as u32,
        )?;
        self.write32(window, tile, queue::FW_INT, queue::FW_INT_VAL)?;

        // Wait for the response.
        let rptr = self.read32(window, tile, word(queue::RESPONSE_RPTR))? as u64;
        loop {
            let resp_wptr = self.read32(window, tile, word(queue::RESPONSE_WPTR))? as u64;
            if resp_wptr != rptr {
                break;
            }
            if Instant::now() >= deadline {
                return Err(arc_error(format!("no response to message {code:#x}")));
            }
        }
        let slot = queue::HEADER_WORDS + (entries + rptr % entries) * queue::ENTRY_WORDS;
        let mut response = [0u32; 8];
        for (i, w) in response.iter_mut().enumerate() {
            *w = self.read32(window, tile, word(slot + i as u64))?;
        }
        self.write32(
            window,
            tile,
            word(queue::RESPONSE_RPTR),
            ((rptr + 1) % (2 * entries)) as u32,
        )?;

        let status = response[0] & 0xFF;
        if status >= queue::RESPONSE_OK_LIMIT {
            return Err(arc_error(format!(
                "message {code:#x} failed with status {status:#x}"
            )));
        }
        let mut values = [0u32; 7];
        values.copy_from_slice(&response[1..]);
        Ok((response[0] >> 16, values))
    }

    /// Put the chip at its busy operating point (`AICLK_GO_BUSY`), or back at
    /// idle (`AICLK_GO_LONG_IDLE`), as UMD does when it opens and closes a chip.
    ///
    /// Not optional for compute. At the idle point the Matrix Unit's reads of
    /// `SrcA`/`SrcB` drop or misplace datums in chip-specific column pairs --
    /// `MVMUL`, `ELWADD` and `MOV*2D` all return wrong values, deterministically
    /// per chip -- while the SFPU, `UnpackToDst` and the packer are unaffected.
    /// Measured on both p150a cards (`docs/learnings/ttsim-divergence.md` row 48).
    /// A no-op on the simulator, which has no ARC firmware.
    ///
    /// Waits for the transition to finish: the ARC acknowledges at once, but
    /// AICLK and VCORE take tens of milliseconds to reach the busy point
    /// (measured 30-66 ms on p150a, `silicon_measure::m33`), and compute issued
    /// in that window runs at the idle point -- which is the failure this exists
    /// to prevent. Returns the settled AICLK in MHz.
    pub fn set_busy(&mut self, window: &Window, busy: bool) -> Result<Option<u32>> {
        if self.transport().is_simulated() {
            return Ok(None);
        }
        let table = super::telemetry::TelemetryTable::read(self, window)?;
        let sample = |dev: &mut Self| -> Result<(Option<u32>, Option<u32>)> {
            Ok((
                table.read_tag(dev, window, arc::tag::AICLK)?,
                table.read_tag(dev, window, arc::tag::VCORE)?,
            ))
        };
        let before = sample(self)?;
        let code = if busy {
            arc::msg::AICLK_GO_BUSY
        } else {
            arc::msg::AICLK_GO_LONG_IDLE
        };
        self.arc_message(window, code, &[], Duration::from_secs(1))?;

        // Settled means: changed from before (or 200 ms passed, for a chip that
        // was already there), then three consecutive samples 5 ms apart agree.
        let start = Instant::now();
        let mut recent = Vec::new();
        loop {
            let now = sample(self)?;
            recent.push(now);
            let changed = now != before || start.elapsed() >= Duration::from_millis(200);
            let n = recent.len();
            if changed && n >= 3 && recent[n - 1] == recent[n - 2] && recent[n - 2] == recent[n - 3]
            {
                return Ok(now.0);
            }
            if start.elapsed() >= Duration::from_secs(2) {
                return Err(arc_error(format!(
                    "AICLK/VCORE did not settle after message {code:#x}: last {now:?}"
                )));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
