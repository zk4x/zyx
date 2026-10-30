//! Reading the ARC's telemetry table: what this particular chip is.
//!
//! The one query that has to happen before anything else. Which Tensix columns a
//! chip has is a per-ASIC fact, it cannot be discovered by probing — probing a
//! fused-off tile hangs the NoC, see [`tt_isa::arc`] — and the loaded driver will
//! not answer it. `TENSTORRENT_IOCTL_GET_HARVESTING` exists in the ABI but is an
//! unimplemented stub in 2.11.0 (`chardev.c:786-787`: a bare `break`, no handler,
//! and no struct for it in `ioctl.h`), and the driver's telemetry cache is sparse
//! — `tt_telemetry_probe` records addresses only for tags that back a sysfs or
//! hwmon attribute (`telemetry.c:42-77`), which the harvesting tags do not.
//!
//! So we read the table ourselves, the same way the driver does
//! (`blackhole.c:470-529`): a handful of dwords from the ARC's CSM over the NoC.
//! Every one of those reads goes to the ARC tile, which is not harvestable and
//! whose coordinate is translation-invariant, so the bootstrap is safe even though
//! we do not yet know the translation state or the harvesting mask.

use tt_isa::arc;
use tt_isa::noc::grid::Tensix;

use super::device::{Device, Window};
use super::transport::{Result, Transport, TransportError};

fn malformed(what: &'static str) -> TransportError {
    TransportError::Io(std::io::Error::other(format!(
        "the ARC telemetry table is malformed: {what}. \
         Refusing to derive the Tensix grid from it -- a wrong grid puts an access \
         on a fused-off tile, which hangs the NoC and drops the PCIe link."
    )))
}

/// The chip's own account of itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChipTelemetry {
    /// Which Tensix tiles this chip has.
    pub tensix: Tensix,
    /// Does the NoC translate coordinates?
    ///
    /// Worth carrying next to `tensix`, because the two are only jointly
    /// meaningful: [`Tensix`] is derived assuming translated space, where
    /// `NoC/Coordinates.md:54` guarantees harvested columns sit at maximal X. With
    /// translation off, the surviving columns are wherever the fuses left them and
    /// the count alone does not identify them.
    pub noc_translation: bool,
    /// `ENABLED_TENSIX_COL` exactly as firmware published it.
    ///
    /// Kept raw so a caller can report it. The interpretation in
    /// [`Tensix::from_enabled_column_mask`] is reasoned from the specification
    /// rather than from a published bit layout, so the evidence for it should
    /// survive into the log rather than being reduced to a tile count.
    pub enabled_tensix_col_raw: u32,
    /// `HARVESTING_STATE`, raw. Field layout unpublished; a cross-check only.
    pub harvesting_state_raw: Option<u32>,
    /// `TIMER_HEARTBEAT`, raw. Changes while the ARC is alive.
    pub heartbeat: Option<u32>,
}

/// The telemetry table's tag directory, resolved to CSM addresses.
///
/// Walking the table costs one uncached MMIO round trip per entry and there are
/// on the order of fifty, so it is walked once and kept.
pub struct TelemetryTable {
    entries: Vec<(u16, u64)>,
}

impl TelemetryTable {
    /// Walk the ARC's telemetry directory.
    ///
    /// Mirrors `blackhole_populate_telemetry_cache` (`blackhole.c:470-529`),
    /// including its bounds checks: every address comes from firmware, and an
    /// out-of-CSM address is not a bad value but a NoC read to an unmapped ARC
    /// address.
    pub fn read<T: Transport>(dev: &mut Device<T>, window: &Window) -> Result<Self> {
        let arc = arc::arc_tile();

        let base = dev.read32(window, arc, arc::TELEMETRY_PTR)? as u64;
        let data = dev.read32(window, arc, arc::TELEMETRY_DATA)? as u64;

        // The header is two dwords: a version, then an entry count.
        if !arc::is_within_csm(base, 8) || !arc::is_within_csm(data, 4) {
            return Err(malformed("the header or data pointer is outside the CSM"));
        }

        let version = dev.read32(window, arc, base)?;
        let major = (version >> 16) & 0xFF;
        if major > arc::TELEMETRY_MAX_MAJOR_VERSION {
            return Err(malformed(
                "the table declares a major version this code does not know",
            ));
        }

        let num_entries = dev.read32(window, arc, base + 4)?;
        let tags = base + 8;
        if num_entries > arc::TELEMETRY_MAX_ENTRIES
            || !arc::is_within_csm(tags, num_entries as u64 * 4)
        {
            return Err(malformed("the entry count does not fit the CSM"));
        }

        let mut entries = Vec::with_capacity(num_entries as usize);
        for i in 0..num_entries as u64 {
            let entry = dev.read32(window, arc, tags + i * 4)?;
            let tag_id = (entry & 0xFFFF) as u16;
            let offset = (entry >> 16) & 0xFFFF;
            let address = data + offset as u64 * 4;
            // The driver skips a bad entry rather than failing the whole table
            // (`blackhole.c:517-521`); one unusable tag should not cost us the
            // ones we need.
            if arc::is_within_csm(address, 4) {
                entries.push((tag_id, address));
            }
        }
        Ok(TelemetryTable { entries })
    }

    /// The CSM address of `tag`, if the chip publishes it.
    pub fn address_of(&self, tag: u16) -> Option<u64> {
        self.entries
            .iter()
            .find(|&&(id, _)| id == tag)
            .map(|&(_, addr)| addr)
    }

    /// Read `tag`, or `None` if this chip does not publish it.
    pub fn read_tag<T: Transport>(
        &self,
        dev: &mut Device<T>,
        window: &Window,
        tag: u16,
    ) -> Result<Option<u32>> {
        match self.address_of(tag) {
            None => Ok(None),
            Some(addr) => Ok(Some(dev.read32(window, arc::arc_tile(), addr)?)),
        }
    }

    /// Every tag this chip publishes, ascending. For reporting.
    pub fn tags(&self) -> impl Iterator<Item = u16> + '_ {
        let mut ids: Vec<u16> = self.entries.iter().map(|&(id, _)| id).collect();
        ids.sort_unstable();
        ids.into_iter()
    }
}

impl<T: Transport> Device<T> {
    /// Ask the chip what it is, before addressing any Tensix tile.
    ///
    /// Needs a window, and uses it only against the ARC. The caller supplies one
    /// rather than having it allocated here so that this can run before any
    /// gate-specific setup, on a device that has just been opened.
    pub fn chip_telemetry(&mut self, window: &Window) -> Result<ChipTelemetry> {
        let table = TelemetryTable::read(self, window)?;

        let enabled_raw = table
            .read_tag(self, window, arc::tag::ENABLED_TENSIX_COL)?
            .ok_or_else(|| malformed("the chip does not publish ENABLED_TENSIX_COL (tag 34)"))?;

        let tensix = Tensix::from_enabled_column_mask(enabled_raw).ok_or_else(|| {
            TransportError::Io(std::io::Error::other(format!(
                "ENABLED_TENSIX_COL = {enabled_raw:#x}: the enabled Tensix columns \
                 are not contiguous from bit 0, which contradicts \
                 NoC/Coordinates.md:54 (harvested columns sit at maximal X). \
                 Refusing to guess the bit order -- report this word."
            )))
        })?;

        if tensix.enabled_column_count() == 0 {
            return Err(malformed("ENABLED_TENSIX_COL reports no usable columns"));
        }

        let noc_translation = table
            .read_tag(self, window, arc::tag::NOC_TRANSLATION)?
            .is_some_and(|v| v != 0);

        Ok(ChipTelemetry {
            tensix,
            noc_translation,
            enabled_tensix_col_raw: enabled_raw,
            harvesting_state_raw: table.read_tag(self, window, arc::tag::HARVESTING_STATE)?,
            heartbeat: table.read_tag(self, window, arc::tag::TIMER_HEARTBEAT)?,
        })
    }

    /// The chip's Tensix grid, read from its ARC.
    ///
    /// Refuses when the NoC is not translating, because [`Tensix`] is only
    /// derivable from a column *count* under translation — see
    /// [`ChipTelemetry::noc_translation`].
    pub fn tensix_grid(&mut self, window: &Window) -> Result<Tensix> {
        let t = self.chip_telemetry(window)?;
        if !t.noc_translation {
            return Err(TransportError::Io(std::io::Error::other(
                "this chip reports NoC coordinate translation disabled. The Tensix \
                 grid is derived from a column count, which only identifies the \
                 surviving columns when translation puts harvested ones at maximal \
                 X (NoC/Coordinates.md:54). Refusing to guess.",
            )));
        }
        Ok(t.tensix)
    }
}

/// Human-readable one-liner for a log or a gate's failure message.
impl core::fmt::Display for ChipTelemetry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} Tensix tiles ({} of {} columns; harvested at X",
            self.tensix.tile_count(),
            self.tensix.enabled_column_count(),
            tt_isa::noc::grid::TENSIX_COLUMNS.len(),
        )?;
        let mut any = false;
        for x in self.tensix.harvested_columns() {
            write!(f, "{}{}", if any { "," } else { " " }, x)?;
            any = true;
        }
        if !any {
            write!(f, " none")?;
        }
        write!(
            f,
            "), translation {}, ENABLED_TENSIX_COL={:#x}",
            if self.noc_translation { "on" } else { "OFF" },
            self.enabled_tensix_col_raw
        )
    }
}

/// A NoC #0 witness that the types line up; the real checks need hardware.
#[cfg(test)]
mod tests {
    use super::*;
    use tt_isa::noc::Noc0;

    #[test]
    fn the_arc_tile_is_addressable_on_noc0() {
        let arc: tt_isa::noc::NocCoord<Noc0> = arc::arc_tile();
        assert_eq!((arc.x(), arc.y()), (arc::ARC_X, arc::ARC_Y));
    }
}
