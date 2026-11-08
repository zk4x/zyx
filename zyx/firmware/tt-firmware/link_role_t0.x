/* Link script for the T0 role image: in its slot of `tt_isa::l1::roles`
 * (0x0000A800, `roles::BASE[0]`), reached through T0's reset-PC override
 * (`Device::load_and_start` always sets it). The slots are packed against NC's
 * slot so that B's mover keeps everything below them (`tt_isa::dm::IMAGE_MAX`).
 * LENGTH is `roles::SLOT`; tt-firmware-images checks the entry and the size
 * against the constants, as the numbers here cannot follow them. */
MEMORY
{
  L1    (rwx) : ORIGIN = 0x0000A800, LENGTH = 0x2800
  LOCAL (rw)  : ORIGIN = 0xFFB00000, LENGTH = 4K
}

INCLUDE sections.x
