/* Link script for the Ethernet-tile image, run on RISCV E1.
 *
 * ORIGIN is tt_isa::eth::E1_IMAGE and LENGTH is tt_isa::eth::E1_IMAGE_MAX, which
 * ends at the E1 mailbox; tt-firmware-images checks the ELF entry against the
 * former. Both are inside the Ethernet L1 that base firmware was measured not to
 * use (tt_isa::eth::FIRMWARE_L1).
 *
 * LOCAL is E1's own 8 KiB data RAM (EthernetTile/BabyRISCV/README.md), which the
 * NoC cannot reach -- so, as on Tensix, nothing is staged there; it holds only
 * the stack.
 */
MEMORY
{
  L1    (rwx) : ORIGIN = 0x00010000, LENGTH = 0xF000
  LOCAL (rw)  : ORIGIN = 0xFFB00000, LENGTH = 8K
}

INCLUDE sections.x
