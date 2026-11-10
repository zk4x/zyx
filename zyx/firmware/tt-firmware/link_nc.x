/* Link script for RISCV NC's images (`nc_probe`, and the NC mover).
 *
 * NC's own slot, from its default reset PC 0x12000 to the B mover's list at
 * 0x14000, is 8 KiB: too small for a mover. The image lives instead at
 * `tt_isa::dm::nc::IMAGE_BASE` (`tt_isa::l1::NC_IMAGE`), between B's mover
 * area and the data arena, and the host puts a two-instruction jump to it at
 * 0x12000 (`dm::nc::stub`). LENGTH stops at the data arena
 * (`dm::nc::IMAGE_MAX`). NC has 8 KiB of local data RAM
 * (`BabyRISCV/README.md:146`).
 */
MEMORY
{
  L1    (rwx) : ORIGIN = 0x0001A000, LENGTH = 0x6000
  LOCAL (rw)  : ORIGIN = 0xFFB00000, LENGTH = 8K
}

INCLUDE sections.x
