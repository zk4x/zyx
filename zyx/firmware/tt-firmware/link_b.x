/* Link script for the RISCV B data mover (`dm_b`).
 *
 * B's entry point is hardwired to L1 offset 0 and it has no reset-PC override
 * (`tt_isa::tensix::Core::B`), so the image lives at 0. LENGTH stops at 0xA800,
 * where the first role image (T0's slot, `tt_isa::l1::roles`) begins, so the
 * mover and the three role images can all be resident in one tile at once -- a
 * link error, not a clobbered T0 entry point, if it ever grows past that
 * (`tt_isa::dm::IMAGE_MAX`).
 */
MEMORY
{
  L1    (rwx) : ORIGIN = 0x00000000, LENGTH = 0xA800
  LOCAL (rw)  : ORIGIN = 0xFFB00000, LENGTH = 4K
}

INCLUDE sections.x
