/* Link script for a single firmware image on any of T0, T1 or T2.
 *
 * The image is placed at L1 offset 0x6000, which is T0's default reset PC
 * (SoftReset.md:118-123), so it runs on T0 with no further setup. Any other T-core
 * runs the same image by pointing its reset-PC override here, which is what
 * `Device::load_and_start` does. Only RISCV B cannot: its entry point is hardwired
 * to L1 offset 0 and it has no override register.
 *
 * LENGTH stops at 0xA000, which is T1's default reset PC. That is not a guess at
 * how much code will fit -- it is a hard boundary: an image that overran it would
 * silently overwrite T1's entry point, and a link error is a much better way to
 * discover that than a hung T1 three steps later.
 */
MEMORY
{
  L1    (rwx) : ORIGIN = 0x00006000, LENGTH = 0x4000
  LOCAL (rw)  : ORIGIN = 0xFFB00000, LENGTH = 4K
}

INCLUDE sections.x
