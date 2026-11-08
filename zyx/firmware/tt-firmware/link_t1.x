/* Link script for the T1 role image: at T1's default reset PC
 * (SoftReset.md:118-123), so it needs no override and coexists with the T0 image
 * at 0x6000 and the other role images. LENGTH stops at the next core's default
 * reset PC (0xE000, T2), for the reason link.x gives. */
MEMORY
{
  L1    (rwx) : ORIGIN = 0x0000A000, LENGTH = 0x4000
  LOCAL (rw)  : ORIGIN = 0xFFB00000, LENGTH = 4K
}

INCLUDE sections.x
