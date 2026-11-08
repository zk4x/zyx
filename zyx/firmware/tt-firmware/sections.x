/* Everything but the MEMORY block, shared by every image's link script.
 *
 * The images differ only in where their code lives in L1 -- see link.x (T0's
 * default reset PC, used by every single-core image) and link_t1.x/link_t2.x
 * (the role images, each at its own core's default reset PC, so three cores can
 * hold three different images at once). */
ENTRY(_start)

SECTIONS
{
  .text :
  {
    KEEP(*(.text.start))
    /* A mover's per-entry path (`#[link_section = ".text.hot"]` in mover.rs),
     * together and first, then the NoC issue it calls (by its v0-mangled
     * function section; one copy per NIU): RISCV B's and NC's instruction
     * caches are about 4 KiB each (measured, `probe_icache`; a miss costs
     * ~5.5 cycles per 32 bytes), and with the cold paths out of line this
     * takes a WAIT entry from 0.31 to 0.29 us. */
    *(.text.hot .text.hot.*)
    *(.text.*11tt_firmware3noc13issue_dram_on*)
    *(.text.*11tt_firmware3noc8issue_on*)
    /* The write path's NIU choice (`issue_write` in mover.rs): taken by every
     * write, kept off the reads' stretch above. */
    *(.text.warm .text.warm.*)
    /* The fast read path's NoC issue (X6: `noc::issue_read`, which `issue_via`
     * in mover.rs calls when the host chose it), after the slow path's code so
     * that adding it moves nothing in the stretch above. */
    *(.text.*11tt_firmware3noc10issue_read*)
    *(.text .text.*)
  } > L1

  .rodata : ALIGN(4)
  {
    *(.rodata .rodata.*)
    *(.srodata .srodata.*)
  } > L1

  /* Initialised data lives in L1 alongside the code, so the host stages it as
   * part of the single image write. Nothing is copied at run time, and nothing is
   * staged into local data RAM -- which is what keeps the 2048-cycle local-RAM
   * zeroing window (BabyRISCV/README.md:152) from being a hazard here. */
  .data : ALIGN(4)
  {
    *(.data .data.*)
    *(.sdata .sdata.*)
  } > L1

  .bss (NOLOAD) : ALIGN(4)
  {
    __bss_start = .;
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end = .;
  } > L1

  /* Per-core state the firmware keeps for itself (`#[link_section = ".local"]`),
   * at the bottom of local data RAM: 2-cycle access, and the stack grows down
   * from the top. NOLOAD -- nothing can be staged there (the NoC cannot reach
   * it while the core is in reset) -- so it has no initialiser: the firmware
   * writes it before use. */
  .local (NOLOAD) : ALIGN(4)
  {
    *(.local .local.*)
  } > LOCAL

  /* Stack in local data RAM: 2-cycle access, and the core's own accesses stall
   * automatically during the post-reset zeroing window, so setting it up in
   * _start is safe. */
  _stack_top = ORIGIN(LOCAL) + LENGTH(LOCAL);

  /* `.riscv.attributes` records which extensions the image was built for. It is
   * not loaded (objcopy leaves it out of the flat binary), but discarding it
   * makes every disassembler fall back to base RV32I, where an `m`-extension
   * instruction decodes as `<unknown>` -- which would quietly blind the
   * instruction-set gate in crates/tt-tests/build.rs. Keep it. */
  .riscv.attributes 0 : { *(.riscv.attributes) }

  /DISCARD/ :
  {
    *(.eh_frame)
    *(.eh_frame_hdr)
    *(.comment)
  }
}
