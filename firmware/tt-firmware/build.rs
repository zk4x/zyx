//! Pick each image's link script, and tell it where its mailbox is.
//!
//! Every Tensix image uses `link.x` (T0's default reset PC) except the three role
//! images, which live in their own slots (`link_role_t{0,1,2}.x`,
//! `tt_isa::l1::roles`) so that one tile can hold all three role images and a
//! mover of up to `dm::IMAGE_MAX` at once, and the per-core probes
//! (`icache_t1`, `icache_t2`), which use their cores' default reset PCs. The
//! Ethernet image uses `link_e1.x`.
//! Per-binary link arguments are the only way to vary this within one crate.
//!
//! The mailbox base is a `--defsym` taken from `tt-isa` (a build-dependency), not
//! a number in a link script: the host polls the same constant, so the two cannot
//! drift.

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-arg-bins=-L{dir}");
    let tensix = [
        ("heartbeat", "link.x"),
        ("sfpu_mul", "link.x"),
        ("prng_seed", "link.x"),
        ("corpus", "link.x"),
        ("corpus_t0", "link.x"),
        ("role_t0", "link_role_t0.x"),
        ("role_t1", "link_role_t1.x"),
        ("role_t2", "link_role_t2.x"),
    ];
    for (bin, script) in tensix {
        link(bin, script, tt_isa::mailbox::MAILBOX_BASE);
    }
    link("eth_e1", "link_e1.x", tt_isa::eth::MAILBOX_BASE);
    link("dm_b", "link_b.x", tt_isa::dm::MAILBOX_BASE);
    link("nc_probe", "link_nc.x", tt_isa::dm::nc::MAILBOX_BASE);
    link("dm_nc", "link_nc.x", tt_isa::dm::nc::MAILBOX_BASE);
    link("icache_b", "link_b.x", tt_isa::mailbox::MAILBOX_BASE);
    link("tag_search_b", "link_b.x", tt_isa::mailbox::MAILBOX_BASE);
    link("noc_probe_b", "link_b.x", tt_isa::mailbox::MAILBOX_BASE);
    link("read_loop_b", "link_b.x", tt_isa::mailbox::MAILBOX_BASE);
    link("icache_nc", "link_nc.x", tt_isa::dm::nc::MAILBOX_BASE);
    for (bin, script, t) in [
        ("icache_t0", "link.x", 0),
        ("icache_t1", "link_t1.x", 1),
        ("icache_t2", "link_t2.x", 2),
    ] {
        link(
            bin,
            script,
            tt_isa::mailbox::role::BASE + t * tt_isa::mailbox::role::STRIDE,
        );
    }
    // The `.ttinsn` experiment image, linked as the T1 role image is
    // (`docs/plans/ttinsn-push-experiment.md`). Only when the feature is on: cargo
    // refuses link arguments for a binary it is not building.
    if std::env::var_os("CARGO_FEATURE_TTINSN_EXPERIMENT").is_some() {
        link("ttinsn_t1", "link_role_t1.x", tt_isa::mailbox::MAILBOX_BASE);
    }
    for f in [
        "link.x",
        "link_t1.x",
        "link_t2.x",
        "link_role_t0.x",
        "link_role_t1.x",
        "link_role_t2.x",
        "link_e1.x",
        "link_b.x",
        "link_nc.x",
        "sections.x",
    ] {
        println!("cargo:rerun-if-changed={f}");
    }
}

fn link(bin: &str, script: &str, mailbox: u64) {
    println!("cargo:rustc-link-arg-bin={bin}=-T{script}");
    println!("cargo:rustc-link-arg-bin={bin}=--defsym=__mailbox={mailbox:#x}");
}
