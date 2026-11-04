//! Firmware images for the baby RISC-V cores.
//!
//! Built from `crates/tt-firmware` by this crate's `build.rs` and checked there
//! for instructions Blackhole cannot execute. The role images are what
//! `tt_kernels::runtime::run` loads; the rest serve the bring-up gates in
//! `tt-tests`, and live here so that the firmware is built in one place.

#![no_std]

use tt_isa::tensix::Core;

/// Whether these images were built with `fast-reads`: the B mover runs host-encoded reads
/// (`tt_isa::dm::encoded`). The mover reports the same thing at run time (`dm::CAPS`), which
/// is what a host consults; this is for code that must know at compile time, a gate that
/// exists only in one mode.
pub const FAST_READS: bool = cfg!(feature = "fast-reads");

/// One image: the core it runs on, its bytes, and where it is loaded in L1.
///
/// The same shape as `tt_kernels::runtime::RoleImages`' entries.
pub type Image = (Core, &'static [u8], u64);

/// Where `link.x` places an image: RISCV T0's default reset PC, which every
/// image but the role images and the per-core probes uses
/// (`crates/tt-firmware/build.rs`).
pub const LOAD_ADDRESS: u64 = Core::T0.default_reset_pc() as u64;

/// The three role images of the LLK-shaped datapath -- unpack on T0, math on
/// T1, pack on T2 -- each linked at its own slot (`tt_isa::l1::roles`), so all
/// three fit in one tile's L1 at once alongside the mover and each starts
/// through its reset-PC override.
pub const ROLES: [Image; 3] = [
    (
        Core::T0,
        include_bytes!(env!("FIRMWARE_ROLE_T0")),
        tt_isa::l1::roles::BASE[0],
    ),
    (
        Core::T1,
        include_bytes!(env!("FIRMWARE_ROLE_T1")),
        tt_isa::l1::roles::BASE[1],
    ),
    (
        Core::T2,
        include_bytes!(env!("FIRMWARE_ROLE_T2")),
        tt_isa::l1::roles::BASE[2],
    ),
];

/// The step 3 heartbeat: increments a counter in L1 forever.
pub const HEARTBEAT: &[u8] = include_bytes!(env!("FIRMWARE_HEARTBEAT"));

/// The step 4 gate: computes 3.0 x 2.0 on the Vector Unit and publishes the FP32
/// bit pattern.
pub const SFPU_MUL: &[u8] = include_bytes!(env!("FIRMWARE_SFPU_MUL"));

/// RISC-V seed-register diagnostic on T1, linked at [`LOAD_ADDRESS`].
pub const PRNG_SEED: &[u8] = include_bytes!(env!("FIRMWARE_PRNG_SEED"));

/// The generic single-thread Tensix program runner, built for T1 (the one core
/// ttsim lets read `Dst`, divergence row 12).
pub const CORPUS: &[u8] = include_bytes!(env!("FIRMWARE_CORPUS"));

/// [`CORPUS`] built for T0, which is where silicon runs single-thread programs
/// (divergence row 45).
pub const CORPUS_T0: &[u8] = include_bytes!(env!("FIRMWARE_CORPUS_T0"));

/// The Ethernet-tile image, for RISCV E1, loaded at `tt_isa::eth::E1_IMAGE`.
pub const ETH_E1: &[u8] = include_bytes!(env!("FIRMWARE_ETH_E1"));

/// The RISCV B data mover (`tt_isa::dm`), loaded at L1 offset 0.
pub const DM_B: Image = (
    Core::B,
    include_bytes!(env!("FIRMWARE_DM_B")),
    tt_isa::dm::IMAGE_BASE,
);

/// RISCV NC's bring-up probe (`tt-firmware/src/bin/nc_probe.rs`), loaded at
/// NC's image base; NC reaches it through the stub at its reset PC
/// (`tt_isa::dm::nc::stub`).
pub const NC_PROBE: Image = (
    Core::NC,
    include_bytes!(env!("FIRMWARE_NC_PROBE")),
    tt_isa::dm::nc::IMAGE_BASE,
);

/// The RISCV NC data mover (`tt_isa::dm::Mover::NC`): the same mover as
/// [`DM_B`], on NC's mailbox and ring, loaded at NC's image base.
pub const DM_NC: Image = (
    Core::NC,
    include_bytes!(env!("FIRMWARE_DM_NC")),
    tt_isa::dm::nc::IMAGE_BASE,
);

/// The instruction-cache probes (`tt-firmware/src/icache_probe.rs`): B's at
/// L1 0, NC's at NC's image base.
pub const ICACHE_B: Image = (
    Core::B,
    include_bytes!(env!("FIRMWARE_ICACHE_B")),
    tt_isa::dm::IMAGE_BASE,
);
pub const ICACHE_NC: Image = (
    Core::NC,
    include_bytes!(env!("FIRMWARE_ICACHE_NC")),
    tt_isa::dm::nc::IMAGE_BASE,
);

/// The L1 Cache Tag Search Accelerator probe (`tt-firmware/src/bin/tag_search_b.rs`)
/// for RISCV B, at L1 0: the block is usable by B only.
pub const TAG_SEARCH_B: Image = (
    Core::B,
    include_bytes!(env!("FIRMWARE_TAG_SEARCH_B")),
    tt_isa::dm::IMAGE_BASE,
);

/// The NoC atomic, multicast and completion probe (`tt-firmware/src/bin/noc_probe_b.rs`)
/// for RISCV B, at L1 0.
pub const NOC_PROBE_B: Image = (
    Core::B,
    include_bytes!(env!("FIRMWARE_NOC_PROBE_B")),
    tt_isa::dm::IMAGE_BASE,
);

/// The bare read-loop probe (`tt-firmware/src/bin/read_loop_b.rs`) for RISCV B, at
/// L1 0: B's GDDR reads with none of the interpreter, for `silicon_probe_read_loop`.
pub const READ_LOOP_B: Image = (
    Core::B,
    include_bytes!(env!("FIRMWARE_READ_LOOP_B")),
    tt_isa::dm::IMAGE_BASE,
);

/// The instruction-cache probes for T0, T1 and T2, each at its core's
/// default reset PC.
pub const ICACHE_T: [Image; 3] = [
    (
        Core::T0,
        include_bytes!(env!("FIRMWARE_ICACHE_T0")),
        Core::T0.default_reset_pc() as u64,
    ),
    (
        Core::T1,
        include_bytes!(env!("FIRMWARE_ICACHE_T1")),
        Core::T1.default_reset_pc() as u64,
    ),
    (
        Core::T2,
        include_bytes!(env!("FIRMWARE_ICACHE_T2")),
        Core::T2.default_reset_pc() as u64,
    ),
];

/// The `.ttinsn` push-fusion experiment's T1 image
/// (`docs/plans/ttinsn-push-experiment.md`): [`ROLES`]'s math image with two hooks,
/// built with fused `.ttinsn` words for a few programs. Only with the
/// `ttinsn-experiment` feature, which no shippable build turns on.
#[cfg(feature = "ttinsn-experiment")]
pub const TTINSN_T1: Image = (
    Core::T1,
    include_bytes!(env!("FIRMWARE_TTINSN_T1")),
    tt_isa::l1::roles::BASE[1],
);

/// The ELF entry points `build.rs` read, by image name.
const ENTRIES: [(&str, &str); 20] = [
    ("heartbeat", env!("FIRMWARE_HEARTBEAT_ENTRY")),
    ("sfpu_mul", env!("FIRMWARE_SFPU_MUL_ENTRY")),
    ("prng_seed", env!("FIRMWARE_PRNG_SEED_ENTRY")),
    ("corpus", env!("FIRMWARE_CORPUS_ENTRY")),
    ("corpus_t0", env!("FIRMWARE_CORPUS_T0_ENTRY")),
    ("role_t0", env!("FIRMWARE_ROLE_T0_ENTRY")),
    ("role_t1", env!("FIRMWARE_ROLE_T1_ENTRY")),
    ("role_t2", env!("FIRMWARE_ROLE_T2_ENTRY")),
    ("eth_e1", env!("FIRMWARE_ETH_E1_ENTRY")),
    ("dm_b", env!("FIRMWARE_DM_B_ENTRY")),
    ("nc_probe", env!("FIRMWARE_NC_PROBE_ENTRY")),
    ("dm_nc", env!("FIRMWARE_DM_NC_ENTRY")),
    ("icache_b", env!("FIRMWARE_ICACHE_B_ENTRY")),
    ("icache_nc", env!("FIRMWARE_ICACHE_NC_ENTRY")),
    ("icache_t0", env!("FIRMWARE_ICACHE_T0_ENTRY")),
    ("icache_t1", env!("FIRMWARE_ICACHE_T1_ENTRY")),
    ("icache_t2", env!("FIRMWARE_ICACHE_T2_ENTRY")),
    ("tag_search_b", env!("FIRMWARE_TAG_SEARCH_B_ENTRY")),
    ("noc_probe_b", env!("FIRMWARE_NOC_PROBE_B_ENTRY")),
    ("read_loop_b", env!("FIRMWARE_READ_LOOP_B_ENTRY")),
];

/// The entry point of the image called `name`, as linked.
pub fn entry(name: &str) -> Option<u32> {
    #[cfg(feature = "ttinsn-experiment")]
    if name == "ttinsn_t1" {
        return Some(
            env!("FIRMWARE_TTINSN_T1_ENTRY")
                .parse()
                .expect("build.rs writes a decimal entry point"),
        );
    }
    ENTRIES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, e)| e.parse().expect("build.rs writes a decimal entry point"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A role image loaded anywhere but where it was linked runs someone else's
    /// code, or nothing. The load addresses above are derived from the cores'
    /// reset PCs and the link scripts say the same thing independently, so this
    /// checks the two against each other.
    #[test]
    fn each_role_image_is_linked_where_it_is_loaded() {
        for ((core, image, at), name) in ROLES.iter().zip(["role_t0", "role_t1", "role_t2"]) {
            assert!(!image.is_empty(), "{name} is empty");
            assert_eq!(
                entry(name).map(u64::from),
                Some(*at),
                "{name} is linked away from {core:?}'s slot"
            );
            assert!(
                image.len() as u64 <= tt_isa::l1::roles::SLOT,
                "{name} outgrew its slot"
            );
        }
    }

    /// The experiment image is a T1 role image: loaded where `role_t1` is.
    #[cfg(feature = "ttinsn-experiment")]
    #[test]
    fn the_experiment_image_is_linked_where_it_is_loaded() {
        assert_eq!(
            entry("ttinsn_t1").map(u64::from),
            Some(TTINSN_T1.2),
            "ttinsn_t1 is linked away from T1's slot"
        );
    }

    #[test]
    fn every_other_image_is_linked_at_the_load_address() {
        for name in ["heartbeat", "sfpu_mul", "prng_seed", "corpus", "corpus_t0"] {
            assert_eq!(entry(name).map(u64::from), Some(LOAD_ADDRESS), "{name}");
        }
    }

    /// `link_e1.x` spells `E1_IMAGE` as a number; this is what holds it to it, and
    /// to the firmware-owned L1 the image must stay out of.
    #[test]
    fn the_ethernet_image_is_linked_where_it_is_loaded() {
        use tt_isa::eth;
        assert_eq!(entry("eth_e1").map(u64::from), Some(eth::E1_IMAGE));
        assert!(ETH_E1.len() as u64 <= eth::E1_IMAGE_MAX);
        assert!(eth::is_customer_l1(eth::E1_IMAGE, eth::E1_IMAGE_MAX));
    }

    /// B cannot be redirected, so its image must be linked at 0 and stay below
    /// the first role image.
    #[test]
    fn the_data_mover_is_linked_where_b_starts() {
        for (name, (core, image, at)) in [
            ("dm_b", DM_B),
            ("tag_search_b", TAG_SEARCH_B),
            ("noc_probe_b", NOC_PROBE_B),
            ("read_loop_b", READ_LOOP_B),
        ] {
            assert_eq!(entry(name).map(u64::from), Some(at));
            assert_eq!(at, core.default_reset_pc() as u64);
            assert!(image.len() as u64 <= tt_isa::dm::IMAGE_MAX);
        }
    }

    /// NC's images run from `dm::nc::IMAGE_BASE`, reached by the stub, and
    /// stay below NC's list ring.
    #[test]
    fn the_nc_images_are_linked_at_ncs_image_base() {
        for (name, (core, image, at)) in [
            ("nc_probe", NC_PROBE),
            ("dm_nc", DM_NC),
            ("icache_nc", ICACHE_NC),
        ] {
            assert_eq!(core, Core::NC, "{name}");
            assert_eq!(entry(name).map(u64::from), Some(at), "{name}");
            assert_eq!(at, tt_isa::dm::nc::IMAGE_BASE, "{name}");
            assert!(image.len() as u64 <= tt_isa::dm::nc::IMAGE_MAX, "{name}");
        }
    }
}
