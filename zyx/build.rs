//! Builds the device firmware and checks its instruction encoding.
//!
//! Runs as part of any build that depends on this crate, so a fresh clone needs
//! no separate step. Moved here from `tt-tests/build.rs` so that the role images a
//! backend runs are owned by a shippable crate rather than by the test harness.
//!
//! The firmware is a separate workspace targeting `riscv32im-unknown-none-elf`.
//! It is built into `OUT_DIR` with its own `--target-dir`, and the host's
//! `CARGO_*` environment is stripped from the child so the nested invocation does
//! not inherit the parent's target directory or profile.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Binaries in `tt-firmware/src/bin` to build and expose.
const BINARIES: &[&str] = &[
    "heartbeat",
    "sfpu_mul",
    "prng_seed",
    "corpus",
    "corpus_t0",
    "role_t0",
    "role_t1",
    "role_t2",
    "eth_e1",
    "dm_b",
    "nc_probe",
    "dm_nc",
    "icache_b",
    "icache_nc",
    "icache_t0",
    "icache_t1",
    "icache_t2",
    "tag_search_b",
    "noc_probe_b",
    "read_loop_b",
    "exec_b",
];

/// The one image the `ttinsn-experiment` feature adds, and the only one whose
/// `.ttinsn` words the gate admits (`check_ttinsn_blocks`).
const TTINSN_IMAGE: &str = "ttinsn_t1";

const TARGET: &str = "riscv32im-unknown-none-elf";

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let firmware_dir = manifest.join("firmware").join("tt-firmware");
    println!("cargo:rerun-if-changed=build.rs");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Track individual files rather than directories. A directory-level
    // `rerun-if-changed` does not reliably notice an edit to a file nested inside
    // it, which showed up here as firmware changes that never rebuilt.
    track_sources(&firmware_dir);

    // The firmware sources ship inside this crate but are not a package of
    // their own in the tree (`Cargo.toml.in`, not `Cargo.toml`): cargo refuses
    // to package a nested package. Copy them into `OUT_DIR` and give them the
    // manifest there, so the build works both from the tree and from the
    // published crate.
    let build_dir = out_dir.join("firmware-src");
    copy_tree(&firmware_dir, &build_dir);
    std::fs::copy(firmware_dir.join("Cargo.toml.in"), build_dir.join("Cargo.toml"))
        .expect("could not write the firmware manifest");
    println!("cargo:rerun-if-changed={}", firmware_dir.join("Cargo.toml.in").display());

    let target_dir = out_dir.join("firmware-target");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());

    let experiment = std::env::var_os("CARGO_FEATURE_TTINSN_EXPERIMENT").is_some();
    let mut cmd = Command::new(&cargo);
    cmd.current_dir(&build_dir).args(["build", "--release", "--target", TARGET]).arg("--target-dir").arg(&target_dir);
    let mut features = Vec::new();
    if experiment {
        features.push("ttinsn-experiment");
    }
    // The mover's read mode is chosen here, at build time (`tt-firmware`'s features).
    if std::env::var_os("CARGO_FEATURE_FAST_READS").is_some() {
        features.push("fast-reads");
    }
    if std::env::var_os("CARGO_FEATURE_ENCODED_GUARD").is_some() {
        features.push("encoded-guard");
    }
    if !features.is_empty() {
        cmd.args(["--features", &features.join(",")]);
    }
    // A nested cargo inherits variables that redirect it back at the parent
    // workspace; strip them so the firmware build is genuinely independent.
    // `RUSTC_WORKSPACE_WRAPPER` is `cargo clippy`'s `clippy-driver`: left in, the
    // images of a clippy run came out about 900 bytes larger than a build's
    // (`dm_b` 24,540 bytes against 23,644, of the 24,576 it may have), so a
    // mover change that fit a build overflowed clippy. Firmware is linted on
    // its own (`AGENTS.md`).
    for (key, _) in std::env::vars() {
        if key.starts_with("CARGO_") || key == "RUSTFLAGS" || key == "RUSTC_WRAPPER" || key == "RUSTC_WORKSPACE_WRAPPER" {
            cmd.env_remove(key);
        }
    }
    let status = cmd.status().expect("could not run cargo for the firmware build");
    assert!(status.success(), "firmware build failed");

    let objcopy = tool("llvm-objcopy");
    let objdump = tool("llvm-objdump");

    let binaries = BINARIES.iter().copied().chain(experiment.then_some(TTINSN_IMAGE));
    for name in binaries {
        let elf = target_dir.join(TARGET).join("release").join(name);
        assert!(elf.exists(), "firmware binary {name} was not produced at {}", elf.display());

        let bin = out_dir.join(format!("{name}.bin"));
        let status =
            Command::new(&objcopy).args(["-O", "binary"]).arg(&elf).arg(&bin).status().expect("could not run llvm-objcopy");
        assert!(status.success(), "objcopy failed for {name}");

        check_instruction_set(&objdump, &elf, name);

        // The entry point, so the crate can check each image against the core
        // it is loaded for instead of trusting a literal next to it.
        println!("cargo:rustc-env=FIRMWARE_{}_ENTRY={}", name.to_uppercase(), elf_entry(&elf));

        println!("cargo:rustc-env=FIRMWARE_{}={}", name.to_uppercase(), bin.display());
    }
}

/// Emit `rerun-if-changed` for every source file under `dir`, skipping build
/// output.
fn track_sources(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if name == "target" || name == ".git" {
            continue;
        }
        if path.is_dir() {
            track_sources(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Copy the firmware source tree (no `target/`, no `.git/`) into `dst`.
fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("could not create the firmware build directory");
    for entry in std::fs::read_dir(src).expect("could not read the firmware sources").flatten() {
        let name = entry.file_name();
        if name == "target" || name == ".git" {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        if from.is_dir() {
            copy_tree(&from, &to);
        } else {
            std::fs::copy(&from, &to).expect("could not copy a firmware source file");
        }
    }
}

/// Locate one of the `llvm-tools` binaries shipped with the active toolchain.
fn tool(name: &str) -> PathBuf {
    let sysroot = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .arg("--print")
        .arg("sysroot")
        .output()
        .expect("could not ask rustc for its sysroot");
    let sysroot = String::from_utf8(sysroot.stdout).unwrap();
    let host = std::env::var("HOST").unwrap_or_else(|_| "x86_64-unknown-linux-gnu".into());
    let path = Path::new(sysroot.trim()).join("lib/rustlib").join(host).join("bin").join(name);
    assert!(path.exists(), "{name} not found at {}. Install it with `rustup component add llvm-tools`.", path.display());
    path
}

/// Assert the image contains nothing Blackhole cannot execute.
///
/// This is a correctness gate, not hygiene. Each of these is a documented gap in
/// the baby RISC-V implementation, and executing an unimplemented instruction is
/// `UndefinedBehavior` — which, per the glossary, includes possible physical
/// damage. There is no illegal-instruction trap to fall back on.
fn check_instruction_set(objdump: &Path, elf: &Path, name: &str) {
    // Every image but the experiment's is disassembled whole and judged, exactly
    // as it always was. The experiment's `.ttinsn` words cannot be disassembled
    // (they sit in the compressed encoding space this gate refuses), so for that
    // one image the words are validated directly and the rest is judged in the
    // gaps between them.
    let blocks = if name == TTINSN_IMAGE {
        check_ttinsn_blocks(objdump, elf, name)
    } else {
        Vec::new()
    };
    let mut checked = 0usize;
    if blocks.is_empty() {
        assert!(name != TTINSN_IMAGE, "`{name}` has no `.ttinsn` words: the experiment image is the one image that must");
        checked += judge(&disassemble(objdump, elf, name, None), name);
    } else {
        let mut from = 0u64;
        for &(lo, hi) in &blocks {
            if from < lo {
                checked += judge(&disassemble(objdump, elf, name, Some((from, lo))), name);
            }
            from = hi;
        }
        checked += judge(&disassemble(objdump, elf, name, Some((from, u32::MAX as u64 + 1))), name);
    }

    // A parser that matches nothing passes everything. This check exists because
    // an earlier version of it did exactly that: llvm-objdump puts spaces between
    // the address colon and the mnemonic, and the first parser looked for ":\t".
    assert!(
        checked > 8,
        "the instruction-set check only recognised {checked} instructions in `{name}`, \
         which means it is not actually reading the disassembly"
    );
}

/// The disassembly of `elf`, or of `range` of it.
fn disassemble(objdump: &Path, elf: &Path, name: &str, range: Option<(u64, u64)>) -> String {
    let mut cmd = Command::new(objdump);
    // `F` on top of the image's own attributes (rv32im): firmware reaches
    // the FP32 unit through inline assembly (`tt_firmware::float`), and
    // those instructions must decode so this gate can judge them -- it
    // refuses the ones Blackhole lacks or gets wrong, below -- rather than
    // meet them as `<unknown>`.
    cmd.args(["-d", "--no-show-raw-insn", "--mattr=+f"]);
    if let Some((lo, hi)) = range {
        cmd.arg(format!("--start-address={lo:#x}")).arg(format!("--stop-address={hi:#x}"));
    }
    let out = cmd.arg(elf).output().expect("could not run llvm-objdump");
    assert!(out.status.success(), "objdump failed for {name}");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Judge every instruction line of `text`, and say how many there were.
fn judge(text: &str, name: &str) -> usize {
    let mut checked = 0usize;
    for line in text.lines() {
        // Instruction lines are "<hex address>:<whitespace><mnemonic><whitespace><operands>".
        // Label lines ("00006000 <_start>:") end in ':' and have no mnemonic after it.
        let Some((addr, rest)) = line.split_once(':') else {
            continue;
        };
        if addr.trim().is_empty() || !addr.trim().chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let Some(mnemonic) = rest.split_whitespace().next() else {
            continue;
        };
        checked += 1;
        // Data in a code section. The assembler marks `.word` as data, so the
        // disassembler prints it as such instead of failing to decode it -- which
        // means an instruction the gate refuses (a compressed one, a `.ttinsn`
        // word) would pass as a harmless literal. No image has any, so none is
        // allowed; the experiment image's `.ttinsn` runs never reach this
        // (`check_ttinsn_blocks` validates them and `check_instruction_set`
        // disassembles only the gaps).
        if rest.split_whitespace().any(|t| matches!(t, ".word" | ".byte" | ".half" | ".short" | ".2byte" | ".4byte")) {
            panic!(
                "firmware `{name}` has data in a code section, which the gate cannot \
                 judge: the disassembler prints undecodable words as `.word`\n  in:{line}"
            );
        }
        let operands = rest.trim().strip_prefix(mnemonic).unwrap_or("").trim();
        if let Some(why) = forbidden_reason(mnemonic).or_else(|| fence_reason(mnemonic, operands)) {
            panic!(
                "firmware `{name}` contains `{mnemonic}`, which Blackhole does not \
                 implement: {why}\n  in:{line}"
            );
        }
    }
    checked
}

/// The runs of `.ttinsn` words in the experiment image (`ttinsn-experiment`), as
/// `(first byte, one past the last)`, after checking every word in them.
///
/// The words are in the compressed encoding space the gate refuses, so the
/// disassembler cannot judge them (it prints `<unknown>` or `c.*`, and may
/// misread what follows). Instead each run is bracketed by the symbols
/// `ttw_b<N>`/`ttw_e<N>`, and what the gate checks is:
///
/// * the runs lie inside the image's `.ttinsn` section, in whole words;
/// * every word in them is a Tensix instruction rotated left two bits: its low
///   two bits are not `11`, which is `.ttinsn`'s own condition
///   (`PushTensixInstruction.md:21`), and its opcode byte, rotated back, is one
///   in the instruction table;
/// * everything outside the runs, including the rest of the section, goes
///   through the ordinary judgement, each gap disassembled on its own so that
///   decoding starts on an instruction boundary.
///
/// No other image reaches here, and an image built without the feature has no
/// `.ttinsn` section, so a `.ttinsn` word anywhere else is still `<unknown>`.
fn check_ttinsn_blocks(objdump: &Path, elf: &Path, name: &str) -> Vec<(u64, u64)> {
    use std::collections::BTreeMap;
    let nm = Command::new(tool("llvm-nm")).arg(elf).output().expect("could not run llvm-nm");
    assert!(nm.status.success(), "llvm-nm failed for {name}");
    let mut begins = BTreeMap::new();
    let mut ends = BTreeMap::new();
    for line in String::from_utf8_lossy(&nm.stdout).lines() {
        let mut it = line.split_whitespace();
        let (Some(addr), Some(_kind), Some(symbol)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let at = u64::from_str_radix(addr, 16).expect("llvm-nm prints a hex address");
        if let Some(n) = symbol.strip_prefix("ttw_b") {
            begins.insert(n.to_string(), at);
        } else if let Some(n) = symbol.strip_prefix("ttw_e") {
            ends.insert(n.to_string(), at);
        }
    }
    assert!(
        begins.keys().eq(ends.keys()) && !begins.is_empty(),
        "`{name}`: the `.ttinsn` runs' begin and end symbols do not pair up"
    );

    // The section the words live in, from its header: " N .ttinsn SIZE VMA ...".
    let headers = Command::new(objdump).arg("-h").arg(elf).output().expect("could not run llvm-objdump -h");
    let headers = String::from_utf8_lossy(&headers.stdout).into_owned();
    let section = headers
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (f.len() >= 4 && f[1] == ".ttinsn")
                .then(|| (u64::from_str_radix(f[3], 16).unwrap(), u64::from_str_radix(f[2], 16).unwrap()))
        })
        .next()
        .unwrap_or_else(|| panic!("`{name}` has no `.ttinsn` section"));
    let (base, size) = section;

    let dump = out_dir_file(&format!("{name}.ttinsn.bin"));
    let status = Command::new(tool("llvm-objcopy"))
        .arg("--dump-section")
        .arg(format!(".ttinsn={}", dump.display()))
        .arg(elf)
        .arg(out_dir_file(&format!("{name}.ttinsn.scratch")))
        .status()
        .expect("could not run llvm-objcopy");
    assert!(status.success(), "could not dump `.ttinsn` of `{name}`");
    let bytes = std::fs::read(&dump).expect("the dumped `.ttinsn` section");
    assert_eq!(bytes.len() as u64, size, "`{name}`: `.ttinsn` dump size");

    let mut blocks = Vec::new();
    for (n, &lo) in &begins {
        let hi = ends[n];
        assert!(
            lo % 4 == 0 && hi % 4 == 0 && lo < hi && lo >= base && hi <= base + size,
            "`{name}`: `.ttinsn` run {n} ({lo:#x}..{hi:#x}) is not whole words inside the \
             section ({base:#x}, {size:#x} bytes)"
        );
        for at in (lo..hi).step_by(4) {
            let k = (at - base) as usize;
            let w = u32::from_le_bytes(bytes[k..k + 4].try_into().unwrap());
            // `.ttinsn IMM32` is `IMM32` rotated left two bits, and is only valid
            // below `0xC000_0000`: after rotating back, the top two bits are the
            // word's low two, which must not both be set.
            let tensix = w.rotate_right(2);
            assert!(
                tensix < 0xC000_0000,
                "`{name}`: word {w:#010x} at {at:#x} has the low bits `11` of an ordinary \
                 instruction, which is not a `.ttinsn` word"
            );
            assert!(
                tt_isa::isa::generated::ALL.iter().any(|d| d.opcode() == (tensix >> 24) as u8),
                "`{name}`: word {w:#010x} at {at:#x} rotates back to {tensix:#010x}, whose \
                 opcode is not a Tensix instruction"
            );
        }
        blocks.push((lo, hi));
    }
    blocks.sort();
    blocks
}

/// A file in this build script's `OUT_DIR`.
fn out_dir_file(name: &str) -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(name)
}

/// Why this `fence` must not appear, if it must not.
///
/// `pause` (Zihintpause) shares an encoding with `fence w, 0`, and a disassembler
/// that does not know the extension prints the latter — so checking the mnemonic
/// alone misses it. Blackhole does not list Zihintpause among the extensions it
/// implements, and `core::hint::spin_loop()` emits it, which is an easy way to
/// put an unimplemented instruction into a spin loop without noticing.
///
/// The permitted forms are the ordering fences the specification's own recipes
/// use. Anything else is flagged rather than assumed safe.
fn fence_reason(mnemonic: &str, operands: &str) -> Option<&'static str> {
    if mnemonic != "fence" {
        return None;
    }
    const ALLOWED: &[&str] = &["", "iorw, iorw", "rw, rw", "rw, w", "r, rw"];
    if ALLOWED.contains(&operands) {
        return None;
    }
    if operands == "w, 0" {
        return Some(
            "this is the `pause` encoding (Zihintpause), which Blackhole does not implement. \
             `core::hint::spin_loop()` emits it -- use an empty asm block instead",
        );
    }
    Some("an unrecognised fence ordering; only the forms the specification's recipes use are allowed")
}

/// Why this mnemonic must not appear in a firmware image, if it must not.
///
/// Each is a documented gap in the baby RISC-V implementation, and executing an
/// unimplemented instruction is `UndefinedBehavior` — which per `Glossary.md`
/// includes possible physical damage. There is no illegal-instruction trap to
/// fall back on, so this is a correctness gate rather than hygiene.
fn forbidden_reason(mnemonic: &str) -> Option<&'static str> {
    // Strip the ordering suffixes AMOs may carry, so `lr.w.aq` is caught too.
    let base = mnemonic.trim_end_matches(".aq").trim_end_matches(".rl");
    if mnemonic == "<unknown>" {
        return Some(
            "the disassembler could not decode it, so neither this gate nor a reader can \
             tell what it is. Usually a missing `.riscv.attributes` section or an \
             instruction outside the target's extension set",
        );
    }
    if mnemonic == "fence.i" {
        return Some(
            "Zifencei is not implemented. `fence.i` executes as a `nop`, which \
             `InstructionCache.md:23` classes as NonContractualBehavior, and ttsim refuses it \
             outright: \"fence.i does not flush the instruction cache on babyrisc and should \
             not be used\". Write the 5-bit mask to `RISCV_IC_INVALIDATE_InvalidateAll` instead",
        );
    }
    if mnemonic.starts_with("c.") {
        return Some("the C extension is not implemented, and its encoding space is reused by `.ttinsn`");
    }
    if base.starts_with("lr.") || base.starts_with("sc.") {
        return Some("Zalrsc is not implemented; there is no load-reserved/store-conditional");
    }
    // Every F-extension instruction, `fence` and `fence.i` aside (the latter
    // refused above): the baby cores move data and push instructions, and do
    // no floating-point arithmetic of their own. Float math is the SFPU's and
    // the Matrix Unit's, where it is fast and held to Flex; a slow scalar copy
    // on RISCV B is a trap someone would one day route a tensor through.
    if mnemonic.starts_with('f') && !mnemonic.starts_with("fence") {
        return Some(
            "no floating point on the baby cores: float math belongs on the SFPU \
             or the Matrix Unit (the data mover only moves data)",
        );
    }
    // Integer divide and remainder are allowed: every baby core implements
    // the whole M extension (`BabyRISCV/InstructionSet.md:7`). The only
    // divides missing are T2's vector `vdiv`/`vrem` (`InstructionSet.md:33`),
    // which no image uses. A scalar divide takes 6-33 cycles
    // (`BabyRISCV/README.md:51`); this gate once refused it on a misreading.
    None
}

/// `e_entry` of a 32-bit little-endian ELF.
fn elf_entry(elf: &Path) -> u32 {
    let bytes = std::fs::read(elf).expect("could not read the firmware ELF");
    assert!(
        bytes.len() >= 28 && &bytes[..4] == b"\x7fELF" && bytes[4] == 1 && bytes[5] == 1,
        "{} is not a 32-bit little-endian ELF",
        elf.display()
    );
    u32::from_le_bytes(bytes[24..28].try_into().unwrap())
}
