// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

fn main() {
    // Only build the TT shim when the feature is enabled
    if std::env::var("CARGO_FEATURE_TENSTORRENT").is_err() {
        return;
    }

    let tt_metal_runtime_root = std::env::var("TT_METAL_RUNTIME_ROOT").unwrap_or_else(|_| {
        panic!(
            "\n\n\
             TT_METAL_RUNTIME_ROOT is not set.\n\
             To build the Tenstorrent backend, point TT_METAL_RUNTIME_ROOT at your tt-metal checkout:\n\
               export TT_METAL_RUNTIME_ROOT=$HOME/path-to-tt-metal\n\
             \n\
             Without it, the C++ shim (tt_runtime_shim) cannot be compiled.\n"
        );
    });

    let build_dir = std::path::PathBuf::from(&tt_metal_runtime_root).join("build_Release");
    let lib_dir = build_dir.join("lib");

    // Find the spdlog CPM cache directory for bundled fmt headers
    let cpm_spdlog = std::path::PathBuf::from(&tt_metal_runtime_root).join(".cpmcache").join("spdlog");
    let cpm_fmt = std::path::PathBuf::from(&tt_metal_runtime_root).join(".cpmcache").join("fmt");
    let cpm_caches = [
        cpm_spdlog,
        cpm_fmt,
        std::path::PathBuf::from(&tt_metal_runtime_root).join(".cpmcache").join("nlohmann_json"),
        std::path::PathBuf::from(&tt_metal_runtime_root).join(".cpmcache").join("tt-logger"),
        std::path::PathBuf::from(&tt_metal_runtime_root).join(".cpmcache").join("enchantum"),
    ];
    let cpm_include = cpm_caches.iter().filter_map(|cache| {
        std::fs::read_dir(cache).ok().and_then(|mut it| {
            it.find_map(|e| {
                let e = e.ok()?;
                let path = e.path();
                if path.is_dir() && path.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.len() == 40) {
                    let include = path.join("include");
                    if include.is_dir() {
                        Some(include)
                    } else {
                        let sub = path.join("enchantum/include");
                        if sub.is_dir() { Some(sub) } else { None }
                    }
                } else {
                    None
                }
            })
        })
    });

    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("backend");
    let shim_src = src_dir.join("tt_runtime_shim.cpp");
    let shim_obj = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("tt_runtime_shim.o");

    // Compile the shim to an object file (header-only fmt: no external libfmt
    // needed for the shim TU itself).
    let mut cmd = std::process::Command::new("g++");
    cmd.arg("-std=c++20").arg("-Wall").arg("-Wextra").arg("-O3");
    cmd.arg("-fPIC");
    cmd.arg("-Wno-deprecated-declarations");
    cmd.arg("-DFMT_HEADER_ONLY");

    // Include paths (same layout as the former runtime exe build).
    cmd.arg(format!("-I{tt_metal_runtime_root}"));
    cmd.arg(format!("-I{tt_metal_runtime_root}/tt_metal"));
    cmd.arg(format!("-I{tt_metal_runtime_root}/tt_metal/include"));
    cmd.arg(format!("-I{}", build_dir.join("include").display()));
    cmd.arg(format!("-I{tt_metal_runtime_root}/tt_metal/api"));
    cmd.arg(format!("-I{tt_metal_runtime_root}/tt_metal/api/tt-metalium"));
    cmd.arg(format!("-I{tt_metal_runtime_root}/tt_metal/third_party/umd/device/api"));
    cmd.arg(format!("-I{tt_metal_runtime_root}/tt_stl"));
    cmd.arg(format!("-I{tt_metal_runtime_root}/tt_metal/hostdevcommon/api"));
    cmd.arg(format!("-I{tt_metal_runtime_root}/tt_metal/hw/inc"));
    cmd.arg(format!("-I{tt_metal_runtime_root}/src"));
    for include in cpm_include {
        cmd.arg(format!("-I{}", include.display()));
    }

    // Compile-time default for TT_METAL_RUNTIME_ROOT (used by the shim for setenv).
    cmd.arg(format!("-DTT_METAL_RUNTIME_ROOT_DEFAULT=\"{tt_metal_runtime_root}\""));

    cmd.arg("-c").arg("-o").arg(&shim_obj).arg(&shim_src);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to invoke g++: {e}");
    });
    assert!(status.success(), "g++ shim build failed");

    // Link the shim as a versioned cdylib and ship it to a persistent
    // dir. The Rust backend dlopens it at device init (see backend/
    // tenstorrent.rs) instead of linking tt-metal into every downstream
    // binary — so user binaries carry no tt-metal NEEDED entries and need
    // no loader path of their own. The cdylib's own link is fully owned
    // here (rpath included), which is what makes that work.
    //
    // Filename hash inputs: shim sources + TT_METAL_RUNTIME_ROOT + zyx version, so
    // any of those changing ships a fresh file (stale shims never shadow).
    // cargo reruns this script when the shim sources or TT_METAL_RUNTIME_ROOT
    // change; an explicit existence check covers a wiped ship dir.
    // Link flags (v0.75 layout: tt_metal/tt_stl/umd/fmt/spdlog all in separate dirs)
    let lib_dirs = [
        lib_dir.clone(),
        build_dir.join("tt_metal"),
        build_dir.join("tt_stl"),
        build_dir.join("tt_metal/third_party/umd/lib"),
        build_dir.join("_deps/fmt-build"),
        build_dir.join("_deps/spdlog-build"),
    ];
    let shim_hdr = src_dir.join("tt_runtime_shim.h");
    let hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        std::fs::read(&shim_src).expect("read shim src").hash(&mut h);
        std::fs::read(&shim_hdr).expect("read shim hdr").hash(&mut h);
        tt_metal_runtime_root.hash(&mut h);
        std::env::var("CARGO_PKG_VERSION").unwrap_or_default().hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let shim_name = format!("libzyx_tt_shim-{hash}.so");
    // Ship dir: the XDG config dir (XDG_CONFIG_HOME, else ~/.config).
    // Mirrored in backend/tenstorrent.rs; keep in sync.
    let config_base = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| {
            panic!("\n\nNeither XDG_CONFIG_HOME nor HOME is set; zyx ships the TT shim under the XDG config dir.\n")
        });
        format!("{home}/.config")
    });
    let ship_dir = std::path::PathBuf::from(config_base).join("zyx");
    std::fs::create_dir_all(&ship_dir).expect("create ship dir");
    let shipped = ship_dir.join(&shim_name);
    if !shipped.is_file() {
        let tmp = ship_dir.join(format!(".{shim_name}.tmp"));
        let mut link = std::process::Command::new("g++");
        link.arg("-shared").arg("-O2");
        link.arg("-o").arg(&tmp).arg(&shim_obj);
        for dir in &lib_dirs {
            link.arg(format!("-L{}", dir.display()));
        }
        link.arg("-ltt_metal").arg("-ltt-umd").arg("-ltt_stl").arg("-lfmt").arg("-lspdlog");
        for dir in &lib_dirs {
            link.arg(format!("-Wl,-rpath,{}", dir.display()));
        }
        link.arg("-static-libstdc++").arg("-static-libgcc");
        let status = link.status().unwrap_or_else(|e| panic!("failed to invoke g++ for shim cdylib: {e}"));
        assert!(status.success(), "g++ shim cdylib link failed");
        std::fs::rename(&tmp, &shipped).expect("ship shim cdylib");
    }
    println!("cargo:rerun-if-changed={}", shim_src.display());
    println!("cargo:rerun-if-changed={}", shim_hdr.display());
    println!("cargo:rerun-if-env-changed=TT_METAL_RUNTIME_ROOT");
    // Expected shim filename, baked into the rlib; the backend resolves
    // $HOME/.config/zyx/<this> at device init. Owning-crate-only env, no
    // downstream propagation involved.
    println!("cargo:rustc-env=ZYX_TT_SHIM={shim_name}");
}
