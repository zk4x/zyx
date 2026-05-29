// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

fn main() {
    // Only build the TT runtime when the feature is enabled
    if std::env::var("CARGO_FEATURE_TENSTORRENT").is_err() {
        return;
    }

    let tt_metal_root = std::env::var("TT_METAL_ROOT").unwrap_or_else(|_| {
        panic!(
            "\n\n\
             TT_METAL_ROOT is not set.\n\
             To build the Tenstorrent backend, point TT_METAL_ROOT at your tt-metal checkout:\n\
               export TT_METAL_ROOT=$HOME/Dev/cpp/tt-metal\n\
             or wherever you cloned tt-metal.\n\
             \n\
             Without it, the C++ runtime binary (zyx-tt-runtime) cannot be compiled.\n"
        );
    });

    let build_dir = std::path::PathBuf::from(&tt_metal_root).join("build_Release");
    let lib_dir = build_dir.join("lib");

    // Find the spdlog CPM cache directory for bundled fmt headers
    let cpm_spdlog = std::path::PathBuf::from(&tt_metal_root).join(".cpmcache").join("spdlog");
    let cpm_include = std::fs::read_dir(&cpm_spdlog).ok().and_then(|mut it| {
        it.find_map(|e| {
            let e = e.ok()?;
            let path = e.path();
            if path.is_dir() && path.file_name().and_then(|s| s.to_str()).is_some_and(|s| s.len() == 40) {
                Some(path.join("include"))
            } else {
                None
            }
        })
    });

    let mut cmd = std::process::Command::new("g++");
    cmd.arg("-std=c++20").arg("-Wall").arg("-Wextra").arg("-O3");
    cmd.arg("-Wno-deprecated-declarations");

    // Include paths
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/include"));
    cmd.arg(format!("-I{}", build_dir.join("include").display()));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/api/tt-metalium"));
    cmd.arg(format!("-I{tt_metal_root}/src"));
    if let Some(p) = cpm_include {
        cmd.arg(format!("-I{}", p.display()));
    }

    // Compile-time default for TT_METAL_ROOT (used by runtime.cpp for setenv)
    cmd.arg(format!("-DTT_METAL_ROOT_DEFAULT=\"{tt_metal_root}\""));

    // Source file
    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("backend")
        .join("tenstorrent");
    cmd.arg(src_dir.join("runtime.cpp"));

    // Link flags
    cmd.arg(format!("-L{}", lib_dir.display()));
    cmd.arg("-ltt_metal")
        .arg("-ltt-umd")
        .arg("-ltt_stl")
        .arg("-lfmt")
        .arg("-lspdlog");
    cmd.arg(format!("-Wl,-rpath,{}", lib_dir.display()));

    // Output to config dir
    let config_base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|p| p.starts_with('/'))
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.config")))
        .unwrap_or_else(|| "/tmp".to_string());
    let runtime_path = std::path::Path::new(&config_base).join("zyx/zyx-tt-runtime");
    std::fs::create_dir_all(runtime_path.parent().unwrap()).ok();
    cmd.arg("-o").arg(&runtime_path);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to invoke g++: {e}");
    });
    assert!(status.success(), "g++ build failed");

    let kernel_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("backend")
        .join("tenstorrent")
        .join("kernels");
    println!("cargo:rustc-env=ZYX_TT_KERNEL_DIR={}", kernel_dir.display());

    // Rerun if C++ sources change
    println!("cargo:rerun-if-changed={}", src_dir.join("runtime.cpp").display());
    for e in std::fs::read_dir(&kernel_dir).unwrap().flatten() {
        println!("cargo:rerun-if-changed={}", e.path().display());
    }
}
