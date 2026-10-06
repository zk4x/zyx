// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

fn main() {
    // Only build the TT shim when the feature is enabled
    if std::env::var("CARGO_FEATURE_TENSTORRENT").is_err() {
        return;
    }

    let tt_metal_root = std::env::var("TT_METAL_ROOT").unwrap_or_else(|_| {
        panic!(
            "\n\n\
             TT_METAL_ROOT is not set.\n\
             To build the Tenstorrent backend, point TT_METAL_ROOT at your tt-metal checkout:\n\
               export TT_METAL_ROOT=$HOME/path-to-tt-metal\n\
             \n\
             Without it, the C++ shim (tt_runtime_shim) cannot be compiled.\n"
        );
    });

    let build_dir = std::path::PathBuf::from(&tt_metal_root).join("build_Release");
    let lib_dir = build_dir.join("lib");

    // Find the spdlog CPM cache directory for bundled fmt headers
    let cpm_spdlog = std::path::PathBuf::from(&tt_metal_root).join(".cpmcache").join("spdlog");
    let cpm_fmt = std::path::PathBuf::from(&tt_metal_root).join(".cpmcache").join("fmt");
    let cpm_caches = [
        cpm_spdlog,
        cpm_fmt,
        std::path::PathBuf::from(&tt_metal_root).join(".cpmcache").join("nlohmann_json"),
        std::path::PathBuf::from(&tt_metal_root).join(".cpmcache").join("tt-logger"),
        std::path::PathBuf::from(&tt_metal_root).join(".cpmcache").join("enchantum"),
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
    cmd.arg("-Wno-deprecated-declarations");
    cmd.arg("-DFMT_HEADER_ONLY");

    // Include paths (same layout as the former runtime exe build).
    cmd.arg(format!("-I{tt_metal_root}"));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal"));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/include"));
    cmd.arg(format!("-I{}", build_dir.join("include").display()));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/api"));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/api/tt-metalium"));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/third_party/umd/device/api"));
    cmd.arg(format!("-I{tt_metal_root}/tt_stl"));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/hostdevcommon/api"));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/hw/inc"));
    cmd.arg(format!("-I{tt_metal_root}/src"));
    for include in cpm_include {
        cmd.arg(format!("-I{}", include.display()));
    }

    // Compile-time default for TT_METAL_ROOT (used by the shim for setenv).
    cmd.arg(format!("-DTT_METAL_ROOT_DEFAULT=\"{tt_metal_root}\""));

    cmd.arg("-c").arg("-o").arg(&shim_obj).arg(&shim_src);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!("failed to invoke g++: {e}");
    });
    assert!(status.success(), "g++ shim build failed");

    // Archive into a static library in OUT_DIR so rustc links it.
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let lib = std::path::PathBuf::from(&out_dir).join("libzyx_tt_runtime_shim.a");
    let ar_status = std::process::Command::new("ar").arg("rcs").arg(&lib).arg(&shim_obj).status().expect("ar failed");
    assert!(ar_status.success(), "ar shim archive failed");

    // Link the static shim plus the shared tt-metal libraries it depends on.
    println!("cargo:rerun-if-changed={}", shim_src.display());
    println!("cargo:rerun-if-changed={}", src_dir.join("tt_runtime_shim.h").display());
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=zyx_tt_runtime_shim");

    // Link flags (v0.75 layout: tt_metal/tt_stl/umd/fmt/spdlog all in separate dirs)
    let lib_dirs = [
        lib_dir.clone(),
        build_dir.join("tt_metal"),
        build_dir.join("tt_stl"),
        build_dir.join("tt_metal/third_party/umd/lib"),
        build_dir.join("_deps/fmt-build"),
        build_dir.join("_deps/spdlog-build"),
    ];
    for dir in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    println!("cargo:rustc-link-lib=tt_metal");
    println!("cargo:rustc-link-lib=tt-umd");
    println!("cargo:rustc-link-lib=tt_stl");
    println!("cargo:rustc-link-lib=fmt");
    println!("cargo:rustc-link-lib=spdlog");

    // The static shim is a C++ TU referenced from the Rust rlib; its objects
    // pull in C++ runtime symbols that the Rust link otherwise never resolves.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gcc_s");
    println!("cargo:rustc-link-arg=-static-libstdc++");
    println!("cargo:rustc-link-arg=-static-libgcc");
}
