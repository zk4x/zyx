// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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
               export TT_METAL_ROOT=$HOME/path-to-tt-metal\n\
             \n\
             Without it, the C++ runtime binary (zyx-tt-runtime) cannot be compiled.\n"
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

    let mut cmd = std::process::Command::new("g++");
    cmd.arg("-std=c++20").arg("-Wall").arg("-Wextra").arg("-O3");
    cmd.arg("-Wno-deprecated-declarations");

    // Include paths
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/include"));
    cmd.arg(format!("-I{}", build_dir.join("include").display()));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/api"));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/api/tt-metalium"));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/third_party/umd/device/api"));
    cmd.arg(format!("-I{tt_metal_root}/tt_stl"));
    cmd.arg(format!("-I{tt_metal_root}/tt_metal/hostdevcommon/api"));
    cmd.arg(format!("-I{tt_metal_root}/src"));
    for include in cpm_include {
        cmd.arg(format!("-I{}", include.display()));
    }

    // Compile-time default for TT_METAL_ROOT (used by runtime.cpp for setenv)
    cmd.arg(format!("-DTT_METAL_ROOT_DEFAULT=\"{tt_metal_root}\""));

    // Source file
    let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join("backend");
    cmd.arg(src_dir.join("tt_runtime.cpp"));

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
        cmd.arg(format!("-L{}", dir.display()));
        cmd.arg(format!("-Wl,-rpath,{}", dir.display()));
    }
    cmd.arg("-ltt_metal").arg("-ltt-umd").arg("-ltt_stl").arg("-lfmt").arg("-lspdlog");

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

    // Rerun if C++ sources change
    println!("cargo:rerun-if-changed={}", src_dir.join("tt_runtime.cpp").display());
}
