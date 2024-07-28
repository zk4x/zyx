use std::env;
use std::path::PathBuf;

// Use https://github.com/rust-cuda/cuda-sys/blob/cuda-bindgen/cuda-bindgen/src/main.rs
// OR https://github.com/inducer/pycuda/blob/master/pycuda/compiler.py#L349
pub fn find_cuda2() -> Vec<PathBuf> {
    let mut candidates = read_env();
    candidates.push(PathBuf::from("/opt/cuda"));
    candidates.push(PathBuf::from("/usr/local/cuda"));
    for e in std::fs::read_dir("/usr/local/").unwrap() {
        if let Ok(e) = e {
            let path = e.path();
            if path.starts_with("cuda-") {
                candidates.push(path)
            }
        }
    }

    let mut valid_paths = vec![];
    for base in &candidates {
        let lib = PathBuf::from(base).join("lib64");
        if lib.is_dir() {
            valid_paths.push(lib.clone());
            valid_paths.push(lib.join("stubs"));
        }
        let base = base.join("targets/x86_64-linux");
        let header = base.join("include/cuda.h");
        if header.is_file() {
            valid_paths.push(base.join("lib"));
            valid_paths.push(base.join("lib/stubs"));
            continue;
        }
    }
    eprintln!("Found CUDA paths: {:?}", valid_paths);
    valid_paths
}

#[cfg(feature = "cuda")]
fn find_cuda() -> Option<PathBuf> {
    let cuda_env = env::var("CUDA_LIBRARY_PATH")
        .ok()
        .unwrap_or(String::from(""));
    let mut paths: Vec<PathBuf> = env::split_paths(&cuda_env).collect();
    paths.push(PathBuf::from("/usr/local/cuda"));
    paths.push(PathBuf::from("/opt/cuda"));
    for path in paths {
        if path.join("include/nvrtc.h").is_file() {
            return Some(path);
        }
    }
    None
}

pub fn read_env() -> Vec<PathBuf> {
    if let Ok(path) = env::var("CUDA_LIBRARY_PATH") {
        // The location of the libcuda, libcudart, and libcublas can be hardcoded with the
        // CUDA_LIBRARY_PATH environment variable.
        let split_char = if cfg!(target_os = "windows") {
            ";"
        } else {
            ":"
        };
        path.split(split_char).map(|s| PathBuf::from(s)).collect()
    } else {
        vec![]
    }
}

#[cfg(feature = "cuda")]
fn find_cuda_windows() -> PathBuf {
    let paths = read_env();
    if !paths.is_empty() {
        return paths[0].clone();
    }

    if let Ok(path) = env::var("CUDA_PATH") {
        // If CUDA_LIBRARY_PATH is not found, then CUDA_PATH will be used when building for
        // Windows to locate the Cuda installation. Cuda installs the full Cuda SDK for 64-bit,
        // but only a limited set of libraries for 32-bit. Namely, it does not include cublas in
        // 32-bit, which cuda-sys requires.

        // 'path' points to the base of the CUDA Installation. The lib directory is a
        // sub-directory.
        let path = PathBuf::from(path);

        // To do this the right way, we check to see which target we're building for.
        let target = env::var("TARGET")
            .expect("cargo did not set the TARGET environment variable as required.");

        // Targets use '-' separators. e.g. x86_64-pc-windows-msvc
        let target_components: Vec<_> = target.as_str().split("-").collect();

        // We check that we're building for Windows. This code assumes that the layout in
        // CUDA_PATH matches Windows.
        if target_components[2] != "windows" {
            panic!(
                "The CUDA_PATH variable is only used by cuda-sys on Windows. Your target is {}.",
                target
            );
        }

        // Sanity check that the second component of 'target' is "pc"
        debug_assert_eq!(
            "pc", target_components[1],
            "Expected a Windows target to have the second component be 'pc'. Target: {}",
            target
        );

        if path.join("include/nvrtc.h").is_file() {
            return path;
        }
    }

    // No idea where to look for CUDA
    panic!("Cannot find CUDA NVRTC libraries");
}

#[cfg(not(feature = "cuda"))]
fn main() {}

#[cfg(feature = "cuda")]
fn main() {
    let cuda_path;
    if cfg!(target_os = "windows") {
        cuda_path = find_cuda_windows()
    } else {
        if let Some(cp) = find_cuda() {
            cuda_path = cp;
        } else {
            return;
        }
    };

    // Check for Windows
    if cfg!(target_os = "windows") {
        println!(
            "cargo:rustc-link-search=native={}\\lib\\x64",
            cuda_path.display()
        );
    } else {
        println!(
            "cargo:rustc-link-search=native={}/lib64",
            cuda_path.display()
        );
    }

    #[cfg(feature = "static")]
    {
        println!("cargo:rustc-link-lib=static=nvrtc_static");
        println!("cargo:rustc-link-lib=static=nvrtc-builtins_static");
        println!("cargo:rustc-link-lib=static=nvptxcompiler_static");
    }

    #[cfg(not(feature = "static"))]
    println!("cargo:rustc-link-lib=dylib=nvrtc");

    println!("cargo:rerun-if-changed=build.rs");
    if cfg!(target_os = "windows") {
        println!(
            "cargo:rustc-link-search=native={}",
            find_cuda_windows().display()
        );
    } else {
        for path in find_cuda2() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    };

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
}
