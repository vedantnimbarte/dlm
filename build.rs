//! Build script for `dlm`.
//!
//! Links the selected GPU runtime so the FFI in `src/gpu/` resolves at link
//! time: `cuda` → `cudart` (honours `CUDA_PATH`), `rocm` → `amdhip64` (honours
//! `ROCM_PATH`). With neither feature this is a no-op and the crate builds
//! anywhere — the storage engine and VRAM math have no GPU dependency.

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            // Toolkit layout varies: NVIDIA .run/Windows → lib64 / lib\x64;
            // distro packages (CUDA_PATH=/usr) → multiarch lib/<triple> or lib.
            println!("cargo:rustc-link-search=native={cuda_path}/lib64");
            println!("cargo:rustc-link-search=native={cuda_path}/lib/x64");
            println!("cargo:rustc-link-search=native={cuda_path}/lib/x86_64-linux-gnu");
            println!("cargo:rustc-link-search=native={cuda_path}/lib/aarch64-linux-gnu");
            println!("cargo:rustc-link-search=native={cuda_path}/lib");
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    if std::env::var("CARGO_FEATURE_ROCM").is_ok() {
        // ROCm default install is /opt/rocm; libs live under lib.
        let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        println!("cargo:rustc-link-search=native={rocm_path}/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }

    if std::env::var("CARGO_FEATURE_CUDA_KERNELS").is_ok() {
        compile_cuda_kernels();
        println!("cargo:rerun-if-changed=src/gpu/kernels.cu");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}

/// Compile `src/gpu/kernels.cu` into a static library with nvcc and link it.
///
/// Degrades gracefully: if nvcc is not found we emit a warning and skip, so
/// `cargo check --features cuda-kernels` type-checks the Rust FFI without the
/// toolkit. Linking a binary that actually calls the kernels then requires nvcc.
fn compile_cuda_kernels() {
    let nvcc = std::env::var("CUDA_PATH")
        .map(|p| format!("{p}/bin/nvcc"))
        .unwrap_or_else(|_| "nvcc".to_string());

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let lib_path = format!("{out_dir}/libdlm_kernels.a");

    let status = std::process::Command::new(&nvcc)
        .args(["-O3", "-Xcompiler", "-fPIC", "-lib", "src/gpu/kernels.cu", "-o"])
        .arg(&lib_path)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=static=dlm_kernels");
        }
        Ok(s) => {
            println!("cargo:warning=nvcc failed to compile src/gpu/kernels.cu (exit {s}); GPU kernels will be unresolved at link time");
        }
        Err(_) => {
            println!("cargo:warning=cuda-kernels enabled but nvcc was not found; skipping device-code compilation (cargo check still type-checks the FFI)");
        }
    }
}
