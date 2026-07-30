//! Build script for `dlm`.
//!
//! Links the selected GPU runtime so the FFI in `src/gpu/` resolves at link
//! time: `cuda` → `cudart` (honours `CUDA_PATH`), `rocm` → `amdhip64` (honours
//! `ROCM_PATH`). With neither feature this is a no-op and the crate builds
//! anywhere — the storage engine and VRAM math have no GPU dependency.

fn main() {
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");

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
        if std::env::var("CARGO_FEATURE_CUDA_STATIC").is_ok() {
            // Static CUDA runtime: baked into the binary so it needs only the
            // NVIDIA driver at runtime, not the toolkit. On Linux cudart_static
            // pulls in culibos + the usual system libs; on Windows the import
            // lib name is the same and the CRT covers the rest.
            println!("cargo:rustc-link-lib=static=cudart_static");
            if !windows {
                println!("cargo:rustc-link-lib=static=culibos");
                println!("cargo:rustc-link-lib=dylib=rt");
                println!("cargo:rustc-link-lib=dylib=pthread");
                println!("cargo:rustc-link-lib=dylib=dl");
                // CUDA 13's cudart_static drags in C++ runtime code (__cxa_guard_*,
                // __gxx_personality_v0) that 12.x did not; link libstdc++ so those
                // resolve. Emitted last so it satisfies cudart_static's references.
                println!("cargo:rustc-link-lib=dylib=stdc++");
            }
        } else {
            println!("cargo:rustc-link-lib=dylib=cudart");
        }
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

    if std::env::var("CARGO_FEATURE_ROCM_KERNELS").is_ok() {
        compile_hip_kernels();
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
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");

    // MSVC produces `dlm_kernels.lib` and has no `-fPIC` (position-independent
    // code is the only mode); passing it through -Xcompiler makes cl.exe fail.
    let lib_path = if windows {
        format!("{out_dir}/dlm_kernels.lib")
    } else {
        format!("{out_dir}/libdlm_kernels.a")
    };

    let mut cmd = std::process::Command::new(&nvcc);
    cmd.arg("-O3");
    // nvcc hard-errors when the host compiler is newer than the CUDA release
    // officially knows about (e.g. CUDA 12.5's nvcc vs MSVC 14.5x / VS 18 on the
    // Windows CI runner). kernels.cu is plain CUDA C with no STL surface, so the
    // version guard is spurious here — waive it so a newer toolchain still builds.
    cmd.arg("--allow-unsupported-compiler");
    if windows {
        // nvcc shells out to the MSVC host compiler and requires cl.exe on PATH.
        // rustc finds link.exe through its own MSVC probe, so a working Rust build
        // does NOT imply cl.exe is on PATH — point nvcc at it explicitly.
        if let Some(dir) = find_msvc_bin() {
            cmd.arg("-ccbin").arg(dir);
        }
    } else {
        cmd.args(["-Xcompiler", "-fPIC"]);
    }
    let output = cmd.args(["-lib", "src/gpu/kernels.cu", "-o"]).arg(&lib_path).output();

    match output {
        Ok(o) if o.status.success() => {
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=static=dlm_kernels");
        }
        Ok(o) => {
            // Surface nvcc's own diagnostics — the exit code alone can't tell a
            // host-compiler/toolkit mismatch from a missing header or a bad flag.
            for line in String::from_utf8_lossy(&o.stderr).lines() {
                println!("cargo:warning=nvcc: {line}");
            }
            for line in String::from_utf8_lossy(&o.stdout).lines() {
                println!("cargo:warning=nvcc: {line}");
            }
            println!("cargo:warning=nvcc failed to compile src/gpu/kernels.cu (exit {}); GPU kernels will be unresolved at link time", o.status);
        }
        Err(_) => {
            println!("cargo:warning=cuda-kernels enabled but nvcc was not found; skipping device-code compilation (cargo check still type-checks the FFI)");
        }
    }
}

/// Compile `src/gpu/kernels.cu` into a static library with **hipcc** (AMD) and
/// link it. The same source compiles under HIP — its CUDA runtime/fp16 includes
/// map to the HIP equivalents via `#ifdef __HIP_PLATFORM_AMD__` guards in the
/// file, so `hipcc -x cu` builds it directly.
///
/// Degrades gracefully like the nvcc path: a missing hipcc emits a warning and
/// skips, so `cargo check --features rocm-kernels` type-checks the Rust FFI + HIP
/// backend without the toolkit. Linking a running binary then requires hipcc.
fn compile_hip_kernels() {
    let hipcc = std::env::var("ROCM_PATH")
        .map(|p| format!("{p}/bin/hipcc"))
        .unwrap_or_else(|_| "hipcc".to_string());
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let lib_path = format!("{out_dir}/libdlm_kernels.a");

    let output = std::process::Command::new(&hipcc)
        .args(["-O3", "-fPIC", "-x", "cu", "-c", "src/gpu/kernels.cu", "-o"])
        .arg(format!("{out_dir}/dlm_kernels.o"))
        .output();
    // hipcc has no `-lib`; archive the object into a static lib with `ar`.
    let archived = matches!(&output, Ok(o) if o.status.success())
        && std::process::Command::new("ar")
            .args(["crs", &lib_path, &format!("{out_dir}/dlm_kernels.o")])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

    match output {
        Ok(o) if o.status.success() && archived => {
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=static=dlm_kernels");
        }
        Ok(o) => {
            for line in String::from_utf8_lossy(&o.stderr).lines() {
                println!("cargo:warning=hipcc: {line}");
            }
            println!("cargo:warning=hipcc failed to compile src/gpu/kernels.cu (exit {}); GPU kernels will be unresolved at link time", o.status);
        }
        Err(_) => {
            println!("cargo:warning=rocm-kernels enabled but hipcc was not found; skipping device-code compilation (cargo check still type-checks the FFI)");
        }
    }
}

/// Directory holding the MSVC `cl.exe`, located through the Visual Studio
/// installer's `vswhere`. Returns `None` when VS is absent or cl.exe is already
/// resolvable, in which case nvcc's own PATH lookup applies.
#[cfg(windows)]
fn find_msvc_bin() -> Option<std::path::PathBuf> {
    let program_files =
        std::env::var("ProgramFiles(x86)").unwrap_or_else(|_| r"C:\Program Files (x86)".into());
    let vswhere =
        std::path::Path::new(&program_files).join("Microsoft Visual Studio/Installer/vswhere.exe");

    let out = std::process::Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    let root = String::from_utf8(out.stdout).ok()?;
    let msvc = std::path::Path::new(root.trim()).join("VC/Tools/MSVC");

    // Pick the highest installed toolset version.
    let mut versions: Vec<_> = std::fs::read_dir(&msvc)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    versions.sort();
    let bin = versions.last()?.join("bin/Hostx64/x64");
    bin.join("cl.exe").exists().then_some(bin)
}

#[cfg(not(windows))]
fn find_msvc_bin() -> Option<std::path::PathBuf> {
    None
}
