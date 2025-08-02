fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");

    tonic_build::configure().build_server(true).compile_protos(
        &[
            "../ek-proto/ek/control/v1/control.proto",
            "../ek-proto/ek/worker/v1/expert.proto",
            "../ek-proto/ek/object/v1/object.proto",
            "../ek-proto/onnx/onnx.proto",
        ],
        &["../ek-proto"],
    )?;

    eprintln!("protobuf built");
    println!("cargo:rerun-if-changed=ops/stream.cc");
    println!("cargo:rerun-if-changed=ops/stream.h");
    cxx_build::bridge("src/ffn/expert_torch/stream.rs")
        .file("ops/stream.cc")
        .flag_if_supported("-std=c++17")
        .includes([
            format!("{}/include", std::env::var("LIBTORCH")?),
            format!("{}/include", std::env::var("CUDA_PATH")?),
        ])
        .warnings(false)
        .compile("ek_torch_stream");

    println!(
        "cargo:rustc-link-search=native={}/lib",
        std::env::var("CUDA_PATH")?
    );
    println!(
        "cargo:rustc-link-search=native={}/lib64",
        std::env::var("CUDA_PATH")?
    );
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
    println!("cargo:rustc-link-arg=-ltorch");

    println!(
        "cargo:rustc-link-search=native={}/lib",
        std::env::var("LIBTORCH")?
    );
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-link-lib=c10_cuda");

    Ok(())
}
