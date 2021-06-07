fn main() {
    println!("cargo:rustc-link-lib=dylib=cuda");
    let cuda_path = std::env::var("CUDA_LIB_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda-11.3/lib64/".to_string());
    println!("cargo:rustc-link-search=native={}", cuda_path);
}
