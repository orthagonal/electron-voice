fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    
    match target.as_str() {
        // Windows targets
        "x86_64-pc-windows-msvc" => {
            println!("cargo:rustc-link-search=native=./binaries/vosk-win64-0.3.45");
            println!("cargo:rustc-link-lib=dylib=libvosk");
        },
        // macOS targets
        "x86_64-apple-darwin" | "aarch64-apple-darwin" => {
            println!("cargo:rustc-link-search=native=./binaries/vosk-macos-0.3.45");
            println!("cargo:rustc-link-lib=dylib=vosk");
        },
        // Linux targets
        "x86_64-unknown-linux-gnu" => {
            println!("cargo:rustc-link-search=native=./binaries/vosk-linux-0.3.45");
            println!("cargo:rustc-link-lib=dylib=vosk");
        },
        _ => {
            panic!("Unsupported target for Steam deployment: {}", target);
        }
    }
}
