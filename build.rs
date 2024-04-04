fn main() {
  println!("cargo:rustc-link-search=native=./binaries/vosk-win64-0.3.45"); 
  println!("cargo:rustc-link-lib=dylib=libvosk"); // Link against `libvosk.dll` (the `lib` prefix and `.dll` suffix are not needed)
}
