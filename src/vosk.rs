//! Thread-confined ownership of the Vosk C API. No native pointer escapes here.
//! Signatures match the bundled vosk_api.h and upstream Vosk 0.3.45.

use std::ffi::{c_char, c_int, CStr, CString};
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;

enum VoskModel {}
enum VoskRecognizer {}

extern "C" {
    fn vosk_set_log_level(level: c_int);
    fn vosk_model_new(path: *const c_char) -> *mut VoskModel;
    fn vosk_model_free(model: *mut VoskModel);
    fn vosk_recognizer_new(model: *mut VoskModel, rate: f32) -> *mut VoskRecognizer;
    fn vosk_recognizer_new_grm(
        model: *mut VoskModel,
        rate: f32,
        grammar: *const c_char,
    ) -> *mut VoskRecognizer;
    fn vosk_recognizer_free(recognizer: *mut VoskRecognizer);
    fn vosk_recognizer_set_max_alternatives(recognizer: *mut VoskRecognizer, count: c_int);
    fn vosk_recognizer_set_words(recognizer: *mut VoskRecognizer, words: c_int);
    fn vosk_recognizer_accept_waveform_s(
        recognizer: *mut VoskRecognizer,
        data: *const i16,
        length: c_int,
    ) -> c_int;
    fn vosk_recognizer_result(recognizer: *mut VoskRecognizer) -> *const c_char;
    fn vosk_recognizer_partial_result(recognizer: *mut VoskRecognizer) -> *const c_char;
    fn vosk_recognizer_reset(recognizer: *mut VoskRecognizer);
}

/// A unique allocation with its matching native destructor. Never Clone/Copy,
/// Send, or Sync: native handles stay on the thread that created them.
struct NativeHandle<T> {
    ptr: NonNull<T>,
    free: unsafe extern "C" fn(*mut T),
}

impl<T> NativeHandle<T> {
    /// # Safety
    /// A non-null pointer must be a uniquely owned allocation returned by the
    /// corresponding constructor, and `free` must be its matching destructor.
    unsafe fn from_raw(
        ptr: *mut T,
        free: unsafe extern "C" fn(*mut T),
        name: &str,
    ) -> Result<Self, String> {
        NonNull::new(ptr)
            .map(|ptr| Self { ptr, free })
            .ok_or_else(|| format!("Vosk could not create {name}"))
    }
}

impl<T> Drop for NativeHandle<T> {
    fn drop(&mut self) {
        // SAFETY: from_raw establishes unique ownership and the matching free
        // function. Drop runs once, including on early returns and unwinding.
        unsafe { (self.free)(self.ptr.as_ptr()) }
    }
}

pub(super) struct Model(NativeHandle<VoskModel>);

impl Model {
    pub(super) fn load(path: &str) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|_| "Model path contains a NUL byte")?;
        validate_model_files(Path::new(path))?;
        // SAFETY: path is NUL-terminated and lives through the call. A successful
        // constructor transfers one model reference to this handle.
        unsafe {
            NativeHandle::from_raw(vosk_model_new(c_path.as_ptr()), vosk_model_free, "model")
                .map(Self)
        }
    }
}

// Older bundled Windows builds can terminate the process rather than return
// null for missing model files. Follow Vosk's V1/V2 layout selection and reject
// incomplete installations before calling C++. This is not a content validator
// and cannot make corrupt native model data safe to load.
fn validate_model_files(root: &Path) -> Result<(), String> {
    let modern = root.join("am/final.mdl").is_file() && root.join("conf/model.conf").is_file();
    let legacy = root.join("final.mdl").is_file() && root.join("mfcc.conf").is_file();
    let (acoustic, config, graph) = if modern {
        (
            root.join("am/final.mdl"),
            root.join("conf"),
            root.join("graph"),
        )
    } else if legacy {
        (root.join("final.mdl"), root.to_owned(), root.to_owned())
    } else {
        return Err(format!("Invalid Vosk model directory {}: expected am/final.mdl and conf/model.conf, or final.mdl and mfcc.conf", root.display()));
    };
    check_file(&acoustic, true)?;
    if modern {
        check_file(&config.join("model.conf"), false)?;
    }
    let features = if config.join("mfcc.conf").is_file() {
        config.join("mfcc.conf")
    } else {
        config.join("fbank.conf")
    };
    check_file(&features, false)?;
    if graph.join("HCLG.fst").is_file() {
        check_file(&graph.join("HCLG.fst"), true)?;
    } else {
        check_file(&graph.join("HCLr.fst"), true)?;
        check_file(&graph.join("Gr.fst"), true)?;
        check_file(&graph.join("disambig_tid.int"), false)?;
    }
    if root.join("ivector/final.ie").exists() {
        for file in [
            "final.ie",
            "splice.conf",
            "online_cmvn.conf",
            "final.mat",
            "global_cmvn.stats",
            "final.dubm",
        ] {
            check_file(&root.join("ivector").join(file), false)?;
        }
    }
    if root.join("rescore/G.carpa").exists() {
        check_file(&root.join("rescore/G.carpa"), true)?;
        check_file(&root.join("rescore/G.fst"), true)?;
    }
    if root.join("rnnlm/final.raw").exists() {
        for file in [
            "final.raw",
            "word_feats.txt",
            "feat_embedding.final.mat",
            "special_symbol_opts.conf",
        ] {
            check_file(&root.join("rnnlm").join(file), false)?;
        }
    }
    Ok(())
}

fn check_file(path: &Path, nonempty: bool) -> Result<(), String> {
    let file = std::fs::File::open(path)
        .map_err(|error| format!("Cannot read Vosk model file {}: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("Cannot inspect Vosk model file {}: {error}", path.display()))?;
    if !metadata.is_file() || (nonempty && metadata.len() == 0) {
        return Err(format!(
            "Invalid or empty Vosk model file: {}",
            path.display()
        ));
    }
    Ok(())
}

pub(super) struct Recognizer<'model> {
    raw: NativeHandle<VoskRecognizer>,
    // Prevents freeing our model reference before the recognizer is destroyed.
    _model: PhantomData<&'model Model>,
}

impl<'model> Recognizer<'model> {
    pub(super) fn new(model: &'model Model, rate: f32, grammar: &str) -> Result<Self, String> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err("Vosk sample rate must be finite and positive".into());
        }
        let use_default = validate_grammar(grammar)?;
        let grammar = CString::new(grammar).map_err(|_| "Grammar contains a NUL byte")?;
        // SAFETY: the model is valid and borrowed for the recognizer lifetime;
        // grammar is NUL-terminated and Vosk consumes it during construction.
        let raw = unsafe {
            let ptr = if use_default {
                vosk_recognizer_new(model.0.ptr.as_ptr(), rate)
            } else {
                vosk_recognizer_new_grm(model.0.ptr.as_ptr(), rate, grammar.as_ptr())
            };
            NativeHandle::from_raw(ptr, vosk_recognizer_free, "recognizer")?
        };
        // SAFETY: raw is a live recognizer exclusively owned on this thread.
        // Preserve the existing recognition settings and result behavior.
        unsafe {
            vosk_recognizer_set_max_alternatives(raw.ptr.as_ptr(), 1);
            vosk_recognizer_set_words(raw.ptr.as_ptr(), 1);
        }
        Ok(Self {
            raw,
            _model: PhantomData,
        })
    }

    pub(super) fn accept(&mut self, samples: &[i16]) -> Result<bool, String> {
        let len = c_int::try_from(samples.len()).map_err(|_| "Audio chunk exceeds Vosk's limit")?;
        // SAFETY: the input slice contains len valid mono i16 samples and lives
        // through this synchronous call. &mut self excludes concurrent access.
        let status = unsafe {
            vosk_recognizer_accept_waveform_s(self.raw.ptr.as_ptr(), samples.as_ptr(), len)
        };
        match status {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(format!("Vosk failed to process audio (status {status})")),
        }
    }

    pub(super) fn result(&mut self, complete: bool) -> Result<String, String> {
        // SAFETY: the recognizer is live and exclusively borrowed. The result
        // is Vosk-owned; copy it before any further recognizer call or Drop.
        unsafe {
            let ptr = if complete {
                vosk_recognizer_result(self.raw.ptr.as_ptr())
            } else {
                vosk_recognizer_partial_result(self.raw.ptr.as_ptr())
            };
            copy_result(ptr)
        }
    }

    pub(super) fn reset(&mut self) {
        // SAFETY: the recognizer is live and exclusively borrowed.
        unsafe { vosk_recognizer_reset(self.raw.ptr.as_ptr()) }
    }
}

/// # Safety
/// If non-null, ptr must reference a readable NUL-terminated Vosk result until
/// this function returns. Its memory remains owned by Vosk and must not be freed.
unsafe fn copy_result(ptr: *const c_char) -> Result<String, String> {
    if ptr.is_null() {
        return Err("Vosk returned a null recognition result".into());
    }
    // SAFETY: caller provides a readable C string; null was checked above.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(str::to_owned)
        .map_err(|_| "Vosk returned a result that is not UTF-8".into())
}

pub(super) fn set_log_level(level: i32) {
    // SAFETY: this Vosk API takes an integer and retains no Rust memory.
    unsafe { vosk_set_log_level(level) }
}

pub(super) fn validate_grammar(grammar: &str) -> Result<bool, String> {
    if grammar.is_empty() {
        return Ok(true);
    }
    if grammar.contains('\0') {
        return Err("Grammar contains a NUL byte".into());
    }
    let phrases: Vec<String> = serde_json::from_str(grammar)
        .map_err(|_| "Grammar must be a JSON array of strings, or an empty string")?;
    if phrases.iter().any(|phrase| phrase.trim().is_empty()) {
        return Err("Grammar phrases must contain words".into());
    }
    // Vosk's grammar constructor can dereference an unset graph for []. Use
    // its ordinary constructor for an empty grammar instead.
    Ok(phrases.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    unsafe extern "C" fn free_counter(ptr: *mut Arc<AtomicUsize>) {
        // SAFETY: test allocates this pointer with Box::into_raw exactly once.
        let counter = unsafe { Box::from_raw(ptr) };
        counter.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn unique_native_handle_frees_once_on_early_return() {
        fn allocate_then_fail(counter: Arc<AtomicUsize>) -> Result<(), String> {
            let ptr = Box::into_raw(Box::new(counter));
            // SAFETY: matching Box allocation and destructor, unique pointer.
            let _handle = unsafe { NativeHandle::from_raw(ptr, free_counter, "test") }?;
            Err("initialization failed later".into())
        }
        let counter = Arc::new(AtomicUsize::new(0));
        assert!(allocate_then_fail(Arc::clone(&counter)).is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn null_native_handle_is_rejected_without_freeing() {
        // SAFETY: null explicitly represents constructor failure.
        assert!(
            unsafe { NativeHandle::from_raw(std::ptr::null_mut(), free_counter, "test") }.is_err()
        );
    }

    #[test]
    fn null_and_invalid_utf8_results_are_recoverable() {
        // SAFETY: null is a supported error, and bytes is a live C string.
        unsafe {
            assert!(copy_result(std::ptr::null()).is_err());
            let bytes = [0xff_u8, 0];
            assert!(copy_result(bytes.as_ptr().cast()).is_err());
        }
    }

    #[test]
    fn native_result_is_copied_without_taking_ownership() {
        let result = CString::new(r#"{"partial":"hello"}"#).unwrap();
        // SAFETY: the CString remains live while its bytes are copied.
        let copied = unsafe { copy_result(result.as_ptr()) }.unwrap();
        drop(result);
        assert_eq!(copied, r#"{"partial":"hello"}"#);
    }

    #[test]
    fn model_path_with_nul_is_rejected_before_entering_vosk() {
        assert!(Model::load("invalid\0path").is_err());
    }

    #[test]
    fn invalid_model_is_rejected_without_aborting_in_native_code() {
        let directory =
            std::env::temp_dir().join(format!("electron-voice-empty-model-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let result = Model::load(directory.to_str().unwrap());
        std::fs::remove_dir(&directory).unwrap();
        assert!(result.is_err());
    }

    struct ModelFixture(std::path::PathBuf);

    impl ModelFixture {
        fn new() -> Self {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let directory = std::env::temp_dir().join(format!(
                "electron-voice-layout-{}-{stamp}",
                std::process::id()
            ));
            std::fs::create_dir(&directory).unwrap();
            Self(directory)
        }

        fn file(&self, name: &str) {
            let path = self.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"layout fixture, never passed to Vosk").unwrap();
        }
    }

    impl Drop for ModelFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn preflight_accepts_both_official_layouts_and_feature_types() {
        for files in [
            vec!["final.mdl", "mfcc.conf", "HCLG.fst"],
            vec![
                "am/final.mdl",
                "conf/model.conf",
                "conf/mfcc.conf",
                "graph/HCLG.fst",
            ],
            vec![
                "am/final.mdl",
                "conf/model.conf",
                "conf/fbank.conf",
                "graph/HCLr.fst",
                "graph/Gr.fst",
                "graph/disambig_tid.int",
            ],
        ] {
            let fixture = ModelFixture::new();
            for file in files {
                fixture.file(file);
            }
            assert!(validate_model_files(&fixture.0).is_ok());
        }
    }

    #[test]
    fn preflight_rejects_missing_graphs_and_incomplete_optional_features() {
        let fixture = ModelFixture::new();
        fixture.file("final.mdl");
        fixture.file("mfcc.conf");
        assert!(validate_model_files(&fixture.0).is_err());
        fixture.file("HCLG.fst");
        assert!(validate_model_files(&fixture.0).is_ok());
        fixture.file("ivector/final.ie");
        assert!(validate_model_files(&fixture.0).is_err());
    }

    #[test]
    fn empty_or_blank_grammar_is_handled_without_an_invalid_native_graph() {
        assert!(validate_grammar("").unwrap());
        assert!(validate_grammar(" [ ] ").unwrap());
        assert!(!validate_grammar("[\"hello\",\"世界\"]").unwrap());
        for grammar in ["[\"\"]", "[\"  \t\"]", "[42]", "{}", "bad\0grammar"] {
            assert!(validate_grammar(grammar).is_err());
        }
    }
}
