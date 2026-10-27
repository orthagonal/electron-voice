//! Native speech recognition for Electron. JavaScript calls only configure or
//! schedule work; one worker owns each environment's CPAL/Vosk resources.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use neon::event::Channel;
use neon::prelude::*;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

mod audio;
mod recognition;
mod vosk;

#[cfg(test)]
mod tests;

#[derive(serde::Serialize, serde::Deserialize)]
struct VoskPartialResult {
    partial: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct VoskTextResult {
    text: String,
}

fn partial_is_only_the_tokens(words: &[String]) -> bool {
    !words.is_empty() && words.iter().all(|word| word == "the")
}

#[derive(Default)]
struct Session {
    cancelled: Arc<AtomicBool>,
    finished: AtomicBool,
    pending_callbacks: AtomicUsize,
}

impl Session {
    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn is_done(&self) -> bool {
        self.is_cancelled() || self.finished.load(Ordering::Acquire)
    }
}

struct StartRequest {
    session: Arc<Session>,
    sink: Arc<JsSink>,
    mic: String,
    model: String,
    grammar: String,
    max_words: usize,
}

struct AppState {
    is_running: bool,
    model_is_loaded: bool,
    name_of_mic: String,
    path_to_model: String,
    grammar: String,
    sample_rate: f32,
    current: Option<Arc<Session>>,
    pending: Option<StartRequest>,
    worker_running: bool,
    worker: Option<JoinHandle<()>>,
    closed: bool,
    words: WordsToLookFor,
}

impl AppState {
    fn new() -> Self {
        Self {
            is_running: false,
            model_is_loaded: false,
            name_of_mic: "default".into(),
            path_to_model: String::new(),
            grammar: String::new(),
            sample_rate: 16000.0,
            current: None,
            pending: None,
            worker_running: false,
            worker: None,
            closed: false,
            words: WordsToLookFor::new(),
        }
    }

    fn begin(&mut self, session: Arc<Session>) -> bool {
        if self.is_running || self.closed {
            return false;
        }
        self.is_running = true;
        self.model_is_loaded = false;
        if let Some(previous) = self.current.take() {
            previous.cancel();
        }
        self.current = Some(session);
        true
    }

    fn owns(&self, session: &Arc<Session>) -> bool {
        self.current
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, session))
    }

    fn ready(&mut self, session: &Arc<Session>, sample_rate: f32) -> bool {
        if !self.owns(session) || session.is_cancelled() {
            return false;
        }
        self.sample_rate = sample_rate;
        self.model_is_loaded = true;
        true
    }

    fn complete(&mut self, session: &Arc<Session>) -> bool {
        session.finished.store(true, Ordering::Release);
        if !self.owns(session) {
            return false;
        }
        self.is_running = false;
        self.model_is_loaded = false;
        // Retain the last token until stop/restart so either operation can also
        // suppress a failure notice that is still queued on the JS event loop.
        true
    }

    fn stop(&mut self) {
        if let Some(session) = self.current.take() {
            session.cancel();
        }
        self.pending = None;
        self.is_running = false;
        self.model_is_loaded = false;
    }
}

struct WordsToLookFor {
    is_active: bool,
}

impl WordsToLookFor {
    fn new() -> Self {
        Self { is_active: true }
    }
}

// A Neon LocalKey is scoped to the JS environment, including Node workers.
// Its destructor cancels capture on environment teardown without joining on JS.
static ENVIRONMENT: neon::thread::LocalKey<Environment> = neon::thread::LocalKey::new();

struct Environment(Arc<Mutex<AppState>>);

impl Drop for Environment {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        state.closed = true;
        state.stop();
    }
}

fn app_state<'a, C: Context<'a>>(cx: &mut C) -> Arc<Mutex<AppState>> {
    Arc::clone(
        &ENVIRONMENT
            .get_or_init(cx, || Environment(Arc::new(Mutex::new(AppState::new()))))
            .0,
    )
}

struct JsSink {
    callback: Arc<Root<JsFunction>>,
    channel: Channel,
}

// Bounds Neon work if Electron temporarily stops servicing its event loop.
// Backpressure happens only on the inference worker, never the audio callback.
const MAX_PENDING_CALLBACKS: usize = 256;

struct DeliveryPermit(Arc<Session>);

impl Drop for DeliveryPermit {
    fn drop(&mut self) {
        self.0.pending_callbacks.fetch_sub(1, Ordering::AcqRel);
    }
}

impl JsSink {
    fn words(&self, session: &Arc<Session>, words: Vec<String>) -> Result<(), String> {
        while session.pending_callbacks.load(Ordering::Acquire) >= MAX_PENDING_CALLBACKS {
            if session.is_done() {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(2));
        }
        if session.is_done() {
            return Ok(());
        }
        session.pending_callbacks.fetch_add(1, Ordering::AcqRel);
        let permit = DeliveryPermit(Arc::clone(session));
        let callback = Arc::clone(&self.callback);
        self.channel
            .try_send(move |mut cx| {
                let _permit = permit;
                if _permit.0.is_done() {
                    return Ok(());
                }
                let callback = callback.to_inner(&mut cx);
                let array = JsArray::new(&mut cx, words.len());
                for (index, word) in words.iter().enumerate() {
                    let word = cx.string(word);
                    array.set(&mut cx, index as u32, word)?;
                }
                let this = cx.undefined();
                if cx
                    .try_catch(|cx| callback.call(cx, this, [array.upcast::<JsValue>()]))
                    .is_err()
                {
                    emit_warning(
                        &mut cx,
                        "The speech-recognition callback threw an exception",
                    )?;
                }
                Ok(())
            })
            .map_err(|_| {
                "The JavaScript environment stopped accepting speech results".to_string()
            })?;
        Ok(())
    }

    fn warning(&self, session: &Arc<Session>, message: String) {
        // Failed scheduling means the JS environment is closing. Cancellation
        // and native cleanup must proceed without waiting for its event loop.
        let session = Arc::clone(session);
        let _ = self.channel.try_send(move |mut cx| {
            if session.is_cancelled() {
                return Ok(());
            }
            emit_warning(&mut cx, &message)
        });
    }
}

fn emit_warning<'a, C: Context<'a>>(cx: &mut C, message: &str) -> NeonResult<()> {
    let process: Handle<JsObject> = cx.global("process")?;
    let emit: Handle<JsFunction> = process.get(cx, "emitWarning")?;
    let error = cx.error(message)?;
    let code = cx.string("ELECTRON_VOICE_ERROR");
    error.set(cx, "code", code)?;
    emit.call(cx, process, [error.upcast::<JsValue>()])?;
    Ok(())
}

fn list_devices(mut cx: FunctionContext) -> JsResult<JsArray> {
    let devices = match cpal::default_host().input_devices() {
        Ok(devices) => devices,
        Err(error) => return cx.throw_error(format!("Could not enumerate microphones: {error}")),
    };
    let array = JsArray::new(&mut cx, 0);
    for (index, device) in devices.enumerate() {
        let name = match device.name() {
            Ok(name) => name,
            Err(error) => {
                return cx.throw_error(format!("Could not read microphone name: {error}"))
            }
        };
        let name = cx.string(name);
        array.set(&mut cx, index as u32, name)?;
    }
    Ok(array)
}

fn set_mic_name(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let name = cx.argument::<JsString>(0)?.value(&mut cx);
    app_state(&mut cx).lock().name_of_mic = name;
    Ok(cx.undefined())
}

fn set_path_to_model(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let path = cx.argument::<JsString>(0)?.value(&mut cx);
    app_state(&mut cx).lock().path_to_model = path;
    Ok(cx.undefined())
}

fn set_grammar(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let grammar = cx.argument::<JsString>(0)?.value(&mut cx);
    app_state(&mut cx).lock().grammar = grammar;
    Ok(cx.undefined())
}

fn validate_start(model: &str, grammar: &str, max_words: f64) -> Result<usize, String> {
    // Preserve fractional-number truncation, while rejecting nonsensical casts.
    if !max_words.is_finite() || max_words < 0.0 || max_words >= usize::MAX as f64 {
        return Err(
            "max_words must be a finite nonnegative number within the platform limit".into(),
        );
    }
    if model.is_empty() || model.contains('\0') {
        return Err("Set a nonempty model path without NUL bytes before startListener".into());
    }
    vosk::validate_grammar(grammar)?;
    Ok(max_words as usize)
}

fn start_listener(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    // Validate JS argument types before creating roots, channels, or threads.
    let callback = cx.argument::<JsFunction>(0)?;
    let max_words = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let shared = app_state(&mut cx);
    let mut state = shared.lock();
    if state.is_running {
        return Ok(cx.undefined()); // An active/starting listener is idempotent.
    }
    let max_words = match validate_start(&state.path_to_model, &state.grammar, max_words) {
        Ok(value) => value,
        Err(error) => return cx.throw_error(error),
    };
    let session = Arc::new(Session::default());
    if !state.begin(Arc::clone(&session)) {
        return cx.throw_error("The speech-recognition environment is shutting down");
    }
    let mut channel = cx.channel();
    channel.unref(&mut cx);
    state.pending = Some(StartRequest {
        session,
        sink: Arc::new(JsSink {
            callback: Arc::new(callback.root(&mut cx)),
            channel,
        }),
        mic: state.name_of_mic.clone(),
        model: state.path_to_model.clone(),
        grammar: state.grammar.clone(),
        max_words,
    });
    if !state.worker_running {
        state.worker_running = true;
        let previous = state.worker.take();
        let worker_state = Arc::clone(&shared);
        match thread::Builder::new()
            .name("electron-voice".into())
            .spawn(move || {
                // Only a finishing worker can remain here; joining stays off JS.
                if let Some(previous) = previous {
                    let _ = previous.join();
                }
                worker_loop(worker_state);
            }) {
            Ok(worker) => state.worker = Some(worker),
            Err(error) => {
                state.stop();
                state.worker_running = false;
                return cx.throw_error(format!(
                    "Could not start speech-recognition worker: {error}"
                ));
            }
        }
    }
    Ok(cx.undefined())
}

fn worker_loop(shared: Arc<Mutex<AppState>>) {
    loop {
        let request = {
            let mut state = shared.lock();
            match state.pending.take() {
                Some(request) => request,
                None => {
                    state.worker_running = false;
                    return;
                }
            }
        };
        // Unwind Rust failures at the thread boundary so state is never stuck
        // active. This cannot recover a process abort inside a native library.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listen(&shared, &request)))
                .unwrap_or_else(|_| Err("Speech-recognition worker panicked".into()));
        let was_cancelled = request.session.is_cancelled();
        let was_current = shared.lock().complete(&request.session);
        if !was_cancelled && was_current {
            if let Err(error) = result {
                request.sink.warning(&request.session, error);
            }
        }
        // All stream/Vosk resources have been dropped before the next request.
    }
}

fn listen(shared: &Arc<Mutex<AppState>>, request: &StartRequest) -> Result<(), String> {
    if request.session.is_cancelled() {
        return Ok(());
    }
    // Snapshot configuration at start. Do not open capture while a large model
    // loads; this prevents an initial backlog of stale audio.
    if !std::path::Path::new(&request.model).is_dir() {
        return Err(format!(
            "Vosk model directory does not exist: {}",
            request.model
        ));
    }
    let model = vosk::Model::load(&request.model)?;
    if request.session.is_cancelled() {
        return Ok(());
    }
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .map_err(|e| format!("Could not enumerate microphones: {e}"))?;
    // Retain the existing fallback to the system default if a name is missing.
    let device = devices
        .filter_map(|device| device.name().ok().map(|name| (name, device)))
        .find_map(|(name, device)| (name == request.mic).then_some(device))
        .or_else(|| host.default_input_device())
        .ok_or("No microphone is available")?;
    let config = device
        .default_input_config()
        .map_err(|e| format!("Could not configure microphone: {e}"))?;
    let sample_rate = config.sample_rate().0 as f32;
    let mut recognizer = vosk::Recognizer::new(&model, sample_rate, &request.grammar)?;
    if request.session.is_cancelled() {
        return Ok(());
    }
    let (stream, input) = audio::open(&device, config, Arc::clone(&request.session.cancelled))?;
    stream
        .play()
        .map_err(|e| format!("Could not start microphone: {e}"))?;
    if !shared.lock().ready(&request.session, sample_rate) {
        return Ok(());
    }
    let mut processor = recognition::RecognitionProcessor::default();
    let mut warned_about_overflow = false;
    let mut warned_about_result = false;
    while !request.session.is_cancelled() {
        if !warned_about_overflow && input.dropped.load(Ordering::Relaxed) > 0 {
            warned_about_overflow = true;
            request.sink.warning(&request.session, "Speech inference fell behind the two-second audio queue; some new audio was dropped".into());
        }
        let samples = match input.next(&request.session.cancelled)? {
            Some(samples) => samples,
            None => continue,
        };
        if request.session.is_cancelled() {
            break;
        }
        let complete = recognizer.accept(&samples)?;
        input.recycle(samples);
        if request.session.is_cancelled() || !shared.lock().words.is_active {
            continue;
        }
        let result = recognizer.result(complete)?;
        let mut delivery = Ok(());
        processor.process(
            &result,
            complete,
            request.max_words,
            |words| delivery = request.sink.words(&request.session, words),
            || recognizer.reset(),
        );
        delivery?;
        if let Some(error) = processor.take_error() {
            // Keep partial recognition alive after malformed/native-schema
            // errors. One warning per session avoids flooding Electron logs.
            if !warned_about_result {
                warned_about_result = true;
                request.sink.warning(&request.session, error);
            }
        }
    }
    // Stream drops first (stopping callbacks), then recognizer, then model.
    // Stop does not flush additional words into a cancelled game session.
    drop(stream);
    Ok(())
}

fn stop_listener(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    app_state(&mut cx).lock().stop();
    Ok(cx.undefined())
}

fn look_for_words(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    app_state(&mut cx).lock().words.is_active = true;
    Ok(cx.undefined())
}

fn is_model_loaded(mut cx: FunctionContext) -> JsResult<JsBoolean> {
    let ready = app_state(&mut cx).lock().model_is_loaded;
    Ok(cx.boolean(ready))
}

fn set_log_level(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let level = cx.argument::<JsNumber>(0)?.value(&mut cx);
    if !level.is_finite() || level < i32::MIN as f64 || level > i32::MAX as f64 {
        return cx.throw_range_error("Log level must be a finite 32-bit number");
    }
    vosk::set_log_level(level as i32);
    Ok(cx.undefined())
}

// Adapters retain the original vector helper contracts for characterization
// tests; sample arithmetic is shared with the pooled live capture path.
#[cfg(test)]
fn convert_f32_to_16(input: &[f32]) -> Vec<i16> {
    use audio::PcmSample;
    input.iter().map(|sample| sample.pcm16()).collect()
}

#[cfg(test)]
fn convert_32_to_16<T: cpal::Sample>(input: &[T]) -> Vec<i16>
where
    i16: cpal::FromSample<T>,
{
    input
        .iter()
        .map(|&sample| audio::convert_sample(sample))
        .collect()
}

#[cfg(test)]
fn convert_stereo_to_mono(input: &[i16]) -> Vec<i16> {
    let (pairs, remainder) = input.as_chunks::<2>();
    if !remainder.is_empty() {
        return input.to_vec();
    }
    pairs
        .iter()
        .map(|pair| (audio::stereo_component(pair[0]) + audio::stereo_component(pair[1])) as i16)
        .collect()
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("setLogLevel", set_log_level)?;
    cx.export_function("startListener", start_listener)?;
    cx.export_function("stopListener", stop_listener)?;
    cx.export_function("listDevices", list_devices)?;
    cx.export_function("setMicName", set_mic_name)?;
    cx.export_function("setPathToModel", set_path_to_model)?;
    cx.export_function("lookForWords", look_for_words)?;
    cx.export_function("isModelLoaded", is_model_loaded)?;
    cx.export_function("setGrammar", set_grammar)?;
    Ok(())
}
