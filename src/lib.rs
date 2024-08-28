// js binding stuff:
use neon::prelude::*;
use neon::types::JsFunction;

// c++ binding stuff for the vosk library:
use std::ffi::{c_char, c_int, CStr, CString};

// rust thread control stuff:
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::sync::Mutex as StdMutex;
use std::sync::Arc;
use std::thread;

// audio input and processing stuff:
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait, },
    SampleFormat
};
use dasp::{sample::ToSample, Sample};

#[derive(serde::Serialize, serde::Deserialize)]
struct VoskPartialResult {
    partial: String,
}

// some vosk stubs
enum VoskModel {}
enum VoskRecognizer {}

// controls the thread state:
struct AppState {
    // audio device is selected and actively pulling in data:
    is_running: bool, 
    // vosk model is loaded and ready to process audio:
    model_is_loaded: bool,
    // name of the audio device to listen to:
    name_of_mic: String,
    // path to the vosk model file on disk 
    // (see https://alphacephei.com/vosk/models)
    path_to_model: String,
    // optional, a grammar is a list of words to expect.
    // When you pass a grammar in, the model will only look for the words in that grammar.
    // Use a grammar when you only need recognize a specific set of keywords 
    // (i.e. you are implementing a voice user interface). 
    // Leave it blank to do general speech-to-text transcription.
    grammar: String,
    // sample rate of the audio device, you have to know the sample_rate 
    // before you can initialize the vosk voice recognizer 
    sample_rate: f32,
}

impl AppState {
    fn new() -> Self {
        AppState {
            is_running: false,
            model_is_loaded: false,
            name_of_mic: String::from("default"),
            path_to_model: String::from(""),
            grammar: String::from(""),
            sample_rate: 16000.0,
        }
    }
}

struct WordsToLookFor {
    // whether we're scanning for a word currently
    is_active: bool,
}

impl WordsToLookFor {
    fn new() -> Self {
        WordsToLookFor {
            is_active: true,
        }
    }
}

// the one true app state
// these are the global variables that are shared between threads
lazy_static! {
    static ref APP_STATE: Mutex<AppState> = Mutex::new(AppState::new());
    static ref LISTEN_FOR_WORDS: StdMutex<WordsToLookFor> = StdMutex::new(WordsToLookFor::new());
}

/*
     __          _       __  __    _____       _             __
  /\ \ \___   __| | ___  \ \/ _\   \_   \_ __ | |_ ___ _ __ / _| __ _  ___ ___
 /  \/ / _ \ / _` |/ _ \  \ \ \     / /\| '_ \| __/ _ | '__| |_ / _` |/ __/ _ \
/ /\  | (_) | (_| |  __/\_/ _\ \ /\/ /_ | | | | ||  __| |  |  _| (_| | (_|  __/
\_\ \/ \___/ \__,_|\___\___/\__/ \____/ |_| |_|\__\___|_|  |_|  \__,_|\___\___|

this section defines functions that you can access from nodejs
**************************************************************/

// returns a js array listing the audio devices on your system
fn list_devices(mut cx: FunctionContext) -> JsResult<JsArray> {
    let host = cpal::default_host();
    let devices = host.input_devices().expect("Failed to get input devices");
    let js_array = JsArray::new(&mut cx, 0);
    for (device_index, device) in devices.enumerate() {
        let name = device.name().unwrap();
        let js_string = cx.string(name);
        js_array.set(&mut cx, device_index as u32, js_string)?;
    }
    Ok(js_array)
}

// select an audio device to listen to
fn set_mic_name(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let name = cx.argument::<JsString>(0)?.value(&mut cx);

    {
        let mut state = APP_STATE.lock();
        state.name_of_mic = name;
    }

    Ok(cx.undefined())
}

// set the file path to the vosk speech model
fn set_path_to_model(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let name = cx.argument::<JsString>(0)?.value(&mut cx);
    {
        let mut state = APP_STATE.lock();
        state.path_to_model = name;
    }

    Ok(cx.undefined())
}

// grammar is a JSON array of words/phrases in string form  : "[\"hello world\", \"electron\", \"voice\"]"
fn set_grammar(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let grammar = cx.argument::<JsString>(0)?.value(&mut cx);
    {
        let mut state = APP_STATE.lock();
        state.grammar = grammar;
    }

    Ok(cx.undefined())
}

// start the listening and interpreting threads
fn start_listener(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    // set up the audio
    let desired_input_device_name = {
        let state = APP_STATE.lock();
        state.name_of_mic.clone()
    };
    // one channel receives raw audio data, the other receives recognized words from that audio
    let channel = cx.channel();
    let (transmit_audio_channel, receive_audio_channel) = std::sync::mpsc::channel();
    let (transmit_words_channel, receive_words_channel) = std::sync::mpsc::channel();

    // js callback to call when words are found
    let all_words_callback = cx.argument::<JsFunction>(0)?.root(&mut cx);
    let max_words = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;

    let all_words_callback_shared = Arc::new(all_words_callback);
    let channel_shared = Arc::new(channel);

    // start the producer thread, this thread opens and listens to your microphone
    thread::spawn(move || {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .unwrap()
            .find(|d| d.name().unwrap() == desired_input_device_name)
            .or_else(|| host.default_input_device())
            .expect("Failed to find a suitable input device");

        let config = device.default_input_config().unwrap();
        let sample_rate = config.sample_rate().0 as f32;
        let sample_format = config.sample_format();
        // send the audio sample rate to the other thread so it can create a matching speech recognizer
        {
            let mut state = APP_STATE.lock();
            state.sample_rate = sample_rate;
            state.is_running = true;
        }
        // println!("set to Sample rate: {:?}", sample_rate);
        let tx_clone = transmit_audio_channel.clone();
        let mut reported = false;
        let stream = device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !reported {
                        reported = true;
                    }
                    // different audio devices use different ways of representing audio data
                    // a 32-bit float, a 16-bit signed integer, or a 16-bit unsigned integer
                    let data16 = match sample_format {
                        SampleFormat::F32 => convert_f32_to_16(data),
                        SampleFormat::I16 => convert_32_to_16(data),
                        SampleFormat::U16 => convert_32_to_16(data),
                        _ => panic!("Unsupported sample format"),
                    };
                    let audio_data = convert_stereo_to_mono(&data16);
                    tx_clone.send(audio_data).expect("Failed to send data");
                },
                move |err| {
                    eprintln!("Error during stream: {}", err);
                },
                None,
            )
           .unwrap();  
        // start the consumer thread
        // this thread takes in the audio from the producer thread and passes it through the 
        // speech recognizer, then sends back any recognized words
        thread::spawn(move || {
            let (grammar, path_to_model, sample_rate) = {
                let state = APP_STATE.lock();
                (state.grammar.clone(), state.path_to_model.clone(), state.sample_rate)
            };
            let model = new_vosk_model(&path_to_model);
            // create  the speech recognizer, optionally use a grammar to only look for specific words
            let recognizer;
            if grammar.len() == 0 {
                recognizer = new_vosk_recognizer(model, sample_rate);
            } else {
                recognizer = recognizer_new_grm(model, sample_rate, &grammar);
                println!("grammar loaded: {:?}", grammar);
            }
            // tell everyone else that the model is loaded and ready to roll
            {
                let mut state = APP_STATE.lock();
                state.model_is_loaded = true;
            }
            // some options for the model
            set_max_alternatives(recognizer, 1);
            set_words(recognizer, 1);
    
            while APP_STATE.lock().is_running {   
                std::thread::sleep(std::time::Duration::from_secs(5));
                for data in &receive_audio_channel {
                    // println!("Received audio data: {:?}", &data[..10]); 
                    // here is where we actually pass the audio waveform data to the voice recognizer
                    // we get the result back as a json string
                    recognizer_accept_waveform_s(recognizer, &data);
                    let result = recognizer_partial_result(recognizer);
                    // println!("Partial result: {:?}", result);
                    match serde_json::from_str::<VoskPartialResult>(&result) {
                        Ok(json) => {
                            // Successfully parsed JSON, now access `partial`
                            // println!("Partial result: {:?}", json.partial);
                            let listen_for_words = LISTEN_FOR_WORDS.lock().unwrap();
                            let is_active = listen_for_words.is_active;
                            if is_active {
                                let words: Vec<String> = json.partial.split_whitespace().map(String::from).collect();
                                transmit_words_channel.send(words.clone()).expect("Failed to send words");
                                // println!("Words: {:?}", words);
                                if words.len() > max_words {
                                    reset_recognizer(recognizer);
                                }
                            }
                        },
                        Err(e) => {
                            // Handle JSON parsing errors
                            println!("Error parsing JSON: {}", e);
                        }
                    }
                }
            }
            let final_result = recognizer_result(recognizer);
            println!("Final result: {:?}", final_result);
        });
        stream.play().unwrap();

        // this is the loop that looks for recognized words and calls the js callback if found
        while APP_STATE.lock().is_running {
            std::thread::sleep(std::time::Duration::from_millis(250));
            for wordlist in &receive_words_channel {
                // println!("received wordlist {:?}", wordlist);
                let cb_clone = Arc::clone(&all_words_callback_shared);
                let ch_clone = Arc::clone(&channel_shared);

                ch_clone.send(move |mut cx| {
                    let callback = cb_clone.to_inner(&mut cx);
                    let this = cx.undefined();
                    let mut args = vec![];
                    let js_array = JsArray::new(&mut cx, wordlist.len() as usize); 
                    for (i,word) in wordlist.iter().enumerate() {
                        let string = cx.string(word);
                        js_array.set(&mut cx, i as u32, string).unwrap(); 
                    }
                    args.push(js_array.upcast::<JsValue>());
                    match callback.call(&mut cx, this, args) {
                        Ok(_) => Ok(()),
                        Err(e) => {
                            // Handle the error, e.g., log it or send it back to JS
                            println!("Error calling JavaScript callback: {:?}", e);
                            Ok(())
                            // Optionally, you can transform this error into a JavaScript error
                            // cx.throw_error("Failed to call callback")
                        }
                    }    
                });
            }
        }
        stream.pause().unwrap();
    });


    Ok(cx.undefined())
}

// fn that converts f32 to i16:
fn convert_f32_to_16(input_data: &[f32]) -> Vec<i16> {
    input_data.iter().map(|v| (v * i16::MAX as f32) as i16).collect()
}

fn convert_32_to_16<T>(input_data: &[T]) -> Vec<i16>
where
    T: Sample + ToSample<i16>,
{
    input_data.iter().map(|v| v.to_sample()).collect()
}

fn  convert_stereo_to_mono(input_data: &[i16]) -> Vec<i16> {
    let mut result = Vec::with_capacity(input_data.len() / 2);
    result.extend(
        input_data
            .chunks_exact(2)
            .map(|chunk| chunk[0] / 2 + chunk[1] / 2),
    );

    result
}

// stop the listening thread
fn stop_listener(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let mut state = APP_STATE.lock();
    state.is_running = false;

    Ok(cx.undefined())
}

fn look_for_words(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let mut words_to_look_for  = LISTEN_FOR_WORDS.lock().unwrap();
    words_to_look_for.is_active = true;
    Ok(cx.undefined())
}

fn is_model_loaded(mut cx: FunctionContext) -> JsResult<JsBoolean> {
    let state = APP_STATE.lock();
    Ok(JsBoolean::new(&mut cx, state.model_is_loaded))
}

// all of the functions defined here are exposed to nodejs when you import this index.node module
// eg:
// const { setLogLevel, startListener, stopListener, listDevices, setMicName, setPathToModel, setSampleRate } = require('index.node');

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

/*
___    _____       _             __
/ __\   \_   \_ __ | |_ ___ _ __ / _| __ _  ___ ___
/ /       / /\| '_ \| __/ _ | '__| |_ / _` |/ __/ _ \
/ /___  /\/ /_ | | | | ||  __| |  |  _| (_| | (_|  __/
\____/  \____/ |_| |_|\__\___|_|  |_|  \__,_|\___\___|

This section is bindings to the vosk dynamic library (.so on linux or .dll on windows)
**************************************************************/

extern "C" {
    fn vosk_set_log_level(level: i32);
    fn vosk_model_new(model_path: *const c_char) -> *mut VoskModel;
    fn vosk_recognizer_new(model: *mut VoskModel, sample_rate: f32) -> *mut VoskRecognizer;
    fn vosk_recognizer_new_grm(
        model: *mut VoskModel,
        sample_rate: f32,
        grammar: *const c_char,
    ) -> *mut VoskRecognizer;

    fn vosk_recognizer_accept_waveform_s(
        recognizer: *mut VoskRecognizer,
        data: *const i16,
        length: i32,
    ) -> i32;
    fn vosk_recognizer_result(recognizer: *mut VoskRecognizer) -> *const c_char;
    fn vosk_recognizer_partial_result(recognizer: *mut VoskRecognizer) -> *const c_char;
    fn vosk_recognizer_set_max_alternatives(recognizer: *mut VoskRecognizer, max_alternatives: c_int) -> i32;
    fn vosk_recognizer_set_words(recognizer: *mut VoskRecognizer, words: c_int) -> i32;
    fn vosk_recognizer_reset(recognizer: *mut VoskRecognizer);
}

fn set_max_alternatives(recognizer: *mut VoskRecognizer, max_alternatives: c_int) -> i32 {
    unsafe { vosk_recognizer_set_max_alternatives(recognizer, max_alternatives) }
}

fn set_words(recognizer: *mut VoskRecognizer, words: c_int) -> i32 {
    unsafe { vosk_recognizer_set_words(recognizer, words) }
}

fn reset_recognizer(recognizer: *mut VoskRecognizer) {
    unsafe { vosk_recognizer_reset(recognizer) }
}

// 'safe' wrappers for the vosk library
fn set_log_level(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let level: i32 = cx.argument::<JsNumber>(0)?.value(&mut cx) as i32;
    unsafe {
        vosk_set_log_level(level);
    }
    Ok(cx.undefined())
}

fn new_vosk_model(model_path: &str) -> *mut VoskModel {
    let c_model_path = CString::new(model_path).expect("CString::new failed");
    unsafe { vosk_model_new(c_model_path.as_ptr()) }
}

fn new_vosk_recognizer(model: *mut VoskModel, sample_rate: f32) -> *mut VoskRecognizer {
    unsafe { vosk_recognizer_new(model, sample_rate) }
}

fn recognizer_new_grm(
    model: *mut VoskModel,
    sample_rate: f32,
    grammar: &str,
) -> *mut VoskRecognizer {
    let c_grammar = CString::new(grammar).expect("CString::new failed");
    unsafe { vosk_recognizer_new_grm(model, sample_rate, c_grammar.as_ptr()) }
}

fn recognizer_accept_waveform_s(recognizer: *mut VoskRecognizer, data: &[i16]) {
    unsafe {
        let _result = vosk_recognizer_accept_waveform_s(recognizer, data.as_ptr(), data.len() as i32);
        // println!("Accept waveform short result: {:?}", result)
    }
}


fn recognizer_result(recognizer: *mut VoskRecognizer) -> String {
    let c_result = unsafe { vosk_recognizer_result(recognizer) };
    let result = unsafe { CStr::from_ptr(c_result) };
    result.to_str().unwrap().to_string()
}

fn recognizer_partial_result(recognizer: *mut VoskRecognizer) -> String {
    let c_result = unsafe { vosk_recognizer_partial_result(recognizer) };
    let result = unsafe { CStr::from_ptr(c_result) };
    result.to_str().unwrap().to_string()
}
