// js binding stuff:
use neon::prelude::*;
use neon::types::JsFunction;
use neon::prelude::Channel as NeonChannel;

// c++ binding stuff for the vosk library:
use std::ffi::{c_char, c_int, CStr, CString};
// thread control stuff:
use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::sync::mpsc::channel;
use std::sync::Mutex as StdMutex;
use std::sync::Arc;
use std::thread;

// audio input and processing stuff:
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait, },
    {SampleFormat}
};
use dasp::{sample::ToSample, Sample};

// some vosk stubs
enum VoskModel {}
enum VoskRecognizer {}

// controls the thread state:
struct AppState {
    is_running: bool,
    name_of_mic: String,
    path_to_model: String,
    sample_rate: f32,
}

impl AppState {
    fn new() -> Self {
        AppState {
            // used to start/stop the thread
            is_running: false,
            // the name of the microphone to listen to
            name_of_mic: String::from("default"),
            // the path to the file containing the vosk model
            path_to_model: String::from(""),
            sample_rate: 16000.0,
        }
    }
}

struct WordsToLookFor {
    // whether we're scanning for a word currently
    is_active: bool,
    // if false, matching any word in the words vec will trigger the callback
    match_all_words: bool,
    words: Vec<String>,
}

impl WordsToLookFor {
    fn new() -> Self {
        WordsToLookFor {
            is_active: false,
            match_all_words: false,
            words: Vec::new(),
        }
    }
}

// the one true app state
lazy_static! {
    static ref APP_STATE: Mutex<AppState> = Mutex::new(AppState::new());
    static ref WORDS_TO_LOOK_FOR: StdMutex<WordsToLookFor> = StdMutex::new(WordsToLookFor::new());
}

/*
     __          _       __  __    _____       _             __
  /\ \ \___   __| | ___  \ \/ _\   \_   \_ __ | |_ ___ _ __ / _| __ _  ___ ___
 /  \/ / _ \ / _` |/ _ \  \ \ \     / /\| '_ \| __/ _ | '__| |_ / _` |/ __/ _ \
/ /\  | (_) | (_| |  __/\_/ _\ \ /\/ /_ | | | | ||  __| |  |  _| (_| | (_|  __/
\_\ \/ \___/ \__,_|\___\___/\__/ \____/ |_| |_|\__\___|_|  |_|  \__,_|\___\___|

this section defines functions that you can access from nodejs
**************************************************************/

fn list_devices(mut cx: FunctionContext) -> JsResult<JsArray> {
    let host = cpal::default_host();
    let devices = host.input_devices().expect("Failed to get input devices");
    let js_array = JsArray::new(&mut cx, 0);
    for (device_index, device) in devices.enumerate() {
        let name = device.name().unwrap();
        let js_string = cx.string(name);
        js_array.set(&mut cx, device_index as u32, js_string)?;
        println!("Device {}: {}", device_index, device.name().unwrap());
    }
    Ok(js_array)
}

fn set_mic_name(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let name = cx.argument::<JsString>(0)?.value(&mut cx);

    {
        let mut state = APP_STATE.lock();
        state.name_of_mic = name;
    }

    Ok(cx.undefined())
}

fn set_path_to_model(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let name = cx.argument::<JsString>(0)?.value(&mut cx);
    {
        let mut state = APP_STATE.lock();
        state.path_to_model = name;
    }

    Ok(cx.undefined())
}

// start the listening thread
fn start_listener(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    // set up the audio
    let desired_input_device_name = {
        let state = APP_STATE.lock();
        state.name_of_mic.clone()
    };
    let channel = cx.channel();
    let (transmit_audio_channel, receive_audio_channel) = std::sync::mpsc::channel();
    let (transmit_words_channel, receive_words_channel) = std::sync::mpsc::channel();
    let on_words_found_callback = cx.argument::<JsFunction>(0)?.root(&mut cx);
    let callback_shared = Arc::new(on_words_found_callback);
    let channel_shared = Arc::new(channel);

    // start the producer thread, this thread opens and listens to your microphone
    thread::spawn(move || {
        // get the js stuff to handle invoking the callback defined in node:

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
        // the other thread will need to know the sample_rate when it creates the recognizer
        {
            let mut state = APP_STATE.lock();
            state.sample_rate = sample_rate;
            state.is_running = true;
        }
        println!("set to Sample rate: {:?}", sample_rate);
        let tx_clone = transmit_audio_channel.clone();
        let mut reported = false;
        let stream = device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !reported {
                        reported = true;
                        println!("Captured audio data: {:?}", &data[..10]); // Print first 10 samples
                    }
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
        // start the consumer thread, this runs the recognizer model
        thread::spawn(move || {
            let (path_to_model, sample_rate) = {
                let state = APP_STATE.lock();
                (state.path_to_model.clone(), state.sample_rate)
            };
            let model = new_vosk_model(&path_to_model);
            let recognizer = new_vosk_recognizer(model, sample_rate);
            set_max_alternatives(recognizer, 1);
            set_words(recognizer, 1);
    
            while APP_STATE.lock().is_running {
                std::thread::sleep(std::time::Duration::from_secs(5));
                {
                    let words_to_look_for = WORDS_TO_LOOK_FOR.lock().unwrap();
                    let words_copy = words_to_look_for.words.clone();
                    println!("binary is Looking for words: {:?}", words_copy);
                } // Lock guard is dropped here
                // println!("Looking for words: {:?}", words_to_look_for.words);
                for data in &receive_audio_channel {
                    // println!("Received audio data: {:?}", &data[..10]); // Print first 10 samples
                    recognizer_accept_waveform_s(recognizer, &data);
                    let result = recognizer_partial_result(recognizer);
                    let result_copy = result.clone();
                    // println!("Partial result: {:?}", result_copy);
                    let mut words_to_look_for = WORDS_TO_LOOK_FOR.lock().unwrap();
                    let is_active = words_to_look_for.is_active;
                    let match_all_words = words_to_look_for.match_all_words;
                    let words = words_to_look_for.words.clone();
                    let words_copy = words.clone();
                    // println!("Looking for words: {:?}", words_copy);
                    if is_active {
                        // if we're matching all words we have to match every word in the sentence
                        if match_all_words {
                            let mut found_all_words = true;
                            for word in words {
                                if !result.contains(&word) {
                                    found_all_words = false;
                                    break;
                                }
                            }
                            if found_all_words {
                                transmit_words_channel.send(words_to_look_for.words.clone()).expect("Failed to send words");
                                words_to_look_for.is_active = false;
                            }
                        } else {
                            for word in words {
                                if result.contains(&word) {
                                    transmit_words_channel.send(vec![word.clone()]).expect("Failed to send words");
                                    words_to_look_for.is_active = false;
                                }
                            }
                        }
                    } // lock guard is dropped here
                }
            }
            let final_result = recognizer_result(recognizer);
            println!("Final result: {:?}", final_result);
        });
        stream.play().unwrap();

        while APP_STATE.lock().is_running {
            std::thread::sleep(std::time::Duration::from_secs(1));
            for wordlist in &receive_words_channel {
                println!("Received words CHANNEL: {:?}", wordlist);
                let cb_clone = Arc::clone(&callback_shared);
                let ch_clone = Arc::clone(&channel_shared);
                print!("calling the clone");
                ch_clone.send(move |mut cx| {
                    let callback = cb_clone.to_inner(&mut cx);
                    let this = cx.undefined();
                    // turn the words array into a js array of js strings
                    let mut args = vec![];
                    for word in wordlist {
                        args.push(cx.string(word).upcast::<JsValue>());
                    }
                    println!("Calling the callback");
                    callback.call(&mut cx, this, args).unwrap();
                    println!("Callback called");
                    Ok(())
                });
            }
        }
        // any thread cleanup code goes here
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
    println!("Looking for words on the rust side called");
    let words = cx.argument::<JsArray>(0)?;
    println!("Words: {:?}", words);
    let match_all_words = cx.argument::<JsBoolean>(1)?.value(&mut cx);
    println!("Match all words: {:?}", match_all_words);
    let mut words_to_look_for  = WORDS_TO_LOOK_FOR.lock().unwrap();
    println!("Words to look for: {:?}", words_to_look_for.words);
    words_to_look_for.words.clear();
    for i in 0..words.len(&mut cx) {
        let word_handle: Handle<JsString> = words.get(&mut cx, i)?;
        words_to_look_for.words.push( word_handle.value(&mut cx));
    }
    words_to_look_for.match_all_words = match_all_words;
    words_to_look_for.is_active = true;
    println!("Looking for words: {:?}", words_to_look_for.words);
    Ok(cx.undefined())
}

// all of the functions defined here are exposed to nodejs when you import this index.node module
// eg:
// const { setLogLevel, startListener, stopListener, listDevices, setMicName, setPathToModel, setSampleRate } = require('index.node');

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("setLogLevel", set_log_level)?;
    // cx.export_function("modelNew", model_new)?;
    cx.export_function("startListener", start_listener)?;
    cx.export_function("stopListener", stop_listener)?;
    cx.export_function("listDevices", list_devices)?;
    cx.export_function("setMicName", set_mic_name)?;
    cx.export_function("setPathToModel", set_path_to_model)?;
    cx.export_function("lookForWords", look_for_words)?;
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
    fn vosk_model_find_word(model: *mut VoskModel, word: *const c_char) -> i32;
    fn vosk_recognizer_new(model: *mut VoskModel, sample_rate: f32) -> *mut VoskRecognizer;
    fn vosk_recognizer_new_grm(
        model: *mut VoskModel,
        sample_rate: f32,
        grammar: *const c_char,
    ) -> *mut VoskRecognizer;
    fn vosk_recognizer_set_grm(recognizer: *mut VoskRecognizer, grammar: *const c_char);

    fn vosk_recognizer_accept_waveform(
        recognizer: *mut VoskRecognizer,
        data: *const i32,
        length: i32,
    ) -> i32;
    fn vosk_recognizer_accept_waveform_f(
        recognizer: *mut VoskRecognizer,
        data: *const f32,
        length: i32,
    ) -> i32;
    fn vosk_recognizer_accept_waveform_s(
        recognizer: *mut VoskRecognizer,
        data: *const i16,
        length: i32,
    ) -> i32;
    fn vosk_recognizer_result(recognizer: *mut VoskRecognizer) -> *const c_char;
    fn vosk_recognizer_partial_result(recognizer: *mut VoskRecognizer) -> *const c_char;
    fn vosk_recognizer_set_max_alternatives(recognizer: *mut VoskRecognizer, max_alternatives: c_int) -> i32;
    fn vosk_recognizer_set_words(recognizer: *mut VoskRecognizer, words: c_int) -> i32;
    // fn vosk_recognizer_set_partial_words(recognizer: *mut VoskRecognizer, words: c_int) -> i32;
}

fn set_max_alternatives(recognizer: *mut VoskRecognizer, max_alternatives: c_int) -> i32 {
    unsafe { vosk_recognizer_set_max_alternatives(recognizer, max_alternatives) }
}

fn set_words(recognizer: *mut VoskRecognizer, words: c_int) -> i32 {
    unsafe { vosk_recognizer_set_words(recognizer, words) }
}

// fn set_partial_words(recognizer: *mut VoskRecognizer, words: c_int) -> i32 {
//     unsafe { vosk_recognizer_set_partial_words(recognizer, words) }
// }

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

fn set_vosk_recognizer_grammar(recognizer: *mut VoskRecognizer, grammar: &str) {
    let c_grammar = CString::new(grammar).expect("CString::new failed");
    unsafe { vosk_recognizer_set_grm(recognizer, c_grammar.as_ptr()) }
}

fn recognizer_accept_waveform(recognizer: *mut VoskRecognizer, data: &[i32]) {
    unsafe {
        let result = vosk_recognizer_accept_waveform(recognizer, data.as_ptr(), data.len() as i32);
        // println!("Accept waveform result: {:?}", result)
    }
}

fn recognizer_accept_waveform_f(recognizer: *mut VoskRecognizer, data: &[f32]) {
    unsafe {
        let result = vosk_recognizer_accept_waveform_f(recognizer, data.as_ptr(), data.len() as i32);
        // println!("Accept waveform result: {:?}", result)
    }
}

fn recognizer_accept_waveform_s(recognizer: *mut VoskRecognizer, data: &[i16]) {
    unsafe {
        let result = vosk_recognizer_accept_waveform_s(recognizer, data.as_ptr(), data.len() as i32);
        // println!("Accept waveform short result: {:?}", result)
    }
}


fn model_find_word(model: *mut VoskModel, word: &str) -> i32 {
    let c_word = CString::new(word).expect("CString::new failed");
    unsafe { vosk_model_find_word(model, c_word.as_ptr()) }
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
