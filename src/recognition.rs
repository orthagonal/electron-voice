use super::{partial_is_only_the_tokens, VoskPartialResult, VoskTextResult};

// The live listener and deterministic tests share this exact result handler.
// Emit and reset remain separate callbacks so words are delivered before reset.
#[derive(Default)]
pub(super) struct RecognitionProcessor {
    all_the_partial_frames: usize,
    last_all_the_partial: String,
}

impl RecognitionProcessor {
    pub(super) fn process(
        &mut self,
        result: &str,
        utterance_complete: bool,
        max_words: usize,
        mut emit: impl FnMut(Vec<String>),
        mut reset: impl FnMut(),
    ) {
        const MAX_ALL_THE_PARTIAL_FRAMES: usize = 12;
        if utterance_complete {
            match serde_json::from_str::<VoskTextResult>(result) {
                Ok(json) => {
                    let words: Vec<String> =
                        json.text.split_whitespace().map(String::from).collect();
                    if words.is_empty() {
                        return;
                    }
                    self.all_the_partial_frames = 0;
                    self.last_all_the_partial.clear();
                    emit(words.clone());
                    if words.len() > max_words {
                        reset();
                    }
                }
                Err(e) => {
                    println!("Error parsing final JSON: {}", e);
                }
            }
            return;
        }
        // println!("Partial result: {:?}", result);
        match serde_json::from_str::<VoskPartialResult>(result) {
            Ok(json) => {
                let words: Vec<String> =
                    json.partial.split_whitespace().map(String::from).collect();
                if words.is_empty() {
                    return;
                }
                if partial_is_only_the_tokens(&words) {
                    let current_all_the_partial = words.join(" ");
                    self.all_the_partial_frames += 1;
                    if self.all_the_partial_frames == 1
                        || current_all_the_partial != self.last_all_the_partial
                        || self.all_the_partial_frames % 4 == 0
                    {
                        // eprintln!(
                        //     "[electron-voice] all-the partial repeat={} raw_partial={:?}",
                        //     self.all_the_partial_frames,
                        //     json.partial
                        // );
                    }
                    self.last_all_the_partial = current_all_the_partial;
                    if self.all_the_partial_frames >= MAX_ALL_THE_PARTIAL_FRAMES {
                        // eprintln!(
                        //     "[electron-voice] resetting recognizer after all-the dead-end repeat={} raw_partial={:?}",
                        //     self.all_the_partial_frames,
                        //     json.partial
                        // );
                        reset();
                        self.all_the_partial_frames = 0;
                        self.last_all_the_partial.clear();
                    }
                    return;
                } else {
                    if self.all_the_partial_frames > 0 {
                        eprintln!(
                            "[electron-voice] all-the dead-end cleared repeat={} next_partial={:?}",
                            self.all_the_partial_frames, json.partial
                        );
                    }
                    self.all_the_partial_frames = 0;
                    self.last_all_the_partial.clear();
                }
                emit(words.clone());
                // println!("Words: {:?}", words);
                if words.len() > max_words {
                    reset();
                    self.all_the_partial_frames = 0;
                    self.last_all_the_partial.clear();
                }
            }
            Err(e) => {
                // Handle JSON parsing errors
                println!("Error parsing JSON: {}", e);
            }
        }
    }
}
