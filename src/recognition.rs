//! Recognition policy shared by the live worker and characterization tests.

use super::{partial_is_only_the_tokens, VoskPartialResult, VoskTextResult};

#[derive(Default)]
pub(super) struct RecognitionProcessor {
    all_the_partial_frames: usize,
    error: Option<String>,
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
        self.error = None;
        // Keep the existing top-level text/partial schemas and emissions. In
        // particular, do not introduce new callbacks for n-best alternatives.
        let text = if utterance_complete {
            serde_json::from_str::<VoskTextResult>(result).map(|result| result.text)
        } else {
            serde_json::from_str::<VoskPartialResult>(result).map(|result| result.partial)
        };
        let text = match text {
            Ok(text) => text,
            Err(error) => {
                self.error = Some(format!("Could not parse Vosk recognition result: {error}"));
                return;
            }
        };
        let words: Vec<String> = text.split_whitespace().map(String::from).collect();
        if words.is_empty() {
            return;
        }
        if !utterance_complete && partial_is_only_the_tokens(&words) {
            self.all_the_partial_frames += 1;
            if self.all_the_partial_frames >= MAX_ALL_THE_PARTIAL_FRAMES {
                reset();
                self.all_the_partial_frames = 0;
            }
            return;
        }
        self.all_the_partial_frames = 0;
        let exceeds_limit = words.len() > max_words;
        emit(words);
        if exceeds_limit {
            reset();
        }
    }

    pub(super) fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }
}
