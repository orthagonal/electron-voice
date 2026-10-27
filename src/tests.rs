use super::*;

#[test]
fn float_audio_preserves_sign_silence_and_current_scaling() {
    assert_eq!(
        convert_f32_to_16(&[-1.0, -0.5, 0.0, 0.5, 1.0]),
        [-32767, -16383, 0, 16383, 32767]
    );
}

#[test]
fn float_audio_saturates_out_of_range_samples() {
    assert_eq!(convert_f32_to_16(&[-2.0, 2.0]), [i16::MIN, i16::MAX]);
}

#[test]
fn signed_integer_audio_preserves_samples() {
    assert_eq!(
        convert_32_to_16(&[i16::MIN, -1, 0, 1, i16::MAX]),
        [i16::MIN, -1, 0, 1, i16::MAX]
    );
}

#[test]
fn unsigned_integer_audio_is_centered_at_zero() {
    assert_eq!(convert_32_to_16(&[0_u16, 32768, 65535]), [-32768, 0, 32767]);
}

#[test]
fn generic_float_conversion_uses_sample_trait_scaling() {
    // This helper has different scaling from convert_f32_to_16 today.
    assert_eq!(convert_32_to_16(&[-1.0_f32, 0.0, 0.5]), [-32768, 0, 16384]);
}

#[test]
fn stereo_audio_averages_each_pair_in_order() {
    assert_eq!(
        convert_stereo_to_mono(&[100, 300, -100, -300, -100, 100]),
        [200, -200, 0]
    );
}

#[test]
fn stereo_audio_avoids_overflow_at_integer_limits() {
    assert_eq!(
        convert_stereo_to_mono(&[i16::MIN, i16::MIN, i16::MAX, i16::MAX]),
        [i16::MIN, 32766]
    );
}

#[test]
fn stereo_audio_retains_current_rounding_toward_zero_per_channel() {
    assert_eq!(convert_stereo_to_mono(&[1, 1, -1, -1, 3, 3]), [0, 0, 2]);
}

#[test]
fn odd_length_stereo_audio_is_returned_unchanged() {
    assert_eq!(convert_stereo_to_mono(&[10, 20, 30]), [10, 20, 30]);
}

#[test]
fn empty_audio_buffers_remain_empty() {
    assert!(convert_f32_to_16(&[]).is_empty());
    assert!(convert_32_to_16::<i16>(&[]).is_empty());
    assert!(convert_stereo_to_mono(&[]).is_empty());
}

#[test]
fn only_nonempty_lowercase_the_sequences_are_filtered() {
    let words = |tokens: &[&str]| {
        tokens
            .iter()
            .map(|word| word.to_string())
            .collect::<Vec<_>>()
    };
    assert!(partial_is_only_the_tokens(&words(&["the"])));
    assert!(partial_is_only_the_tokens(&words(&["the", "the"])));
    for tokens in [&[][..], &["The"][..], &["the", "cat"][..], &["other"][..]] {
        assert!(!partial_is_only_the_tokens(&words(tokens)), "{tokens:?}");
    }
}

#[test]
fn initial_state_is_idle_with_default_audio_settings() {
    let state = AppState::new();
    assert!(!state.is_running);
    assert!(!state.model_is_loaded);
    assert_eq!(state.name_of_mic, "default");
    assert!(state.path_to_model.is_empty());
    assert!(state.grammar.is_empty());
    assert_eq!(state.sample_rate, 16000.0);
    assert!(WordsToLookFor::new().is_active);
}

#[derive(Debug, PartialEq)]
enum Event {
    Words(Vec<String>),
    Reset,
}

#[derive(Default)]
struct RecognitionHarness {
    processor: recognition::RecognitionProcessor,
}

impl RecognitionHarness {
    fn process(&mut self, json: &str, is_final: bool, max_words: usize) -> Vec<Event> {
        let events = std::cell::RefCell::new(Vec::new());
        self.processor.process(
            json,
            is_final,
            max_words,
            |words| events.borrow_mut().push(Event::Words(words)),
            || events.borrow_mut().push(Event::Reset),
        );
        events.into_inner()
    }

    fn partial(&mut self, text: &str, max_words: usize) -> Vec<Event> {
        self.process(
            &serde_json::json!({ "partial": text }).to_string(),
            false,
            max_words,
        )
    }

    fn final_result(&mut self, text: &str, max_words: usize) -> Vec<Event> {
        self.process(
            &serde_json::json!({ "text": text }).to_string(),
            true,
            max_words,
        )
    }
}

fn emitted(tokens: &[&str]) -> Event {
    Event::Words(tokens.iter().map(|token| token.to_string()).collect())
}

#[test]
fn partial_and_final_results_split_whitespace_and_preserve_unicode() {
    for is_final in [false, true] {
        let key = if is_final { "text" } else { "partial" };
        let json = serde_json::json!({ key: "  hello\tcafé\n世界\u{2003}again  " }).to_string();
        assert_eq!(
            RecognitionHarness::default().process(&json, is_final, 25),
            [emitted(&["hello", "café", "世界", "again"])]
        );
    }
}

#[test]
fn result_parsing_ignores_extra_vosk_metadata() {
    let mut harness = RecognitionHarness::default();
    assert_eq!(
        harness.process(r#"{"partial":"hello","partial_result":[]}"#, false, 25),
        [emitted(&["hello"])]
    );
    assert_eq!(
        harness.process(
            r#"{"text":"world","result":[{"word":"world","conf":0.9}]}"#,
            true,
            25
        ),
        [emitted(&["world"])]
    );
}

#[test]
fn repeated_partials_and_matching_final_are_each_emitted() {
    let mut harness = RecognitionHarness::default();
    assert_eq!(
        harness.partial("hello world", 25),
        [emitted(&["hello", "world"])]
    );
    assert_eq!(
        harness.partial("hello world", 25),
        [emitted(&["hello", "world"])]
    );
    assert_eq!(
        harness.final_result("hello world", 25),
        [emitted(&["hello", "world"])]
    );
}

#[test]
fn empty_results_do_not_emit_words_or_reset_the_recognizer() {
    let mut harness = RecognitionHarness::default();
    for text in ["", " ", "\n\t\u{2003}"] {
        assert!(harness.partial(text, 0).is_empty());
        assert!(harness.final_result(text, 0).is_empty());
    }
}

#[test]
fn malformed_or_missing_result_fields_do_not_emit_or_reset() {
    let mut harness = RecognitionHarness::default();
    for is_final in [false, true] {
        for json in ["", "{", "[]", "null", "{}", r#"{"partial":null,"text":42}"#] {
            assert!(harness.process(json, is_final, 0).is_empty(), "{json}");
        }
    }
    assert!(harness.process(r#"{"text":"hello"}"#, false, 0).is_empty());
    assert!(harness
        .process(r#"{"partial":"hello"}"#, true, 0)
        .is_empty());
    assert_eq!(
        harness.partial("still working", 25),
        [emitted(&["still", "working"])]
    );
}

#[test]
fn word_limit_resets_only_above_the_limit_and_after_emitting() {
    for is_final in [false, true] {
        let mut harness = RecognitionHarness::default();
        let key = if is_final { "text" } else { "partial" };
        for (text, expected) in [
            ("one", vec![emitted(&["one"])]),
            ("one two", vec![emitted(&["one", "two"])]),
            (
                "one two three",
                vec![emitted(&["one", "two", "three"]), Event::Reset],
            ),
        ] {
            let json = serde_json::json!({ key: text }).to_string();
            assert_eq!(
                harness.process(&json, is_final, 2),
                expected,
                "final={is_final}, text={text}"
            );
        }
    }
}

#[test]
fn zero_word_limit_emits_then_resets_for_any_real_word() {
    let mut harness = RecognitionHarness::default();
    assert_eq!(
        harness.partial("hello", 0),
        [emitted(&["hello"]), Event::Reset]
    );
    assert_eq!(
        harness.final_result("world", 0),
        [emitted(&["world"]), Event::Reset]
    );
}

#[test]
fn all_the_partials_reset_on_frame_twelve_and_start_counting_again() {
    let mut harness = RecognitionHarness::default();
    for frame in 1..=24 {
        let expected = if frame % 12 == 0 {
            vec![Event::Reset]
        } else {
            vec![]
        };
        assert_eq!(harness.partial("the the", 25), expected, "frame {frame}");
    }
}

#[test]
fn changing_all_the_partial_length_does_not_restart_the_counter() {
    let mut harness = RecognitionHarness::default();
    for frame in 1..=11 {
        let text = if frame % 2 == 0 { "the" } else { "the the the" };
        assert!(harness.partial(text, 0).is_empty());
    }
    assert_eq!(harness.partial("the the", 0), [Event::Reset]);
}

#[test]
fn normal_partial_clears_the_all_the_counter() {
    let mut harness = RecognitionHarness::default();
    for _ in 0..11 {
        assert!(harness.partial("the", 25).is_empty());
    }
    assert_eq!(harness.partial("the cat", 25), [emitted(&["the", "cat"])]);
    for _ in 0..11 {
        assert!(harness.partial("the", 25).is_empty());
    }
    assert_eq!(harness.partial("the", 25), [Event::Reset]);
}

#[test]
fn final_the_is_emitted_and_clears_the_partial_counter() {
    let mut harness = RecognitionHarness::default();
    for _ in 0..11 {
        assert!(harness.partial("the", 25).is_empty());
    }
    assert_eq!(
        harness.final_result("the the", 25),
        [emitted(&["the", "the"])]
    );
    for _ in 0..11 {
        assert!(harness.partial("the", 25).is_empty());
    }
    assert_eq!(harness.partial("the", 25), [Event::Reset]);
}

#[test]
fn empty_and_invalid_results_preserve_the_current_partial_counter() {
    // Characterize current behavior: only nonempty valid phrases clear this
    // counter. Empty results and parsing failures do not clear it today.
    let mut harness = RecognitionHarness::default();
    for _ in 0..11 {
        assert!(harness.partial("the", 25).is_empty());
    }
    assert!(harness.partial("", 25).is_empty());
    assert!(harness.final_result(" ", 25).is_empty());
    assert!(harness.process("{", false, 25).is_empty());
    assert!(harness.process("{}", true, 25).is_empty());
    assert_eq!(harness.partial("the", 25), [Event::Reset]);
}

#[test]
fn recognition_counters_are_independent_per_listener() {
    let mut first = RecognitionHarness::default();
    let mut second = RecognitionHarness::default();
    for _ in 0..11 {
        assert!(first.partial("the", 25).is_empty());
    }
    assert!(second.partial("the", 25).is_empty());
    assert_eq!(first.partial("the", 25), [Event::Reset]);
    assert!(second.partial("the", 25).is_empty());
}

#[test]
fn lifecycle_start_is_reserved_before_loading_and_duplicate_start_is_ignored() {
    let mut state = AppState::new();
    let first = Arc::new(Session::default());
    assert!(state.begin(Arc::clone(&first)));
    assert!(state.is_running);
    assert!(!state.model_is_loaded);
    assert!(!state.begin(Arc::new(Session::default())));
    assert!(state.owns(&first));
    assert!(state.ready(&first, 48000.0));
    assert!(state.model_is_loaded);
    assert_eq!(state.sample_rate, 48000.0);
}

#[test]
fn lifecycle_stop_cancels_loading_and_running_sessions_and_is_idempotent() {
    let mut state = AppState::new();
    for ready in [false, true] {
        let session = Arc::new(Session::default());
        assert!(state.begin(Arc::clone(&session)));
        if ready {
            assert!(state.ready(&session, 16000.0));
        }
        state.stop();
        state.stop();
        assert!(session.is_cancelled());
        assert!(!state.is_running);
        assert!(!state.model_is_loaded);
        assert!(!state.ready(&session, 16000.0));
    }
}

#[test]
fn stale_worker_completion_cannot_clear_a_replacement_listener() {
    let mut state = AppState::new();
    let old = Arc::new(Session::default());
    assert!(state.begin(Arc::clone(&old)));
    state.stop();
    let replacement = Arc::new(Session::default());
    assert!(state.begin(Arc::clone(&replacement)));
    assert!(!state.ready(&old, 44100.0));
    assert!(state.ready(&replacement, 48000.0));
    assert!(!state.complete(&old));
    assert!(state.is_running);
    assert!(state.model_is_loaded);
    assert_eq!(state.sample_rate, 48000.0);
    assert!(state.complete(&replacement));
    assert!(!state.is_running);
    assert!(!state.model_is_loaded);
}

#[test]
fn failed_initialization_can_be_followed_by_another_start() {
    let mut state = AppState::new();
    for _ in 0..100 {
        let session = Arc::new(Session::default());
        assert!(state.begin(Arc::clone(&session)));
        assert!(state.complete(&session));
        assert!(!state.is_running);
        assert!(!state.model_is_loaded);
    }
}

#[test]
fn environment_shutdown_cancels_session_and_disallows_restart() {
    let shared = Arc::new(Mutex::new(AppState::new()));
    let session = Arc::new(Session::default());
    assert!(shared.lock().begin(Arc::clone(&session)));
    drop(Environment(Arc::clone(&shared)));
    assert!(session.is_cancelled());
    assert!(!shared.lock().begin(Arc::new(Session::default())));
}

#[test]
fn invalid_configuration_is_rejected_before_native_initialization() {
    for model in ["", "invalid\0path"] {
        assert!(validate_start(model, "", 25.0).is_err());
    }
    for grammar in ["{", "{}", "null", "[42]", "[null]", "[\"hello\"]\0"] {
        assert!(validate_start("model", grammar, 25.0).is_err());
    }
    for limit in [
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        usize::MAX as f64,
    ] {
        assert!(validate_start("model", "", limit).is_err());
    }
    assert_eq!(validate_start("model", "", 0.0).unwrap(), 0);
    assert_eq!(
        validate_start("model", "[\"hello\",\"世界\"]", 25.9).unwrap(),
        25
    );
    assert_eq!(validate_start("model", "[]", 25.0).unwrap(), 25);
}

#[test]
fn delivery_slots_are_released_when_queued_work_is_dropped() {
    let session = Arc::new(Session::default());
    session.pending_callbacks.store(1, Ordering::Release);
    let permit = DeliveryPermit(Arc::clone(&session));
    session.cancel();
    drop(permit);
    assert_eq!(session.pending_callbacks.load(Ordering::Acquire), 0);
}

#[test]
fn completion_suppresses_words_but_stop_also_suppresses_failure_notices() {
    let mut state = AppState::new();
    let session = Arc::new(Session::default());
    assert!(state.begin(Arc::clone(&session)));
    assert!(state.complete(&session));
    assert!(session.is_done());
    assert!(!session.is_cancelled());
    state.stop();
    assert!(session.is_cancelled());
}

#[test]
fn result_errors_are_recoverable_and_do_not_enable_alternative_callbacks() {
    let mut processor = recognition::RecognitionProcessor::default();
    processor.process(
        r#"{"alternatives":[{"text":"hello"}]}"#,
        true,
        25,
        |_| panic!("Existing final alternatives must not add new game callbacks"),
        || panic!("Invalid result must not reset the recognizer"),
    );
    assert!(processor.take_error().is_some());
    assert!(processor.take_error().is_none());
    let mut words = Vec::new();
    processor.process(
        r#"{"partial":"hello"}"#,
        false,
        25,
        |result| words = result,
        || {},
    );
    assert_eq!(words, ["hello"]);
    assert!(processor.take_error().is_none());
}
