//! Typed CPAL capture with a fixed buffer pool and a bounded mono PCM queue.

use cpal::traits::DeviceTrait;
use cpal::{Device, SampleFormat, SizedSample, Stream, StreamConfig, SupportedStreamConfig};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::Duration;

// Retain CPAL's callback boundaries for normal partial-result cadence. Very
// large callbacks are split at 100 ms to bound individual buffer sizes. Both
// queued samples (two seconds) and the number of packets are bounded.
const QUEUED_CHUNKS: usize = 512;

pub(super) trait PcmSample: SizedSample {
    fn pcm16(self) -> i16;
}

impl PcmSample for f32 {
    fn pcm16(self) -> i16 {
        // Retain the game's existing float scaling, truncation and saturation.
        (self * i16::MAX as f32) as i16
    }
}

impl PcmSample for i16 {
    fn pcm16(self) -> i16 {
        convert_sample(self)
    }
}

impl PcmSample for u16 {
    fn pcm16(self) -> i16 {
        convert_sample(self)
    }
}

pub(super) fn convert_sample<T: cpal::Sample>(sample: T) -> i16
where
    i16: cpal::FromSample<T>,
{
    sample.to_sample()
}

pub(super) fn stereo_component(sample: i16) -> i64 {
    i64::from(sample / 2)
}

struct Capture {
    channels: usize,
    channel_index: usize,
    sum: i64,
    chunk_samples: usize,
    current: Vec<i16>,
    audio: SyncSender<Vec<i16>>,
    free: Receiver<Vec<i16>>,
    recycle: SyncSender<Vec<i16>>,
    cancelled: Arc<AtomicBool>,
    dropped: Arc<AtomicUsize>,
    queued_samples: Arc<AtomicUsize>,
    max_queued_samples: usize,
}

pub(super) struct AudioInput {
    receiver: Receiver<Vec<i16>>,
    errors: Receiver<cpal::StreamError>,
    pub(super) dropped: Arc<AtomicUsize>,
    recycle: SyncSender<Vec<i16>>,
    queued_samples: Arc<AtomicUsize>,
}

impl AudioInput {
    pub(super) fn next(&self, cancelled: &AtomicBool) -> Result<Option<Vec<i16>>, String> {
        if cancelled.load(Ordering::Acquire) {
            return Ok(None);
        }
        if let Ok(error) = self.errors.try_recv() {
            return Err(format!("Microphone stream failed: {error}"));
        }
        match self.receiver.recv_timeout(Duration::from_millis(20)) {
            Ok(samples) => {
                if cancelled.load(Ordering::Acquire) {
                    self.recycle(samples);
                    Ok(None)
                } else {
                    Ok(Some(samples))
                }
            }
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => {
                Err("Microphone audio channel disconnected".into())
            }
        }
    }

    pub(super) fn recycle(&self, mut samples: Vec<i16>) {
        self.queued_samples
            .fetch_sub(samples.len(), Ordering::AcqRel);
        samples.clear();
        // If capture has been dropped there is nobody left to recycle to.
        let _ = self.recycle.try_send(samples);
    }
}

impl Capture {
    fn new(
        channels: usize,
        sample_rate: u32,
        cancelled: Arc<AtomicBool>,
    ) -> Result<(Self, AudioInput, SyncSender<cpal::StreamError>), String> {
        if channels == 0 || sample_rate == 0 {
            return Err("Audio channels and sample rate must be positive".into());
        }
        let chunk_samples = (sample_rate as usize / 10).max(1);
        let (audio, receiver) = mpsc::sync_channel(QUEUED_CHUNKS);
        let (recycle, free) = mpsc::sync_channel(QUEUED_CHUNKS + 1);
        let (errors_tx, errors) = mpsc::sync_channel(1);
        for _ in 0..=QUEUED_CHUNKS {
            recycle
                .try_send(Vec::with_capacity(chunk_samples))
                .map_err(|_| "Could not initialize the audio buffer pool")?;
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let queued_samples = Arc::new(AtomicUsize::new(0));
        let input = AudioInput {
            receiver,
            errors,
            dropped: Arc::clone(&dropped),
            recycle: recycle.clone(),
            queued_samples: Arc::clone(&queued_samples),
        };
        Ok((
            Self {
                channels,
                channel_index: 0,
                sum: 0,
                chunk_samples,
                current: Vec::with_capacity(chunk_samples),
                audio,
                free,
                recycle,
                cancelled,
                dropped,
                queued_samples,
                max_queued_samples: sample_rate as usize * 2,
            },
            input,
            errors_tx,
        ))
    }

    fn push<T: PcmSample>(&mut self, data: &[T]) {
        if self.cancelled.load(Ordering::Acquire) {
            return;
        }
        for &sample in data {
            let sample = sample.pcm16();
            // Preserve per-channel stereo rounding; use an overflow-safe mean
            // for more than two channels. Keep partial frames across callbacks.
            self.sum += if self.channels == 2 {
                stereo_component(sample)
            } else {
                i64::from(sample)
            };
            self.channel_index += 1;
            if self.channel_index != self.channels {
                continue;
            }
            let mono = if self.channels <= 2 {
                self.sum as i16
            } else {
                (self.sum / self.channels as i64) as i16
            };
            self.channel_index = 0;
            self.sum = 0;
            self.current.push(mono);
            if self.current.len() == self.chunk_samples {
                self.flush();
            }
        }
        // Do not delay short device callbacks to fill a larger packet, and do
        // not turn one normal callback into multiple repeated partial results.
        if !self.current.is_empty() {
            self.flush();
        }
    }

    fn flush(&mut self) {
        let queued = self.queued_samples.load(Ordering::Acquire);
        if self.current.len() > self.max_queued_samples.saturating_sub(queued) {
            self.dropped
                .fetch_add(self.current.len(), Ordering::Relaxed);
            self.current.clear();
            return;
        }
        let Ok(mut replacement) = self.free.try_recv() else {
            self.dropped
                .fetch_add(self.current.len(), Ordering::Relaxed);
            self.current.clear();
            return;
        };
        replacement.clear();
        let chunk = std::mem::replace(&mut self.current, replacement);
        self.queued_samples.fetch_add(chunk.len(), Ordering::AcqRel);
        // Never wait for inference on the device callback. Only drop new audio
        // after the entire two-second queue has filled, preserving FIFO order.
        match self.audio.try_send(chunk) {
            Ok(()) => {}
            Err(TrySendError::Full(mut chunk)) => {
                self.queued_samples.fetch_sub(chunk.len(), Ordering::AcqRel);
                self.dropped.fetch_add(chunk.len(), Ordering::Relaxed);
                chunk.clear();
                let _ = self.recycle.try_send(chunk);
            }
            Err(TrySendError::Disconnected(chunk)) => {
                self.queued_samples.fetch_sub(chunk.len(), Ordering::AcqRel);
                self.cancelled.store(true, Ordering::Release);
            }
        }
    }
}

pub(super) fn open(
    device: &Device,
    config: SupportedStreamConfig,
    cancelled: Arc<AtomicBool>,
) -> Result<(Stream, AudioInput), String> {
    let format = config.sample_format();
    let (capture, input, errors) = Capture::new(
        usize::from(config.channels()),
        config.sample_rate().0,
        cancelled,
    )?;
    let config: StreamConfig = config.into();
    let stream = match format {
        SampleFormat::F32 => build::<f32>(device, &config, capture, errors),
        SampleFormat::I16 => build::<i16>(device, &config, capture, errors),
        SampleFormat::U16 => build::<u16>(device, &config, capture, errors),
        other => return Err(format!("Unsupported microphone sample format: {other:?}")),
    }
    .map_err(|error| format!("Could not open microphone stream: {error}"))?;
    Ok((stream, input))
}

fn build<T: PcmSample>(
    device: &Device,
    config: &StreamConfig,
    mut capture: Capture,
    errors: SyncSender<cpal::StreamError>,
) -> Result<Stream, cpal::BuildStreamError> {
    device.build_input_stream(
        config,
        move |data: &[T], _| capture.push(data),
        move |error| {
            let _ = errors.try_send(error);
        },
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(channels: usize) -> (Capture, AudioInput) {
        // Small sample rate keeps fixtures and pool sizes small.
        let (capture, input, _) =
            Capture::new(channels, 100, Arc::new(AtomicBool::new(false))).unwrap();
        (capture, input)
    }

    fn drain(input: &AudioInput) -> Vec<i16> {
        let mut samples = Vec::new();
        while let Ok(chunk) = input.receiver.try_recv() {
            samples.extend_from_slice(&chunk);
            input.recycle(chunk);
        }
        samples
    }

    #[test]
    fn live_capture_supports_all_three_sample_formats() {
        let (mut capture, input) = capture(1);
        capture.push(&[-1.0_f32, -0.5, 0.0, 0.5, 1.0]);
        assert_eq!(drain(&input), [-32767, -16383, 0, 16383, 32767]);
        capture.push(&[i16::MIN, 0, i16::MAX]);
        assert_eq!(drain(&input), [i16::MIN, 0, i16::MAX]);
        capture.push(&[0_u16, 32768, 65535]);
        assert_eq!(drain(&input), [i16::MIN, 0, i16::MAX]);
    }

    #[test]
    fn live_stereo_matches_existing_rounding_and_preserves_partial_frames() {
        let (mut capture, input) = capture(2);
        capture.push(&[1_i16, 1, -1]);
        assert_eq!(drain(&input), [0]);
        capture.push(&[-1_i16, 100, 300]);
        assert_eq!(drain(&input), [0, 200]);
    }

    #[test]
    fn multichannel_frames_are_averaged_without_overflow() {
        let (mut capture, input) = capture(4);
        capture.push(&[i16::MAX; 4]);
        capture.push(&[i16::MIN; 4]);
        capture.push(&[100_i16, 200, 300, 400]);
        assert_eq!(drain(&input), [i16::MAX, i16::MIN, 250]);
    }

    #[test]
    fn buffering_is_bounded_fifo_and_recovers_after_saturation() {
        let (mut capture, input) = capture(1);
        capture.push(&[123_i16; 200]);
        assert_eq!(input.dropped.load(Ordering::Relaxed), 0);
        capture.push(&[999_i16; 5]);
        assert_eq!(input.dropped.load(Ordering::Relaxed), 5);
        assert_eq!(drain(&input), vec![123; 200]);
        capture.push(&[42_i16]);
        assert_eq!(drain(&input), [42]);
    }

    #[test]
    fn cancelled_capture_does_not_enqueue_audio() {
        let (mut capture, input) = capture(1);
        capture.cancelled.store(true, Ordering::Release);
        capture.push(&[1_i16, 2, 3]);
        assert!(drain(&input).is_empty());
    }

    #[test]
    fn invalid_audio_configuration_is_recoverable() {
        for (channels, rate) in [(0, 16000), (1, 0)] {
            assert!(Capture::new(channels, rate, Arc::new(AtomicBool::new(false))).is_err());
        }
    }

    #[test]
    fn normal_device_callbacks_keep_their_original_packet_boundaries() {
        let (mut capture, input) = capture(1);
        capture.push(&[1_i16, 2, 3]);
        capture.push(&[4_i16, 5]);
        let first = input.receiver.try_recv().unwrap();
        assert_eq!(first, [1, 2, 3]);
        input.recycle(first);
        let second = input.receiver.try_recv().unwrap();
        assert_eq!(second, [4, 5]);
        input.recycle(second);
        assert_eq!(input.queued_samples.load(Ordering::Acquire), 0);
    }

    #[test]
    fn idle_audio_wait_is_bounded_and_cancelled_wait_does_not_block() {
        let (_capture, input) = capture(1);
        let cancelled = AtomicBool::new(false);
        let started = std::time::Instant::now();
        assert!(input.next(&cancelled).unwrap().is_none());
        assert!(started.elapsed() < Duration::from_secs(1));
        cancelled.store(true, Ordering::Release);
        assert!(input.next(&cancelled).unwrap().is_none());
    }

    #[test]
    fn disconnected_audio_and_device_errors_are_recoverable() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let (capture, input, errors) = Capture::new(1, 100, Arc::clone(&cancelled)).unwrap();
        errors
            .try_send(cpal::StreamError::DeviceNotAvailable)
            .unwrap();
        assert!(input
            .next(&cancelled)
            .unwrap_err()
            .contains("Microphone stream failed"));
        drop(capture);
        assert!(input.next(&cancelled).unwrap_err().contains("disconnected"));
    }
}
