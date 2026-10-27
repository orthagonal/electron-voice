# electron-voice

Native Vosk speech recognition for an Electron main process or a Node.js worker.
CPAL captures microphone audio on native audio threads; an owned background Rust
worker runs Vosk in the **same process** as Electron. This is not a separate
process and does not bypass operating-system microphone permissions.

The JavaScript API delivers recognized words through Neon as `callback(words)`,
where `words` is an array of strings. General recognition, optional grammar,
partial results, and the existing word-limit reset policy are retained.

## Build and test

The npm package targets Windows x64. Development requires Node.js 22 or newer,
Rust 1.88 or newer, the `x86_64-pc-windows-msvc` target, and MSVC build tools.

```sh
npm install
npm run build
npm test
```

Installation builds the native addon. `npm run build` runs the locked Rust
release build and copies the DLL to `build/Release/electron-voice.node`, the
`package.json` entry point. Rebuilding also replaces an existing addon.

Ship the matching Vosk DLL and its companion DLLs where the Windows loader can
find them. The repository bundles these under `binaries/vosk-win64-0.3.45` and
also contains a root-level Vosk DLL. Those copies differ; do not silently swap
native libraries in an existing game deployment.

The configured Cargo targets also include Linux x64 GNU and macOS Intel/Apple
Silicon. They require matching native Vosk libraries and platform build tools.
The macOS configuration retains its custom cross-linker settings; native builds
can override the corresponding `CARGO_TARGET_*_LINKER` setting with `clang`.
`npm test` supports Windows/Linux x64; macOS can use `cargo test` with a matching
Vosk library. See [TESTING.md](TESTING.md) for checks and platform limitations.

## Electron usage

```js
const { app } = require('electron');
const voice = require('electron-voice');

// Native initialization/device errors arrive asynchronously without changing
// the existing callback(words) signature. Argument/configuration errors throw.
process.on('warning', (warning) => {
  if (warning.code === 'ELECTRON_VOICE_ERROR') {
    console.error(warning.message);
  }
});

const devices = voice.listDevices();
if (devices.length > 0) voice.setMicName(devices[0]);
voice.setPathToModel('path/to/vosk-model-small-en-us-0.15');
voice.setGrammar('["hello", "world"]'); // Optional; '' or '[]' selects general recognition.
voice.startListener((words) => {
  console.log(words); // ['hello', 'world']
}, 25);

// Poll voice.isModelLoaded() if the game needs a readiness indicator.
app.on('before-quit', () => voice.stopListener());
```

Use a compatible, intact [Vosk model](https://alphacephei.com/vosk/models).
Grammar requires a model supporting runtime graphs; Vosk may ignore grammar
for a model with a static graph. Grammar is a JSON array of nonblank phrases,
passed as a string. Setters retain their original signatures and store values;
configuration is validated when starting.

## API and lifecycle

The exports are unchanged: `setLogLevel`, `startListener`, `stopListener`,
`listDevices`, `setMicName`, `setPathToModel`, `lookForWords`, `isModelLoaded`,
and `setGrammar`. Void JavaScript functions continue returning `undefined`.

`startListener(callback, max_words)` schedules initialization without loading a
model or joining threads on the JavaScript thread. Starting while a listener is
starting or active is a no-op and retains the first callback. Each start snapshots
the microphone, model, and grammar configuration; later setter calls apply to
the next start. A missing named microphone still falls back to the default.

`isModelLoaded()` becomes true only after model/recognizer initialization and
stream startup succeed. `stopListener()` clears readiness and cancels pending
word delivery immediately; resource cleanup continues on the worker. A new
start waits there for the previous session's stream and native resources to be
released. Model loading and individual native inference calls cannot be
interrupted midway, so stopping during those calls delays physical cleanup
until they return. No additional words are flushed after a stop.

Each Node/Electron JavaScript environment owns its own state. Environment
teardown cancels its session. The Neon channel is unreferenced so an idle Node
process can exit; an Electron app naturally keeps its own event loop alive.

`max_words` resets the recognizer **after** emitting a result with more than the
configured number of words. The existing truncation of finite fractional values
is retained. Negative, nonfinite, or out-of-range limits are rejected. Partial
results remain cumulative and may repeat; no new deduplication is applied.
Partials containing only lowercase `the` tokens remain suppressed and trigger
a recognizer reset after 12 such frames. Nonempty normal or final phrases clear
that counter; empty and malformed results retain its existing behavior.

The retained `max_alternatives = 1` Vosk setting can produce final JSON under an
`alternatives` field. The existing result handler accepts top-level `partial`
and `text`; alternative-only finals still do not add callbacks. A schema error
now produces one warning per session while partial recognition continues.
Changing final-result delivery should be a separate, game-tested behavior change.

## Capture, errors, and native safety

F32, I16, and U16 streams use matching CPAL callback types. Mono passes through;
stereo retains the original per-channel rounding; multichannel input uses an
overflow-safe mean. Vosk receives the device's actual sample rate.

Capture uses preallocated mono buffers and nonblocking queue operations. Normal
CPAL callback boundaries are retained; callbacks larger than 100 ms are split
into bounded packets. Audio waiting for/undergoing inference is limited to two
seconds and 512 packets. Only after a bound is reached is new audio discarded,
with one warning per session. Up to 256 word deliveries can wait on JavaScript;
further work waits on the inference worker with cancellation checks. The device
callback never waits for inference or JavaScript.

Initialization, microphone disconnects, stream errors, malformed native results,
queue overflow, and callback exceptions are reported as Node warnings with code
`ELECTRON_VOICE_ERROR`. Initialization/stream failures clear readiness and allow
retry. Callback exceptions are caught and reported; they do not unwind Rust or
change callback arguments.

Vosk pointers are private, null-checked, and owned by RAII handles with matching
native destructors. A recognizer borrows its model and stays on its owning
thread. Result strings are copied while their recognizer is alive. FFI signatures
and model-layout checks were checked against the official
[Vosk C header](https://github.com/alphacep/vosk-api/blob/v0.3.45/src/vosk_api.h)
and [model loader](https://github.com/alphacep/vosk-api/blob/v0.3.45/src/model.cc).

The loaded Windows Vosk DLL terminated a test process for an empty model
directory. Filesystem preflight now rejects missing/incomplete supported layouts
before entering C++. This does **not** verify model contents or protect against
native aborts, corrupt files, ABI-incompatible libraries, or files changed during
loading. Deploy trusted, intact model files and keep them stable during use.
