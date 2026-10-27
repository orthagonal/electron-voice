# Regression tests

Run the ordinary suite from the repository root:

```sh
npm test
```

No microphone, recording, or speech model is required. Use Windows x64 with MSVC
or Linux x64 GNU with ALSA development libraries, Node.js 22+, and Rust 1.88+.
Bundled Vosk libraries must be available; the runner sets child-process DLL or
library search paths. It does not change system settings. Cargo uses the lockfile
and may download Rust dependencies on the first run.

The runner executes `cargo test --locked`, builds the current release addon,
copies it to the package entry point, and runs isolated JavaScript scenarios
with a 10-second subprocess timeout. It always tests a fresh native binary.

## Baseline and verification

Before hardening: **25 Rust tests and 7 JavaScript tests passed**.
The original test assertions have been retained; new tests are additive.

After hardening, local Windows x64 verification with Node.js 25.1.0 and Rust
1.99.0 produced:

```text
Rust:       52 passed; 0 failed; 0 ignored
JavaScript: 13 passed; 0 failed; 0 skipped
```

Additional checks:

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
npm run build
```

`cargo test --locked` is included in `npm test`. For a direct Windows invocation,
put the repository and `binaries/vosk-win64-0.3.45` on that process's PATH first.
The production Windows build command also handles an existing output directory.

The CI workflow runs formatting, Clippy, Rust tests, and native API tests on
Windows with Node 22/24 and Linux with Node 22. It also configures compile checks
for both macOS architectures. CI changes were not remotely executed in this
local session. Linux/macOS compilation and runtime behavior were not verified
on this Windows host; matching toolchains and native Vosk libraries are required.
The older macOS release workflow remains separate from these compile checks.

## Coverage

| Source | Checks |
| --- | --- |
| `src/tests.rs` | Original audio/recognition characterization tests; lifecycle reservation, duplicate starts, readiness, stop/restart, stale completion, teardown, configuration errors, delivery cleanup, and retained final-result policy |
| `src/audio.rs` | Actual typed F32/I16/U16 capture, stereo rounding, partial channel frames, multichannel mixing, callback boundaries, bounded FIFO buffering/recovery, idle receive timeout, cancellation, channel disconnect, and device-error propagation |
| `src/vosk.rs` | Unique native-handle cleanup on early return, null handles/results, UTF-8 handling, NUL paths, missing/incomplete model layouts, supported layout variants, and grammar validation |
| `tests/addon.test.cjs` | The original nine exports and seven API scenarios, invalid configuration, asynchronous errors and retries, 100 rapid cancellations, process exit, and independent Node worker environments |

The original vector-helper tests use adapters around the live sample arithmetic.
The capture tests additionally exercise the actual pooled callback path. Result
handling tests drive the same processor used by the listener and observe word
emissions and reset requests in order.

Model layout fixtures contain placeholder bytes and are used **only** with the
filesystem preflight. They are never passed into the native model constructor.
The initial empty-directory constructor probe terminated the test process; its
regression test now confirms the production wrapper rejects that input before
calling native code. Pointer/destructor tests use uniquely owned test allocations
with a matching destructor; they do not pretend to load a real Vosk model.

## Optional microphone smoke test

This is separate from `npm test`. With an existing compatible model and an
available microphone, run in PowerShell:

```powershell
$env:ELECTRON_VOICE_TEST_MODEL = 'C:\models\vosk-model-small-en-us-0.15'
npm run test:hardware
```

Optionally set `ELECTRON_VOICE_TEST_MIC` and `ELECTRON_VOICE_TEST_GRAMMAR`.
The test runs three actual start/ready/stop cycles, checks word-array shapes if
speech occurs, and checks that callbacks stop after cancellation. Without the
model environment variable, it is explicitly skipped. It was not run against
hardware during this hardening session.

## Manual Electron checks

1. Test the production model/grammar and actual shipped Vosk DLLs in the game.
   Verify command timing and repeated partials, including phrases containing
   `the` and utterances above the word limit.
2. Repeatedly start/stop during speech and during model loading. Immediately
   restart and confirm readiness belongs to the newest session. Watch microphone
   ownership and process memory over many cycles.
3. Test default, named, and unavailable devices; unplug the active microphone,
   observe `ELECTRON_VOICE_ERROR`, then reconnect and start again. Include available
   F32/I16/U16 and multichannel hardware, not just the unit-test vectors.
4. Pause or heavily load the game/event loop; verify bounded memory, useful
   warning reporting, and acceptable recovery when inference or JS falls behind.
5. Test package loading and shutdown in the actual Electron/Steam build. Node
   subprocess tests do not establish Electron ABI/deployment correctness.

Remaining limits: a native model load/inference call cannot be interrupted; a
native abort cannot be caught by Rust; filesystem preflight does not validate
model contents or prevent replacement races. The existing final `alternatives`
behavior is deliberately retained to avoid adding duplicate game commands.
