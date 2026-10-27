# Regression tests

Run the complete suite from the repository root on Windows x64:

```sh
npm test
```

You need Node.js (22 or newer), Rust with the
`x86_64-pc-windows-msvc` target, and the MSVC build tools. The bundled Vosk DLLs
must be present. No microphone, audio recording, speech model, or npm dependency
installation is required. Cargo uses `Cargo.lock` and may download Rust
dependencies on the first run.

The runner in `scripts/test.cjs`:

1. Adds the repository and bundled Windows Vosk directory to its child processes'
   DLL search path without changing your system settings.
2. Runs `cargo test --locked`.
3. Builds the current release addon and copies it to the package entry point.
4. Runs the JavaScript tests against that addon, with a fresh process and a
   10-second timeout for each scenario.

The Windows workflow in `.github/workflows/regression-tests.yml` runs the same
command for pushes and pull requests.

## Coverage

| Tests | Behavior covered |
| --- | --- |
| `src/tests.rs`: audio | Float and integer scaling, clipping, silence, stereo averaging, integer limits, rounding, empty buffers, and the existing odd-length-buffer fallback |
| `src/tests.rs`: recognition | Partial/final JSON parsing, Unicode and whitespace, extra metadata, invalid/empty results, repeated results, word-limit boundaries, emission before reset, and repeated `the` suppression/reset/recovery |
| `tests/addon.test.cjs` | Loading through `package.json`, all nine API exports, initial readiness, setter argument types and return values, invalid listener arguments, and repeated idle stop/activation calls |

Recognition tests supply JSON directly to the result handler used by the live
listener. They record emitted word arrays and reset requests in order. They do
not mock a second implementation of that handler.

These are characterization tests for the current working-tree behavior. For
example, they preserve the current audio rounding, repeated partial emissions,
strict `word_count > max_words` reset boundary, and partial counter behavior
across empty or invalid results. An intentional behavior change should update
the relevant expectation alongside a test reproducing the issue.

## Baseline verification

Local verification on Windows x64 with Node.js 25.1.0:

```text
Rust:       25 passed; 0 failed; 0 ignored
JavaScript:  7 passed; 0 failed; 0 skipped
```

The first 12 Rust helper tests and all seven API tests also passed before the
recognition handler was extracted. Two temporary mutations in an isolated copy
confirmed that the tests detect a reset threshold changed from 12 to 13 and a
word-limit comparison changed from `>` to `>=`.

## Remaining checks and existing issues

Real Vosk recognition and its actual result schema, audio-device selection and
stream formats, asynchronous delivery into a JavaScript callback, and active
listener stop/restart still need integration coverage. The current suite does
not start audio threads or load a speech model.

The existing `npm run build` Windows command can skip copying the addon when
`build/Release` already exists: its copy is inside a conditional command chain.
This was observed while adding the package-loading tests. The regression runner
builds and copies explicitly so it always tests a fresh addon. The production
build script has not been changed; it remains a separate fix.
