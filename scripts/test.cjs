const { spawnSync } = require('node:child_process');
const { copyFileSync, mkdirSync } = require('node:fs');
const path = require('node:path');

// Windows must find Vosk and its companion DLLs when loading either the Rust
// test executable or the Node addon. This changes only child-process settings.
const root = path.resolve(__dirname, '..');
const env = { ...process.env };
const pathKey = Object.keys(env).find((key) => key.toLowerCase() === 'path') || 'PATH';
env[pathKey] = [root, path.join(root, 'binaries', 'vosk-win64-0.3.45'), env[pathKey]]
  .filter(Boolean).join(path.delimiter);

if (process.platform !== 'win32' || process.arch !== 'x64') {
  throw new Error('The full addon regression suite currently requires Windows x64.');
}

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, env, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}

run('cargo', ['test', '--locked']);
run('cargo', ['build', '--locked', '--release', '--target', 'x86_64-pc-windows-msvc']);
// Always replace the addon: the existing npm build command can skip its copy
// when build/Release already exists. Test the actual package entry point using
// this build, rather than accidentally testing a stale binary.
const addonPath = path.join(root, require('../package.json').main);
mkdirSync(path.dirname(addonPath), { recursive: true });
copyFileSync(path.join(root, 'target', 'x86_64-pc-windows-msvc', 'release', 'electron_voice.dll'), addonPath);
run(process.execPath, ['--test', '--test-concurrency=1', 'tests/addon.test.cjs']);
