const { spawnSync } = require('node:child_process');
const { copyFileSync, mkdirSync } = require('node:fs');
const path = require('node:path');

// Resolve the bundled native library for both Rust tests and the Node addon.
// Loader paths change only in child processes, never in system settings.
const root = path.resolve(__dirname, '..');
const env = { ...process.env };
const pathKey = Object.keys(env).find((key) => key.toLowerCase() === 'path') || 'PATH';
env[pathKey] = [root, path.join(root, 'binaries', 'vosk-win64-0.3.45'), env[pathKey]]
  .filter(Boolean).join(path.delimiter);

const platforms = {
  win32: { target: 'x86_64-pc-windows-msvc', artifact: 'electron_voice.dll' },
  linux: { target: 'x86_64-unknown-linux-gnu', artifact: 'libelectron_voice.so' },
};
const platform = platforms[process.platform];
if (!platform || process.arch !== 'x64') {
  throw new Error('The addon regression runner supports Windows x64 and Linux x64. macOS can run cargo test with a matching Vosk library.');
}
if (process.platform === 'linux') {
  env.LD_LIBRARY_PATH = [path.join(root, 'binaries', 'linux'), env.LD_LIBRARY_PATH]
    .filter(Boolean).join(path.delimiter);
}

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, env, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}

run('cargo', ['test', '--locked']);
run('cargo', ['build', '--locked', '--release', '--target', platform.target]);
// Always test the actual package entry point using the current build.
const addonPath = path.join(root, require('../package.json').main);
mkdirSync(path.dirname(addonPath), { recursive: true });
copyFileSync(path.join(root, 'target', platform.target, 'release', platform.artifact), addonPath);
const testFile = process.argv.includes('--hardware') ? 'tests/hardware.test.cjs' : 'tests/addon.test.cjs';
run(process.execPath, ['--test', '--test-concurrency=1', testFile]);
