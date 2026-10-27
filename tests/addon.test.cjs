const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const test = require('node:test');

// Native global state is isolated per scenario. A crash or an event-loop leak
// fails the test, and no scenario starts a microphone or loads a speech model.
function inFreshProcess(body) {
  const result = spawnSync(process.execPath, ['-e', `
    const assert = require('node:assert/strict');
    const addon = require('./');
    ${body}
  `], {
    cwd: path.resolve(__dirname, '..'),
    encoding: 'utf8',
    timeout: 10000,
  });
  assert.ifError(result.error);
  assert.equal(result.status, 0, `Native test failed (${result.signal || result.status}):\n${result.stdout}\n${result.stderr}`);
}

test('the built package exposes the complete existing API', () => {
  inFreshProcess(`
    const names = [
      'setLogLevel', 'startListener', 'stopListener', 'listDevices',
      'setMicName', 'setPathToModel', 'lookForWords', 'isModelLoaded', 'setGrammar',
    ];
    assert.deepEqual(Object.keys(addon).sort(), names.sort());
    for (const name of names) assert.equal(typeof addon[name], 'function', name);
  `);
});

test('model readiness is false before starting a listener', () => {
  inFreshProcess(`assert.equal(addon.isModelLoaded(), false);`);
});

test('string setters accept empty and Unicode configuration without loading a model', () => {
  inFreshProcess(`
    for (const name of ['setMicName', 'setPathToModel', 'setGrammar']) {
      assert.equal(addon[name](''), undefined);
      assert.equal(addon[name]('configuration with spaces \\u00e9'), undefined);
    }
    assert.equal(addon.setGrammar('["hello", "world"]'), undefined);
    assert.equal(addon.isModelLoaded(), false);
  `);
});

test('string setters reject missing and non-string arguments', () => {
  inFreshProcess(`
    for (const name of ['setMicName', 'setPathToModel', 'setGrammar']) {
      assert.throws(() => addon[name](), TypeError, name);
      for (const value of [undefined, null, false, 1, {}, [], () => {}]) {
        assert.throws(() => addon[name](value), TypeError, name);
      }
    }
  `);
});

test('log level accepts numbers and rejects other argument types', () => {
  inFreshProcess(`
    assert.equal(addon.setLogLevel(-1), undefined);
    assert.equal(addon.setLogLevel(0), undefined);
    assert.throws(() => addon.setLogLevel(), TypeError);
    for (const value of [undefined, null, false, '0', {}, []]) {
      assert.throws(() => addon.setLogLevel(value), TypeError);
    }
  `);
});

test('startListener rejects invalid arguments before starting audio threads', () => {
  inFreshProcess(`
    assert.throws(() => addon.startListener(), TypeError);
    for (const value of [undefined, null, false, 1, 'callback', {}, []]) {
      assert.throws(() => addon.startListener(value, 25), TypeError);
    }
    assert.throws(() => addon.startListener(() => {}), TypeError);
    for (const value of [undefined, null, false, '25', {}, []]) {
      assert.throws(() => addon.startListener(() => {}, value), TypeError);
    }
    assert.equal(addon.isModelLoaded(), false);
  `);
});

test('stopListener and lookForWords can be called repeatedly while idle', () => {
  inFreshProcess(`
    for (let i = 0; i < 3; i++) {
      assert.equal(addon.stopListener(), undefined);
      assert.equal(addon.lookForWords(), undefined);
    }
    assert.equal(addon.isModelLoaded(), false);
  `);
});

test('startListener validates configuration and leaves readiness false on errors', () => {
  inFreshProcess(`
    assert.throws(() => addon.startListener(() => {}, 25), /model path/i);
    addon.setPathToModel('unused-model-path');
    for (const limit of [-1, NaN, Infinity, -Infinity]) {
      assert.throws(() => addon.startListener(() => {}, limit), /max_words/i);
    }
    for (const grammar of ['{', '{}', 'null', '[42]', '[null]']) {
      addon.setGrammar(grammar);
      assert.throws(() => addon.startListener(() => {}, 25), /grammar/i);
    }
    addon.setGrammar('');
    addon.setPathToModel('model\\0path');
    assert.throws(() => addon.startListener(() => {}, 25), /model path/i);
    assert.equal(addon.isModelLoaded(), false);
    assert.equal(addon.stopListener(), undefined);
  `);
});

test('log level rejects nonfinite and out-of-range numbers', () => {
  inFreshProcess(`
    for (const value of [NaN, Infinity, -Infinity, 2 ** 32]) {
      assert.throws(() => addon.setLogLevel(value), RangeError);
    }
  `);
});

test('missing models report asynchronous warnings and allow repeated retries', () => {
  inFreshProcess(`
    const path = require('node:path');
    addon.setPathToModel(path.join(process.cwd(), 'target', 'absent-model-for-regression-test'));
    let callbacks = 0;
    async function cycle() {
      const warning = new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Missing initialization warning')), 3000);
        process.once('warning', (warning) => {
          clearTimeout(timer);
          resolve(warning);
        });
      });
      assert.equal(addon.startListener(() => { callbacks++; }, 25), undefined);
      // Duplicate calls preserve the first listener and do not create a second.
      assert.equal(addon.startListener(() => { callbacks++; }, 25), undefined);
      const error = await warning;
      assert.equal(error.code, 'ELECTRON_VOICE_ERROR');
      assert.match(error.message, /model directory does not exist/i);
      assert.equal(addon.isModelLoaded(), false);
      assert.equal(addon.stopListener(), undefined);
    }
    (async () => {
      for (let i = 0; i < 3; i++) await cycle();
      assert.equal(callbacks, 0);
    })().catch((error) => { console.error(error); process.exitCode = 1; });
  `);
});

test('rapid start/stop cancels pending initialization without late callbacks or warnings', () => {
  inFreshProcess(`
    const path = require('node:path');
    addon.setPathToModel(path.join(process.cwd(), 'target', 'absent-model-for-cancellation-test'));
    let warnings = 0;
    let callbacks = 0;
    process.on('warning', () => { warnings++; });
    for (let i = 0; i < 100; i++) {
      addon.startListener(() => { callbacks++; }, 25);
      addon.stopListener();
      assert.equal(addon.isModelLoaded(), false);
    }
    setTimeout(() => {
      assert.equal(warnings, 0);
      assert.equal(callbacks, 0);
    }, 100);
  `);
});

test('a pending listener does not keep the JavaScript environment alive', () => {
  inFreshProcess(`
    addon.setPathToModel('absent-model-for-exit-test');
    addon.startListener(() => {}, 25);
  `);
});

test('Node worker environments have independent addon configuration', () => {
  inFreshProcess(`
    const { Worker } = require('node:worker_threads');
    addon.setPathToModel('main-environment-model');
    const worker = new Worker(\`
      const assert = require('node:assert/strict');
      const { workerData } = require('node:worker_threads');
      const addon = require(workerData);
      assert.equal(addon.isModelLoaded(), false);
      assert.throws(() => addon.startListener(() => {}, 25), /model path/i);
      addon.stopListener();
    \`, { eval: true, workerData: process.cwd() });
    worker.on('error', (error) => { console.error(error); process.exitCode = 1; });
    worker.on('exit', (code) => { assert.equal(code, 0); });
  `);
});
