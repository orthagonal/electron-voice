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
