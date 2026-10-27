// Opt-in only: requires a real microphone and an existing compatible Vosk model.
// No audio recording is needed, and this file is excluded from ordinary tests.
const assert = require('node:assert/strict');
const test = require('node:test');

const model = process.env.ELECTRON_VOICE_TEST_MODEL;

test('real microphone supports repeated load/start/stop and releases each session', {
  skip: !model && 'Set ELECTRON_VOICE_TEST_MODEL to opt in',
  timeout: 180000,
}, async () => {
  const addon = require('../');
  let callbacks = 0;
  let failure;
  const onWarning = (warning) => {
    if (warning.code === 'ELECTRON_VOICE_ERROR') failure = warning;
  };
  process.on('warning', onWarning);
  addon.setPathToModel(model);
  if (process.env.ELECTRON_VOICE_TEST_MIC) addon.setMicName(process.env.ELECTRON_VOICE_TEST_MIC);
  addon.setGrammar(process.env.ELECTRON_VOICE_TEST_GRAMMAR || '');
  const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  try {
    for (let cycle = 0; cycle < 3; cycle++) {
      addon.startListener((words) => {
        assert.ok(Array.isArray(words));
        for (const word of words) assert.equal(typeof word, 'string');
        callbacks++;
      }, 25);
      const deadline = Date.now() + 50000;
      while (!addon.isModelLoaded()) {
        if (failure) throw failure;
        assert.ok(Date.now() < deadline, 'Model did not become ready');
        await delay(20);
      }
      addon.startListener(() => { failure = new Error('Duplicate listener replaced the callback'); }, 25);
      await delay(250);
      if (failure) throw failure;
      addon.stopListener();
      assert.equal(addon.isModelLoaded(), false);
      const stoppedCount = callbacks;
      await delay(150);
      assert.equal(callbacks, stoppedCount, 'Words arrived after stopListener');
    }
  } finally {
    addon.stopListener();
    process.off('warning', onWarning);
  }
});
