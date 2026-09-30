// router.js - Central Voice Assistant Intent Dispatcher
const SettingsAgent = require('./agents/settingsAgent');
const MediaAgent = require('./agents/mediaAgent');

const settingsAgent = new SettingsAgent();
const mediaAgent = new MediaAgent();

/**
 * Dispatches parsed NLU speech intents to corresponding domain agents.
 */
async function dispatchVoiceIntent(intent, params) {
  switch (intent) {
    case 'open_bluetooth_settings':
      return await settingsAgent.openBluetoothSettings(params.sessionId);
    case 'pair_bluetooth_device':
      return await settingsAgent.pairBluetoothDevice(params.sessionId, params.deviceId);
    case 'set_volume':
      return await mediaAgent.adjustVolume(params.sessionId, params.level);
    case 'mute_mic':
      return await mediaAgent.muteMicrophone(params.sessionId);
    default:
      throw new Error(`Unknown voice assistant intent: ${intent}`);
  }
}

module.exports = {
  dispatchVoiceIntent,
};
