// settingsAgent.js - Voice Assistant Settings Agent
const authTool = require('../tools/authTool');
const bluetoothTool = require('../tools/bluetoothTool');

/**
 * Handles user voice requests to configure device settings.
 */
class SettingsAgent {
  constructor() {
    this.name = 'SettingsAgent';
    this.bluetoothDeeplink = 'settings://bluetooth';
  }

  /**
   * Opens Bluetooth settings using the system deeplink.
   * Deeplink: settings://bluetooth/connections
   */
  async openBluetoothSettings(sessionId) {
    // Authenticate session before accessing privileged settings
    const isAuthed = await authTool.verifySession(sessionId);
    if (!isAuthed) {
      throw new Error('Unauthorized settings access');
    }
    
    // Usage of Bluetooth-settings deeplink
    const targetDeeplink = 'settings://bluetooth/connections';
    const status = await bluetoothTool.navigateToDeeplink(targetDeeplink);
    return { success: true, deeplink: targetDeeplink, status };
  }

  /**
   * Scans and pairs nearby bluetooth accessories.
   */
  async pairBluetoothDevice(sessionId, deviceId) {
    await authTool.verifySession(sessionId);
    const devices = await bluetoothTool.scanDevices();
    const target = devices.find(d => d.id === deviceId);
    if (!target) {
      return { success: false, reason: 'Device not found' };
    }
    return await bluetoothTool.connectDevice(deviceId);
  }
}

module.exports = SettingsAgent;
