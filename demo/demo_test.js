/**
 * Ariadne demo input: fictional, self-contained Bluetooth settings code.
 * All sessions and devices below are mock data; no hardware or network access.
 *
 * Run from the Abhilash Prism repository root:
 *   node demo/demo_test.js
 *
 * To search this file in the frontend, restart the API with:
 *   python -m ariadne.frontend_api.server --repo ./demo
 * Then open http://127.0.0.1:5173/?intro=off#demo and try:
 *   1. How is a user session verified before opening Bluetooth settings?
 *   2. Where is the settings://bluetooth/connections deeplink used?
 *   3. Which function scans nearby Bluetooth devices before pairing?
 *
 * Source targets: openBluetoothSettings, navigateToDeeplink, pairBluetoothDevice.
 * The frontend uses retrieval; rank order and scores are model-dependent.
 * For the separate AST engine, try:
 *   which files call tool authTool before bluetoothTool?
 * This should match both SettingsAgent methods below. The frontend API itself
 * does not perform that structural call-order analysis.
 */

'use strict';

async function verifySession(sessionId) {
  // Mock session authentication for the demonstration only.
  return sessionId === 'demo-session';
}

async function navigateToDeeplink(deeplink) {
  // Open the Bluetooth-settings deeplink without accessing real hardware.
  if (deeplink !== 'settings://bluetooth/connections') {
    throw new Error('Unsupported demo deeplink');
  }
  return { opened: true, deeplink };
}

async function scanDevices() {
  // Return fictional nearby Bluetooth accessories for pairing.
  return [{ id: 'headphones-01', name: 'Demo Headphones' }];
}

async function connectDevice(deviceId) {
  return { connected: true, deviceId };
}

const authTool = { verifySession };
const bluetoothTool = { navigateToDeeplink, scanDevices, connectDevice };

class SettingsAgent {
  async openBluetoothSettings(sessionId) {
    // Authenticate the user before opening privileged Bluetooth settings.
    const isAuthed = await authTool.verifySession(sessionId);
    if (!isAuthed) throw new Error('Unauthorized settings access');
    const targetDeeplink = 'settings://bluetooth/connections';
    return bluetoothTool.navigateToDeeplink(targetDeeplink);
  }

  async pairBluetoothDevice(sessionId, deviceId) {
    // Authenticate, scan nearby accessories, and pair the requested device.
    const isAuthed = await authTool.verifySession(sessionId);
    if (!isAuthed) throw new Error('Unauthorized pairing request');
    const devices = await bluetoothTool.scanDevices();
    const target = devices.find(device => device.id === deviceId);
    if (!target) throw new Error('Bluetooth device not found');
    return bluetoothTool.connectDevice(target.id);
  }
}

module.exports = { SettingsAgent, authTool, bluetoothTool };

if (require.main === module) {
  const assert = require('node:assert/strict');
  (async () => {
    const agent = new SettingsAgent();
    const opened = await agent.openBluetoothSettings('demo-session');
    assert.deepEqual(opened, { opened: true, deeplink: 'settings://bluetooth/connections' });
    const paired = await agent.pairBluetoothDevice('demo-session', 'headphones-01');
    assert.deepEqual(paired, { connected: true, deviceId: 'headphones-01' });
    await assert.rejects(agent.openBluetoothSettings('invalid-session'), /Unauthorized/);
    await assert.rejects(agent.pairBluetoothDevice('invalid-session', 'headphones-01'), /Unauthorized/);
    await assert.rejects(agent.pairBluetoothDevice('demo-session', 'missing-device'), /not found/);
    console.log('PASS: settings, pairing, invalid sessions, and missing-device checks.');
    console.log(JSON.stringify({ opened, paired }, null, 2));
  })().catch(error => { console.error(error); process.exitCode = 1; });
}
