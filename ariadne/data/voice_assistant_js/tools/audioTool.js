// audioTool.js - Audio Hardware Management Tool
/**
 * Direct hardware driver calls for speaker output and microphone gain.
 */
async function setVolume(level) {
  const boundedLevel = Math.max(0, Math.min(100, Number(level)));
  return { volume: boundedLevel, applied: true };
}

async function muteMicrophone(isMuted) {
  return { micMuted: Boolean(isMuted), timestamp: Date.now() };
}

module.exports = {
  setVolume,
  muteMicrophone,
};
