// mediaAgent.js - Voice Assistant Media Playback Agent
const authTool = require('../tools/authTool');
const audioTool = require('../tools/audioTool');

/**
 * Handles media playback commands (play music, adjust volume).
 */
class MediaAgent {
  constructor() {
    this.name = 'MediaAgent';
  }

  /**
   * Adjusts device playback volume after verifying user session.
   */
  async adjustVolume(sessionId, level) {
    const isAuthed = await authTool.verifySession(sessionId);
    if (!isAuthed) {
      throw new Error('Unauthorized volume adjustment');
    }
    return await audioTool.setVolume(level);
  }

  /**
   * Mutes all microphone input for privacy mode.
   */
  async muteMicrophone(sessionId) {
    await authTool.verifySession(sessionId);
    return await audioTool.muteMicrophone(true);
  }
}

module.exports = MediaAgent;
