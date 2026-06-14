export class EnergyVAD {
  static computeRMS(samples) {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / samples.length);
  }

  constructor(opts = {}) {
    this._threshold = opts.threshold ?? 0.01;
    this._minSpeech = opts.minSpeech ?? 0.25;
    this._minSilence = opts.minSilence ?? 0.4;
    this._hold = opts.hold ?? 0.15;
    this._sr = opts.sr ?? 16000;
    this._onSpeechStart = opts.onSpeechStart ?? null;
    this._onSpeechEnd = opts.onSpeechEnd ?? null;
    this._active = false;
    this._speechFrames = 0;
    this._silenceFrames = 0;
    this._holdFrames = 0;
    this._minSpeechFrames = Math.round(this._minSpeech * this._sr / 256);
    this._minSilenceFrames = Math.round(this._minSilence * this._sr / 256);
    this._holdFramesMax = Math.round(this._hold * this._sr / 256);
  }

  get active() { return this._active; }

  process(level, nSamples) {
    const frameRms = level;
    const isSpeech = frameRms > this._threshold;

    if (this._active) {
      if (isSpeech) {
        this._silenceFrames = 0;
        this._holdFrames = 0;
      } else {
        this._silenceFrames++;
        this._holdFrames++;
        if (this._holdFrames >= this._holdFramesMax || this._silenceFrames >= this._minSilenceFrames) {
          this._active = false;
          this._speechFrames = 0;
          this._silenceFrames = 0;
          this._holdFrames = 0;
          if (this._onSpeechEnd) this._onSpeechEnd();
        }
      }
    } else {
      if (isSpeech) {
        this._speechFrames++;
        if (this._speechFrames >= this._minSpeechFrames) {
          this._active = true;
          this._speechFrames = 0;
          this._holdFrames = 0;
          if (this._onSpeechStart) this._onSpeechStart();
        }
      } else {
        this._speechFrames = 0;
      }
    }

    return this._active;
  }
}
