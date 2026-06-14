import * as ort from "onnxruntime-web/webgpu";
import { CONFIG, Profiles } from "./config.js";
import {
  buildMelFB,
  buildWindow,
  computeMelOffline,
  StreamingMel,
  detok,
} from "./dsp.js";
import { EnergyVAD } from "./vad.js";

const C = CONFIG;
const CACHE_NAME = "nemotron-asr-int4-v1";
const ORT_WASM_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";

// ────────────────────────────────────────────
//  Session — streaming transcription session
// ────────────────────────────────────────────

export class Session {
  /** @param {import("../types.d.ts").AsrEngine} engine */
  constructor(engine, langId, vadOptions = null) {
    this._engine = engine;
    this._langId = langId;
    this._state = null;
    this._mel = null;
    this._frames = [];
    this._frameOffset = 0;
    this._consumed = 0;
    this._diag = null;
    this._started = false;
    this._ended = false;
    this._vad = null;
    if (vadOptions) {
      this._vad = new EnergyVAD({
        ...(typeof vadOptions === "object" ? vadOptions : {}),
        onSpeechStart: () => this._engine._emit("speechStart"),
        onSpeechEnd: () => this._engine._emit("speechEnd"),
      });
    }
  }

  get speaking() {
    return this._vad ? this._vad.active : true;
  }

  /**
   * Push audio samples for streaming inference.
   * Returns partial result when enough frames accumulate, or null otherwise.
   * When VAD is enabled, non-speech audio is silently skipped.
   * @param {Float32Array} samples - 16 kHz PCM audio.
   * @returns {Promise<{text:string, lang:string|null}|null>}
   */
  async feed(samples) {
    if (this._ended) throw new Error("session has ended");
    if (this._vad) {
      const level = EnergyVAD.computeRMS(samples);
      this._vad.process(level, samples.length);
      if (!this._vad.active) return null;
    }
    const eng = this._engine;

    if (!this._started) {
      this._state = await eng._newState(this._langId);
      this._mel = new StreamingMel(eng._melFB, eng._window);
      this._diag = { start: performance.now(), encoder: 0, joint: 0, decoder: 0 };
      this._started = true;
    }

    const newFrames = this._mel.push(samples);
    for (const fr of newFrames) this._frames.push(fr);

    let partial = null;
    while (this._frames.length + this._frameOffset - this._consumed >= eng._newFrames) {
      await this._runBlock(eng._newFrames);
      const { text, lang } = detok(this._state.emitted, eng._vocab);
      partial = { text, lang };
    }
    return partial;
  }

  /**
   * Finalize streaming session and return complete result.
   * @returns {Promise<import("../types.d.ts").SessionResult|null>}
   */
  async end() {
    if (this._ended) return null;
    this._ended = true;
    if (!this._started) return { text: "", lang: null, tokens: 0, timing: null };
    const eng = this._engine;

    const remaining = this._frames.length + this._frameOffset - this._consumed;
    if (remaining > 0) {
      await this._runBlock(Math.min(eng._newFrames, remaining));
    }
    const { text, lang } = detok(this._state.emitted, eng._vocab);
    const total = performance.now() - this._diag.start;
    const timing = {
      encoder: this._diag.encoder,
      joint: this._diag.joint,
      decoder: this._diag.decoder,
      total,
    };
    return { text, lang, tokens: this._state.emitted.length, timing };
  }

  async _runBlock(validNew) {
    const eng = this._engine;
    const base = this._consumed;
    const buf = eng._getEncBuf();
    for (let i = 0; i < eng._encIn; i++) {
      const gi = base - eng._cacheFrames + i;
      const li = gi - this._frameOffset;
      const fr = gi >= 0 && li >= 0 && li < this._frames.length ? this._frames[li] : null;
      for (let m = 0; m < C.N_MELS; m++)
        buf[i * C.N_MELS + m] = fr ? fr[m] : 0;
    }
    await eng._encoderStep(this._state, eng._cacheFrames + validNew, this._diag);
    this._consumed += eng._newFrames;
    const keepFrom = this._consumed - eng._cacheFrames;
    if (keepFrom > this._frameOffset) {
      this._frames.splice(0, keepFrom - this._frameOffset);
      this._frameOffset = keepFrom;
    }
  }
}

// ────────────────────────────────────────────
//  AsrEngine — model lifecycle + inference
// ────────────────────────────────────────────

export class AsrEngine {
  /**
   * @param {import("../types.d.ts").AsrEngineCallbacks} [callbacks]
   * @param {{ profile?: string, beamWidth?: number, ensureCPU?: boolean, vad?: boolean | import("../types.d.ts").VadOptions }} [options]
   */
  constructor(callbacks = {}, options = {}) {
    this._callbacks = callbacks;

    if (typeof navigator !== "undefined") {
      ort.env.wasm.numThreads = Math.max(1, (navigator.hardwareConcurrency || 2) - 1);
    }
    if (!ort.env.wasm.wasmPaths) ort.env.wasm.wasmPaths = ORT_WASM_CDN;
    ort.env.logLevel = "error";

    this._applyProfile(options.profile || "NORMAL");
    this._beamWidth = Math.max(1, options.beamWidth || 1);
    this._ensureCPU = !!options.ensureCPU;
    this._vadOptions = options.vad || null;
    this._emit("status", `beam width: ${this._beamWidth}${this._beamWidth > 1 ? " (beam search)" : " (greedy)"}`);

    this._enc = null;
    this._dec = null;
    this._joint = null;
    this._vocab = null;
    this._melFB = null;
    this._window = null;
    this._ready = false;
    this._initInFlight = null;
    this._encEP = "webgpu";

    this._perfStats = {};
    this._jointEncBuf = null;
    this._jointDecBuf = null;
    this._jointEncTensor = null;
    this._jointDecTensor = null;
    this._encBuf = null;
    this._encTensor = null;
    this._encFrameBuf = new Float32Array(C.D_MODEL);
  }

  _applyProfile(name) {
    const p = Profiles[name];
    if (!p) throw new Error(`unknown profile: ${name}`);
    this._profile = name;
    this._encName = p.encoder;
    this._encDataName = p.encoderData;
    this._newFrames = 56;
    this._cacheFrames = 9;
    this._encIn = 65;
  }

  get ready() { return this._ready; }
  get profile() { return this._profile; }
  get encoderEP() { return this._encEP; }

  async _releaseSession(s) {
    if (s && typeof s.release === "function") {
      try { await s.release(); } catch {}
    }
  }

  async switchProfile(name) {
    this._applyProfile(name);
    // All profiles share the same encoder (560ms), no reload needed.
  }

  // ── lifecycle ──

  /**
   * Initialize the engine: download vocab, create ONNX sessions.
   * @returns {Promise<void>}
   */
  async init() {
    if (this._ready) return;
    if (this._initInFlight) return this._initInFlight;

    this._initInFlight = (async () => {
      this._emit("status", "fetching vocab");
      await this._releaseSession(this._dec);
      await this._releaseSession(this._joint);
      await this._releaseSession(this._enc);
      const vb = await this._fetchBytes("vocab.txt", "vocab.txt");
      this._vocab = new TextDecoder().decode(vb).split("\n").map((l) => l.replace(/ \d+$/, ""));
      this._melFB = buildMelFB();
      this._window = buildWindow();

      this._dec = await this._createSession(
        "decoder.onnx", "decoder.onnx.data", ["wasm"], "decoder (CPU)",
      );
      this._joint = await this._createSession(
        "joint.onnx", "joint.onnx.data", ["wasm"], "joint (CPU)",
      );

      const hasGPU = !!(typeof navigator !== "undefined" && navigator.gpu);
      const useGPU = hasGPU && !this._ensureCPU;
      const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(
        (typeof navigator !== "undefined" && navigator.userAgent) || "",
      );
      if (!hasGPU && isMobile && !this._ensureCPU) {
        throw new Error(
          "WebGPU is required on mobile for this 690 MB model, but navigator.gpu isn't available in this browser. Try the latest Chrome on Android, or Safari 18+ on iOS.",
        );
      }
      this._encEP = useGPU ? "webgpu" : "wasm";
      this._emit("ep", true, this._encEP);
      try {
        this._enc = await this._createSession(
          this._encName, this._encDataName, [{ name: this._encEP }],
          `encoder (~690 MB, ${this._encEP})`,
        );
      } catch (err) {
        if (this._encEP === "webgpu" && !isMobile) {
          this._encEP = "wasm";
          this._emit("ep", true, "wasm", String((err && err.message) || err));
          this._enc = await this._createSession(
            this._encName, this._encDataName, ["wasm"],
            "encoder (~690 MB, wasm)",
          );
        } else {
          throw err;
        }
      }

      this._ready = true;
      this._encTensorShape = [1, this._encIn, C.N_MELS];
      this._jointEncBuf = new Float32Array(C.D_MODEL);
      this._jointDecBuf = new Float32Array(C.DEC_HID);
      this._jointEncTensor = new ort.Tensor("float32", this._jointEncBuf, [1, 1, C.D_MODEL]);
      this._jointDecTensor = new ort.Tensor("float32", this._jointDecBuf, [1, 1, C.DEC_HID]);
    })();

    try {
      await this._initInFlight;
    } catch (err) {
      this._initInFlight = null;
      throw err;
    }
  }

  async clearCache() {
    try {
      await caches.delete(CACHE_NAME);
      this._emit("status", "cached model cleared");
    } catch (e) {
      this._emit("status", `clear cache failed: ${(e && e.name) || e}`);
    }
  }

  getPerfStats() {
    const out = {};
    for (const k of Object.keys(this._perfStats)) {
      const v = this._perfStats[k];
      out[k] = { ms: v.time, calls: v.count, avg: v.count ? v.time / v.count : 0 };
    }
    return out;
  }

  // ── full transcription ──

  /**
   * Transcribe full audio offline.
   * @param {Float32Array} samples - 16 kHz PCM audio.
   * @param {number} langId - Language ID (101 = auto-detect).
   * @returns {Promise<import("../types.d.ts").TranscriptionResult>}
   */
  async transcribe(samples, langId) {
    const diag = { start: performance.now(), encoder: 0, joint: 0, decoder: 0 };
    const mel = computeMelOffline(samples, this._melFB, this._window);
    const s = await this._newState(langId);
    const nf = this._newFrames, cf = this._cacheFrames;
    const steps = Math.ceil(mel.length / nf);
    for (let step = 0; step < steps; step++) {
      const base = step * nf;
      const buf = this._fillEncBuf(mel, base, cf);
      const validNew = Math.min(nf, mel.length - base);
      await this._encoderStep(s, cf + validNew, diag);
      const { text, lang } = detok(s.emitted, this._vocab);
      this._emit("partial", text, lang, (step + 1) / steps);
    }
    const { text, lang } = detok(s.emitted, this._vocab);
    const total = performance.now() - diag.start;
    return {
      text, lang, tokens: s.emitted.length,
      timing: { encoder: diag.encoder, joint: diag.joint, decoder: diag.decoder, total },
    };
  }

  // ── benchmark ──

  /**
   * Benchmark RTF across latency profiles, fastest first.
   * Stops after RTF > 1 unless forceAll is true.
   * @param {object} [opts]
   * @param {string[]} [opts.profiles] - Profiles to test (default: all 5, fastest first)
   * @param {number} [opts.duration]  - Synthetic audio duration in seconds (default: 10, ignored if samples given)
   * @param {number} [opts.langId]    - Language ID (default: 101)
   * @param {boolean} [opts.warmup]   - Run a brief warmup before measurement (default: true)
   * @param {boolean} [opts.forceAll] - Run all profiles even if RTF > 1 (default: false)
   * @param {Float32Array} [opts.samples] - Real audio samples; if set, duration is ignored
   * @returns {Promise<Array<{profile:string, latencyLabel:string, rtf:number, processingTimeMs:number, audioDurationSec:number, tokens:number, timing:object}>>}
   */
  async benchmark(opts = {}) {
    const FAST_FIRST = ["HIGH", "NORMAL", "BALANCED", "FAST", "TURBO"];
    const { profiles = FAST_FIRST, duration = 10, langId = 101, warmup = true, forceAll = false } = opts;
    if (!this._ready) await this.init();

    const sr = C.SR;
    const audio = opts.samples || (() => {
      const n = sr * duration;
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = (Math.random() - 0.5) * 0.02;
      return a;
    })();
    const audioSec = audio.length / sr;
    const results = [];

    for (const name of profiles) {
      const p = Profiles[name];
      if (!p) throw new Error(`unknown profile: ${name}`);

      await this.switchProfile(name);

      if (warmup) {
        const short = Math.min(sr, audio.length);
        await this.transcribe(audio.subarray(0, short), langId);
      }

      const t0 = performance.now();
      const r = await this.transcribe(audio, langId);
      const procMs = performance.now() - t0;
      const rtf = procMs / 1000 / audioSec;

      this._emit("status", `[bench] ${name}: ${rtf.toFixed(3)} RTF`);

      results.push({
        profile: name,
        rightContext: 6,
        latencyLabel: "560ms",
        processingTimeMs: procMs,
        audioDurationSec: audioSec,
        rtf,
        text: r.text,
        lang: r.lang,
        tokens: r.tokens,
        timing: r.timing,
      });

      if (rtf > 1 && !forceAll) {
        this._emit("status", `[bench] stopping — ${name} RTF ${rtf.toFixed(3)} > 1`);
        break;
      }
    }

    return results;
  }

  // ── session (streaming) ──

  /**
   * Create a streaming session for incremental transcription.
   * When VAD is configured on the engine, it is automatically applied to the session.
   * @param {number} langId
   * @param {boolean | import("../types.d.ts").VadOptions} [vadOverride] - Override engine-level VAD for this session.
   * @returns {Session}
   */
  session(langId, vadOverride) {
    const vad = vadOverride !== undefined ? vadOverride : this._vadOptions;
    return new Session(this, langId, vad);
  }

  // ── internal: helpers ──

  _emit(name, ...args) {
    const fn = this._callbacks[name];
    if (fn) fn(...args);
  }

  _perfEnd(name, t0) {
    const dt = performance.now() - t0;
    let s = this._perfStats[name];
    if (!s) { s = { time: 0, count: 0 }; this._perfStats[name] = s; }
    s.time += dt;
    s.count += 1;
  }

  async _openCache() {
    try { return await caches.open(CACHE_NAME); } catch { return null; }
  }

  async _fetchBytes(name, label) {
    const url = C.BASE + name;
    const cache = await this._openCache();
    if (cache) {
      const t0 = performance.now();
      const hit = await cache.match(url);
      if (hit) {
        this._emit("status", `[cache] ${label}: HIT (${(performance.now() - t0).toFixed(0)}ms)`);
        this._emit("progress", label, 1, 1, true);
        return new Uint8Array(await (await hit.blob()).arrayBuffer());
      }
      this._emit("status", `[cache] ${label}: MISS`);
    } else {
      this._emit("status", `[cache] caches API unavailable`);
    }

    const resp = await fetch(url);
    if (!resp.ok || !resp.body)
      throw new Error(`HTTP ${resp.status || "?"} fetching ${label}`);
    const total = Number(resp.headers.get("content-length")) || 0;

    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      this._emit("progress", label, loaded, total);
    }

    const len = chunks.reduce((s, c) => s + c.byteLength, 0);
    const buf = new Uint8Array(len);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.byteLength; }

    if (cache) {
      try {
        this._emit("status", `[cache] ${label}: storing ${(len / 1048576).toFixed(0)} MB`);
        await cache.put(url, new Response(buf, {
          headers: { "Content-Type": "application/octet-stream", "Content-Length": String(len) },
        }));
      } catch (e) {
        this._emit("status", `[cache] ${label}: store failed — ${(e && e.name) || e}`);
      }
    }
    return buf;
  }

  async _createSession(modelFile, dataFile, executionProviders, label) {
    this._emit("status", `loading ${label}`);
    const modelBytes = await this._fetchBytes(modelFile, modelFile);
    const opts = { executionProviders, graphOptimizationLevel: "all" };
    if (dataFile) {
      const dataBytes = await this._fetchBytes(dataFile, dataFile);
      opts.externalData = [{ path: dataFile, data: dataBytes }];
    }
    try {
      return await ort.InferenceSession.create(modelBytes, opts);
    } catch (err) {
      throw new Error(
        `failed to create ${label} session (${executionProviders.map((e) => e.name || e).join(",")}): ${(err && err.message) || err}`,
      );
    }
  }

  // ── internal: tensor helpers ──

  _f32(a, d) {
    return new ort.Tensor(
      "float32",
      a instanceof Float32Array ? a : Float32Array.from(a),
      d,
    );
  }

  _i64(a, d) {
    return new ort.Tensor(
      "int64",
      BigInt64Array.from(a.map((v) => BigInt(v))),
      d,
    );
  }

  _fillEncBuf(mel, base, cf) {
    const buf = this._getEncBuf();
    for (let i = 0; i < this._encIn; i++) {
      const gi = base - cf + i;
      const fr = gi >= 0 && gi < mel.length ? mel[gi] : null;
      for (let m = 0; m < C.N_MELS; m++)
        buf[i * C.N_MELS + m] = fr ? fr[m] : 0;
    }
    return buf;
  }

  _getEncBuf() {
    const sz = this._encIn * C.N_MELS;
    if (!this._encBuf || this._encBuf.length !== sz) {
      this._encBuf = new Float32Array(sz);
      this._encTensor = null;
    }
    return this._encBuf;
  }

  _getEncTensor() {
    if (!this._encTensor)
      this._encTensor = new ort.Tensor("float32", this._getEncBuf(), this._encTensorShape);
    return this._encTensor;
  }

  // ── internal: RNN-T decode ──

  async _newState(langId) {
    const s = {
      langId,
      cch: new Float32Array(C.LAYERS * 56 * C.D_MODEL),
      cct: new Float32Array(C.LAYERS * C.D_MODEL * 8),
      ccl: 0,
      h: new Float32Array(C.DEC_LAYERS * C.DEC_HID),
      c: new Float32Array(C.DEC_LAYERS * C.DEC_HID),
      decOut: null,
      emitted: [],
    };
    await this._decoderStep(s, C.BLANK);
    return s;
  }

  async _decoderStep(s, token, diag) {
    const __t = performance.now();
    const t0 = performance.now();
    const r = await this._dec.run({
      targets: this._i64([token], [1, 1]),
      h_in: this._f32(s.h, [C.DEC_LAYERS, 1, C.DEC_HID]),
      c_in: this._f32(s.c, [C.DEC_LAYERS, 1, C.DEC_HID]),
    });
    if (diag) diag.decoder += performance.now() - t0;
    this._perfEnd("decoderStep", __t);
    s.h = r.h_out.data;
    s.c = r.c_out.data;
    s.decOut = r.decoder_output.data;
  }

  async _jointLogits(s, encFrame) {
    if (!this._jointEncBuf) {
      this._jointEncBuf = new Float32Array(C.D_MODEL);
      this._jointEncTensor = new ort.Tensor("float32", this._jointEncBuf, [1, 1, C.D_MODEL]);
    }
    if (!this._jointDecBuf) {
      this._jointDecBuf = new Float32Array(C.DEC_HID);
      this._jointDecTensor = new ort.Tensor("float32", this._jointDecBuf, [1, 1, C.DEC_HID]);
    }
    this._jointEncBuf.set(encFrame);
    this._jointDecBuf.set(s.decOut);
    const r = await this._joint.run({
      encoder_output: this._jointEncTensor,
      decoder_output: this._jointDecTensor,
    });
    return r.joint_output.data;
  }

  async _jointArgmax(s, encFrame, diag) {
    const __t = performance.now();
    const t0 = performance.now();
    const logits = await this._jointLogits(s, encFrame);
    if (diag) diag.joint += performance.now() - t0;
    let best = 0;
    let bv = logits[0];
    for (let i = 1; i < C.VOCAB; i++) {
      if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    this._perfEnd("jointArgmax", __t);
    return best;
  }

  _topK(logits, k, blankId) {
    const arr = [];
    for (let i = 0; i < logits.length; i++) {
      if (i === blankId) continue;
      arr.push({ idx: i, v: logits[i] });
    }
    arr.sort((a, b) => b.v - a.v);
    return arr.slice(0, k).map((x) => ({ idx: x.idx, logProb: x.v }));
  }

  async _greedyDecode(enc, encT, s, fbuf, diag) {
    for (let t = 0; t < encT; t++) {
      const fr = enc.subarray(t * C.D_MODEL, (t + 1) * C.D_MODEL);
      fbuf.set(fr);
      let sym = 0;
      while (sym < C.MAX_SYM) {
        const k = await this._jointArgmax(s, fbuf, diag);
        if (k === C.BLANK) break;
        s.emitted.push(k);
        await this._decoderStep(s, k, diag);
        sym++;
      }
    }
  }


  _ensureBeamPool(maxN, size) {
    if (!this._beamPools) this._beamPools = {};
    const key = maxN;
    let p = this._beamPools[key];
    if (!p || p.pool.length < size) {
      const arrs = [];
      for (let i = 0; i < size; i++) {
        arrs.push({
          h: new Float32Array(C.DEC_LAYERS * C.DEC_HID),
          c: new Float32Array(C.DEC_LAYERS * C.DEC_HID),
          decOut: new Float32Array(C.DEC_HID),
          emitted: [],
          score: 0,
        });
      }
      p = { pool: arrs, idx: 0 };
      this._beamPools[key] = p;
    }
    return p;
  }

  async _beamDecode(enc, encT, s, fbuf, diag) {
    const bw = this._beamWidth;
    const tokT = performance.now();

    const perIter = bw * (bw + 2) + 2;
    const totalNeeded = perIter * Math.max(encT, 100);
    const pool = this._ensureBeamPool(bw, totalNeeded);
    pool.idx = 0;
    const pArr = pool.pool;

    const toBeam = (src) => {
      const nb = pArr[pool.idx++];
      nb.h.set(src.h);
      nb.c.set(src.c);
      if (src.decOut) nb.decOut.set(src.decOut);
      nb.emitted = [...(src.emitted || [])];
      nb.score = src.score || 0;
      return nb;
    };

    let beams = [toBeam(s)];
    const hLayer = C.DEC_HID;
    let totalDec = 0;

    // grow-once batch buffers
    if (!this._batchEnc) this._batchEnc = new Float32Array(0);
    if (!this._batchDec) this._batchDec = new Float32Array(0);
    if (!this._batchH)   this._batchH   = new Float32Array(0);
    if (!this._batchC)   this._batchC   = new Float32Array(0);
    if (!this._batchTgt) this._batchTgt = new BigInt64Array(0);

    for (let t = 0; t < encT; t++) {
      const fr = enc.subarray(t * C.D_MODEL, (t + 1) * C.D_MODEL);
      fbuf.set(fr);
      const B = beams.length;

      const encLen = B * C.D_MODEL;
      if (this._batchEnc.length < encLen) this._batchEnc = new Float32Array(encLen + 4096);
      const encBuf = this._batchEnc.subarray(0, encLen);
      for (let b = 0; b < B; b++) encBuf.set(fbuf, b * C.D_MODEL);

      const decLen = B * C.DEC_HID;
      if (this._batchDec.length < decLen) this._batchDec = new Float32Array(decLen + 4096);
      const decBuf = this._batchDec.subarray(0, decLen);
      for (let b = 0; b < B; b++) decBuf.set(beams[b].decOut, b * C.DEC_HID);

      const jr = await this._joint.run({
        encoder_output: new ort.Tensor("float32", encBuf, [B, 1, C.D_MODEL]),
        decoder_output: new ort.Tensor("float32", decBuf, [B, 1, C.DEC_HID]),
      });
      const allLogits = jr.joint_output.data; // [B, 1, 1, VOCAB] → flat B*VOCAB

      // ── 2. Build candidates ──
      const candidates = [];
      const pd = [];

      for (let b = 0; b < B; b++) {
        const off = b * C.VOCAB;
        // joint_output shape is [B,1,1,VOCAB] so stride is B*1*1*VOCAB = B*VOCAB
        const logits = allLogits.subarray(off, off + C.VOCAB);
        const top = this._topK(logits, bw, C.BLANK);

        for (const { idx, logProb } of top) {
          const nb = toBeam(beams[b]);
          nb.emitted.push(idx);
          nb.score += logProb;
          pd.push({ beam: nb, token: idx, hi: beams[b].h, ci: beams[b].c });
          candidates.push(nb);
        }

        const bb = toBeam(beams[b]);
        bb.score += logits[C.BLANK];
        candidates.push(bb);
      }

      // ── 3. Batched decoder ──
      if (pd.length > 0) {
        const decT0 = performance.now();
        const N = pd.length;
        if (this._batchTgt.length < N) this._batchTgt = new BigInt64Array(N + 16);
        if (this._batchH.length < C.DEC_LAYERS * N * hLayer) this._batchH = new Float32Array(C.DEC_LAYERS * (N + 16) * hLayer);
        if (this._batchC.length < C.DEC_LAYERS * N * hLayer) this._batchC = new Float32Array(C.DEC_LAYERS * (N + 16) * hLayer);
        const tgt = this._batchTgt.subarray(0, N);
        const hBuf = this._batchH.subarray(0, C.DEC_LAYERS * N * hLayer);
        const cBuf = this._batchC.subarray(0, C.DEC_LAYERS * N * hLayer);

        for (let i = 0; i < N; i++) {
          tgt[i] = BigInt(pd[i].token);
          for (let l = 0; l < C.DEC_LAYERS; l++) {
            const srcOff = l * hLayer;
            const dstOff = l * N * hLayer + i * hLayer;
            hBuf.set(pd[i].hi.subarray(srcOff, srcOff + hLayer), dstOff);
            cBuf.set(pd[i].ci.subarray(srcOff, srcOff + hLayer), dstOff);
          }
        }

        const dr = await this._dec.run({
          targets: new ort.Tensor("int64", tgt, [N, 1]),
          h_in: new ort.Tensor("float32", hBuf, [C.DEC_LAYERS, N, hLayer]),
          c_in: new ort.Tensor("float32", cBuf, [C.DEC_LAYERS, N, hLayer]),
        });

        const hOut = dr.h_out.data;
        const cOut = dr.c_out.data;
        const dOut = dr.decoder_output.data;

        for (let i = 0; i < N; i++) {
          for (let l = 0; l < C.DEC_LAYERS; l++) {
            const srcOff = l * N * hLayer + i * hLayer;
            pd[i].beam.h.set(hOut.subarray(srcOff, srcOff + hLayer), l * hLayer);
            pd[i].beam.c.set(cOut.subarray(srcOff, srcOff + hLayer), l * hLayer);
          }
          pd[i].beam.decOut.set(dOut.subarray(i * hLayer, (i + 1) * hLayer));
        }
        if (diag) totalDec += performance.now() - decT0;
      }

      // ── 4. Prune ──
      candidates.sort((a, b) => b.score - a.score);
      beams = candidates.slice(0, bw);
    }

    if (diag) {
      diag.joint += performance.now() - tokT - totalDec;
      diag.decoder += totalDec;
    }

    beams.sort((a, b) => b.score - a.score);
    const best = beams[0];
    s.h = best.h;
    s.c = best.c;
    s.decOut = best.decOut;
    s.emitted = best.emitted;
  }

  async _encoderStep(s, length, diag) {
    const __t = performance.now();
    const t0 = performance.now();
    const er = await this._enc.run({
      audio_signal: this._getEncTensor(),
      length: this._i64([length], [1]),
      cache_last_channel: this._f32(s.cch, [1, C.LAYERS, 56, C.D_MODEL]),
      cache_last_time: this._f32(s.cct, [1, C.LAYERS, C.D_MODEL, 8]),
      cache_last_channel_len: this._i64([s.ccl], [1]),
      lang_id: this._i64([s.langId], [1]),
    });
    if (diag) diag.encoder += performance.now() - t0;
    this._perfEnd("encoderStep", __t);
    const enc = er.outputs.data;
    const encT = er.outputs.dims[1];
    s.cch = er.cache_last_channel_next.data;
    s.cct = er.cache_last_time_next.data;
    s.ccl = Number(er.cache_last_channel_len_next.data[0]);
    const fbuf = this._encFrameBuf;
    if (this._beamWidth <= 1) {
      await this._greedyDecode(enc, encT, s, fbuf, diag);
    } else {
      await this._beamDecode(enc, encT, s, fbuf, diag);
    }
  }
}
