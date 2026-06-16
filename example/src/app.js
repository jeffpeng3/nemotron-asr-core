import { CONFIG } from "@jeffpeng3/nemotron-asr-core";

const SR = CONFIG.SR;
const $ = (id) => document.getElementById(id);

const els = {
  warn: $("webgpuWarning"),
  modeBtns: [...document.querySelectorAll(".mode-btn")],
  panels: {
    record: $("panel-record"),
    live: $("panel-live"),
    file: $("panel-file"),
    tab: $("panel-tab"),
  },
  lang: $("langId"),
  profile: $("profile"),
  langChip: $("langChip"),
  stripTag: $("stripTag"),
  vadToggle: $("vadToggle"),
  normalize: $("normalize"),
  beamWidth: $("beamWidth"),
  transcript: $("transcript"),
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  diag: $("diag"),
  diagToggle: $("diagToggle"),
  clearCacheBtn: $("clearCacheBtn"),
  benchBtn: $("benchBtn"),
  modelLoad: $("modelLoad"),
  loadBtn: $("loadBtn"),
  recBtn: $("recBtn"),
  recTimer: $("recTimer"),
  recLevel: $("recLevelFill"),
  liveBtn: $("liveBtn"),
  liveLevel: $("liveLevelFill"),
  liveDot: $("liveDot"),
  tabBtn: $("tabBtn"),
  tabLevel: $("tabLevelFill"),
  tabDot: $("tabDot"),
  file: $("fileInput"),
  fileBtn: $("fileBtn"),
};

const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});

function log(s, cls) {
  const e = document.createElement("span");
  if (cls) e.className = cls;
  e.textContent = s + "\n";
  els.diag.appendChild(e);
  els.diag.scrollTop = els.diag.scrollHeight;
}

function setStatus(text, state) {
  els.statusText.textContent = text;
  els.statusDot.className = "status-dot" + (state ? " " + state : "");
}

let lastText = "";
let lastLang = null;

function formatLine(text, lang) {
  if (!text) return "";
  return els.stripTag.checked || !lang ? text : `${text} <${lang}>`;
}

function setLangChip(lang) {
  const show = lang && !els.stripTag.checked;
  els.langChip.textContent = show ? lang : "";
  els.langChip.classList.toggle("hidden", !show);
}

function setTranscript(text, lang) {
  lastText = text;
  lastLang = lang;
  const show = formatLine(text, lang);
  els.transcript.textContent = show || "—";
  els.transcript.classList.toggle("is-empty", !show);
  setLangChip(lang);
}

let readyState = "idle";
let readyPromise = null;
let readyResolve = null;
let readyReject = null;
// diagnostics timing
let fullDiagStart = null;
let streamDiagStart = null;

function ensureReady() {
  if (readyState === "ready") return Promise.resolve();
  if (readyState === "loading") return readyPromise;
  readyState = "loading";
  readyPromise = new Promise((res, rej) => {
    readyResolve = res;
    readyReject = rej;
  });
  setStatus("loading model…", "loading");
  worker.postMessage({ type: "init", profile: els.profile.value, beamWidth: parseInt(els.beamWidth.value) });
  return readyPromise;
}

worker.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case "progress": {
      const pct = m.cached
        ? "cached"
        : m.total
          ? ((m.loaded / m.total) * 100).toFixed(0) + "%"
          : (m.loaded / 1048576).toFixed(0) + " MB";
      setStatus(`loading ${m.label} · ${pct}`, "loading");
      break;
    }
    case "status":
      log(`· ${m.detail}`, "dim");
      break;
    case "ep":
      log(`execution provider: ${m.ep}`, m.ep === "webgpu" ? "ok" : "err");
      if (m.note) log("webgpu fallback: " + m.note, "dim");
      break;
    case "ready":
      readyState = "ready";
      setStatus("model ready", "online");
      {
        const label = { TURBO: "80ms", FAST: "160ms", BALANCED: "320ms", NORMAL: "560ms", HIGH: "1120ms" }[m.profile] || m.profile;
        log(`models ready (encoder on ${m.encoderEP || "webgpu"}, ${label})`, "ok");
      }
      reflectModelState();
      readyResolve && readyResolve();
      break;
    case "stream-ready":
      break;
    case "stream-tick":
      pendingBlocks = Math.max(0, pendingBlocks - 1);
      tabPending = Math.max(0, tabPending - 1);
      break;
    case "partial":
      if (liveSegmenting) {
        renderLive(m.text, m.lang);
        break;
      }
      if (tabSegmenting) break;
      setTranscript(m.text, m.lang);
      if (m.progress != null)
        setStatus(
          `transcribing · ${(m.progress * 100).toFixed(0)}%`,
          "loading",
        );
      break;
    case "final":
      if (liveSegmenting) {
        commitLiveSegment(m.text, m.lang);
        if (streamDiagStart) {
          const elapsed = ((performance.now() - streamDiagStart) / 1000).toFixed(2);
          log(`segment time: ${elapsed} s`, "dim");
          streamDiagStart = null;
        }
        break;
      }
      if (tabSegmenting) break;
      setTranscript(m.text, m.lang);
      setStatus("model ready", "online");
      log(`done — ${m.tokens} tokens`, "ok");
      if (m.timing) {
        const t = m.timing;
        log(`timings — encoder: ${(t.encoder/1000).toFixed(2)} s, joint: ${(t.joint/1000).toFixed(2)} s, decoder: ${(t.decoder/1000).toFixed(2)} s, total: ${(t.total/1000).toFixed(2)} s`, "dim");
      }
      if (fullDiagStart) {
        const elapsed = ((performance.now() - fullDiagStart) / 1000).toFixed(2);
        log(`total time: ${elapsed} s`, "dim");
        fullDiagStart = null;
      }
      finishBusy();
      break;
    case "perf":
      try {
        const header = m.profile ? `${m.profile} perf stats:` : "perf stats:";
        log(header, "dim");
        const s = m.stats || {};
        for (const k of Object.keys(s)) {
          const v = s[k];
          log(`· ${k}: ${v.ms.toFixed(2)} ms total · ${v.calls} calls · avg ${v.avg.toFixed(2)} ms`, "dim");
        }
      } catch (e) {
        log("perf parse error", "err");
      }
      break;
    case "error":
      setStatus("error", "error");
      log("ERROR: " + m.message, "err");
      if (readyState === "loading") {
        readyState = "idle";
        readyReject && readyReject(new Error(m.message));
      }
      finishBusy();
      stopLiveUI();
      stopTabCapture();
      finishRecordUI();
      reflectModelState();
      break;
    case "benchmark": {
      const results = m.results || [];
      const totalMs = m.totalMs || 0;
      setStatus("model ready", "online");
      log("benchmark complete:", "ok");
      const durStr = results.length ? results[0].audioDurationSec.toFixed(1) + " s audio" : "";
      const sep = "─".repeat(68);
      const lines = [];
      lines.push(`Benchmark RTF — ${durStr}`);
      lines.push(sep);
      lines.push(`Profile     Beam  RTF      Time     Text`);
      lines.push(sep);
      for (const r of results) {
        const text = (r.text || "").slice(0, 38).replace(/\n/g, " ");
        const bwLabel = r.beamWidth === 1 ? "greedy" : `beam=${r.beamWidth}`;
        lines.push(
          `${r.profile.padEnd(12)} ${bwLabel.padEnd(6)} ${r.rtf.toFixed(3).padStart(6)}  ${r.processingTimeMs.toFixed(0).padStart(6)}ms  ${text}`,
        );
      }
      lines.push(sep);
      const best = results.reduce((a, b) => (a.rtf < b.rtf ? a : b));
      const bestBw = best.beamWidth === 1 ? "greedy" : `beam=${best.beamWidth}`;
      lines.push(`Fastest: ${best.profile} (${bestBw})  RTF ${best.rtf.toFixed(3)}  │  total: ${(totalMs / 1000).toFixed(1)}s`);
      setTranscript(lines.join("\n"), null);
      for (const r of results) {
        const bwLabel = r.beamWidth === 1 ? "greedy" : `beam=${r.beamWidth}`;
        log(
          `${r.profile} (${r.latencyLabel}) ${bwLabel} · RTF ${r.rtf.toFixed(3)} · ${r.processingTimeMs.toFixed(0)} ms for ${r.audioDurationSec.toFixed(1)} s · "${(r.text || "").slice(0, 60)}"`,
          "dim",
        );
      }
      finishBusy();
      reflectModelState();
      break;
    }
  }
};

const langId = () => parseInt(els.lang.value || "0", 10);

let busy = false;

function startBusy() {
  busy = true;
  liveSegmenting = false;
}

function finishBusy() {
  busy = false;
}

const HAS_GPU = !!navigator.gpu;
const IS_MOBILE = /Mobi|Android|iPhone|iPad|iPod/i.test(
  navigator.userAgent || "",
);

if (!HAS_GPU) {
  els.warn.classList.remove("hidden");
  els.warn.textContent = IS_MOBILE
    ? "WebGPU isn't available in this browser. This 690 MB model needs WebGPU — use the latest Chrome (Android) or Safari 18+ (iOS)."
    : "WebGPU isn't available. Use Chrome or Edge (or enable the WebGPU flag). A CPU fallback will work but is very slow.";
  if (IS_MOBILE) setStatus("WebGPU unavailable", "error");
}

function reflectModelState() {
  const ready = readyState === "ready";
  const loading = readyState === "loading";
  const blocked = IS_MOBILE && !HAS_GPU;
  els.loadBtn.disabled = ready || loading || blocked;
  els.loadBtn.textContent = loading ? "Loading model…" : "⬇ Load model (~750 MB)";
  els.modelLoad.classList.toggle("hidden", ready);
  els.modeBtns.forEach((b) => (b.disabled = !ready));
  els.recBtn.disabled = !ready;
  els.liveBtn.disabled = !ready;
  els.fileBtn.disabled = !ready || !els.file.files.length;
  els.benchBtn.disabled = loading || busy;
}

els.loadBtn.addEventListener("click", () => {
  if (readyState !== "idle") return;
  ensureReady().catch((err) =>
    log("model load failed: " + (err.message || err), "err"),
  );
  reflectModelState();
});

reflectModelState();

function setMode(mode) {
  els.modeBtns.forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === mode),
  );
  for (const k in els.panels)
    els.panels[k].classList.toggle("hidden", k !== mode);
}

els.modeBtns.forEach((b) =>
  b.addEventListener("click", () => setMode(b.dataset.mode)),
);

setMode("file");

els.diagToggle.addEventListener("click", () =>
  els.diag.classList.toggle("hidden"),
);

els.clearCacheBtn.addEventListener("click", () => {
  worker.postMessage({ type: "clearCache" });
  log("clearing cached model… reload the page to re-download", "dim");
});

els.benchBtn.addEventListener("click", async () => {
  if (busy) return;
  startBusy();
  setStatus("benchmarking…", "loading");
  const file = els.file.files[0];
  let samplesBuf = null;
  let label = "10 s synthetic noise";
  if (file) {
    try {
      setStatus("decoding audio…", "loading");
      const samples = await decodeAudioFile(file);
      normalizeAudio(samples);
      samplesBuf = samples.buffer;
      label = `${file.name} (${(samples.length / SR).toFixed(2)} s)`;
    } catch (err) {
      log("benchmark audio decode failed: " + (err.message || err), "err");
      finishBusy();
      reflectModelState();
      return;
    }
  }
  log(`benchmark started — running all 5 profiles on ${label}`, "dim");
  try {
    const msg = { type: "benchmark", duration: 10, beamWidths: [1, 2, 3] };
    if (samplesBuf) msg.samples = samplesBuf;
    worker.postMessage(msg, samplesBuf ? [samplesBuf] : []);
  } catch (err) {
    setStatus("error", "error");
    log("benchmark failed: " + (err.message || err), "err");
    finishBusy();
    reflectModelState();
  }
});

let audioCtx = null;
let micNode = null;
let micStream = null;
let micSource = null;
let onFrame = null;

async function startMic() {
  audioCtx = new AudioContext({ sampleRate: SR });
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
    },
  });
  if (audioCtx.state === "suspended") await audioCtx.resume();
  await audioCtx.audioWorklet.addModule(
    new URL("./mic-processor.js", import.meta.url),
  );
  micSource = audioCtx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(audioCtx, "mic-processor");
  micNode.port.onmessage = (e) => onFrame && onFrame(e.data);
  micSource.connect(micNode);
  micNode.connect(audioCtx.destination);
  if (audioCtx.sampleRate !== SR)
    log(
      `note: AudioContext is ${audioCtx.sampleRate} Hz, expected ${SR} Hz — accuracy may suffer`,
      "err",
    );
}

async function stopMic() {
  onFrame = null;
  try {
    micSource && micSource.disconnect();
    micNode && micNode.disconnect();
  } catch {}
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) {
    try {
      await audioCtx.close();
    } catch {}
  }
  audioCtx = micNode = micStream = micSource = null;
}

function rms(frame) {
  let s = 0;
  for (let i = 0; i < frame.length; i++) s += frame[i] * frame[i];
  return Math.sqrt(s / frame.length);
}

function setLevel(el, frame) {
  const v = Math.min(1, rms(frame) * 4);
  el.style.width = (v * 100).toFixed(0) + "%";
}

// ── audio normalization ──

const NORM_TARGET = 0.85;
let normPeak = 0;
let smoothGain = 1;

function normalizeAudio(samples) {
  if (!els.normalize.checked) return samples;

  let bufPeak = 0;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > bufPeak) bufPeak = a;
  }

  // envelope follower: fast attack, slow release
  if (bufPeak > normPeak) {
    normPeak += (bufPeak - normPeak) * 0.5;
  } else {
    normPeak += (bufPeak - normPeak) * 0.0005;
  }

  const targetGain = normPeak > 1e-4 ? Math.min(NORM_TARGET / normPeak, 20) : 1;

  // smooth gain to prevent burst on silence→speech transition
  if (targetGain > smoothGain) {
    smoothGain += (targetGain - smoothGain) * 0.08;
  } else {
    smoothGain += (targetGain - smoothGain) * 0.5;
  }

  if (Math.abs(smoothGain - 1) > 0.01) {
    for (let i = 0; i < samples.length; i++) samples[i] *= smoothGain;
  }
  return samples;
}

function micErrorMessage(err) {
  const n = err && err.name;
  if (n === "NotAllowedError" || n === "SecurityError")
    return "microphone permission denied";
  if (n === "NotFoundError") return "no microphone found";
  if (n === "NotReadableError") return "microphone is in use by another app";
  return (err && err.message) || String(err);
}

els.file.addEventListener(
  "change",
  () =>
    (els.fileBtn.disabled = readyState !== "ready" || !els.file.files.length),
);

async function decodeAudioFile(file) {
  const ab = await file.arrayBuffer();
  const tmp = new AudioContext();
  const dec = await tmp.decodeAudioData(ab);
  await tmp.close();
  const off = new OfflineAudioContext(1, Math.ceil(dec.duration * SR), SR);
  const src = off.createBufferSource();
  src.buffer = dec;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0).slice();
}

els.fileBtn.addEventListener("click", async () => {
  if (busy || !els.file.files.length) return;
  startBusy();
  els.fileBtn.disabled = true;
  try {
    setStatus("decoding audio…", "loading");
    const file = els.file.files[0];
    const samples = await decodeAudioFile(file);
    normalizeAudio(samples);
    log(`file: ${file.name} — ${(samples.length / SR).toFixed(2)} s`, "dim");
    await ensureReady();
    setStatus("transcribing…", "loading");
    fullDiagStart = performance.now();
    worker.postMessage(
      { type: "transcribeFull", samples: samples.buffer, langId: langId(), beamWidth: parseInt(els.beamWidth.value) },
      [samples.buffer],
    );
  } catch (err) {
    setStatus("error", "error");
    log("ERROR: " + (err.stack || err.message), "err");
    finishBusy();
  }
  els.fileBtn.disabled = readyState !== "ready" || !els.file.files.length;
});

let recording = false;
let recBuffers = [];
let recStart = 0;
let recTimerId = null;

function finishRecordUI() {
  recording = false;
  if (recTimerId) {
    clearInterval(recTimerId);
    recTimerId = null;
  }
  els.recBtn.textContent = "● Record";
  els.recBtn.classList.remove("is-stop");
  els.recBtn.classList.add("is-start");
  els.recLevel.style.width = "0%";
  els.recBtn.disabled = false;
}

els.recBtn.addEventListener("click", async () => {
  if (!recording) {
    if (busy) return;
    els.recBtn.disabled = true;
    try {
      await startMic();
    } catch (err) {
      log("mic error: " + micErrorMessage(err), "err");
      setStatus("mic error", "error");
      els.recBtn.disabled = false;
      return;
    }
    try {
      recBuffers = [];
      recording = true;
      recStart = performance.now();
      onFrame = (frame) => {
        normalizeAudio(frame);
        recBuffers.push(frame);
        setLevel(els.recLevel, frame);
      };
      els.recBtn.textContent = "■ Stop";
      els.recBtn.classList.replace("is-start", "is-stop");
      els.recBtn.disabled = false;
      els.recTimer.textContent = "0.0s";
      recTimerId = setInterval(
        () =>
          (els.recTimer.textContent =
            ((performance.now() - recStart) / 1000).toFixed(1) + "s"),
        100,
      );
      setStatus("recording…", "loading");
      ensureReady().catch((err) =>
        log("model load failed: " + err.message, "err"),
      );
    } catch (err) {
      log("record error: " + (err.message || err), "err");
      await stopMic();
      finishRecordUI();
    }
  } else {
    clearInterval(recTimerId);
    recTimerId = null;
    recording = false;
    await stopMic();
    els.recBtn.textContent = "● Record";
    els.recBtn.classList.replace("is-stop", "is-start");
    els.recLevel.style.width = "0%";
    const total = recBuffers.reduce((n, f) => n + f.length, 0);
    if (total === 0) {
      setStatus("model ready", "online");
      return;
    }
    const samples = new Float32Array(total);
    let o = 0;
    for (const f of recBuffers) {
      samples.set(f, o);
      o += f.length;
    }
    recBuffers = [];
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
      const a = Math.abs(samples[i]);
      if (a > peak) peak = a;
    }
    log(
      `recorded ${(samples.length / SR).toFixed(2)} s · peak amplitude ${peak.toFixed(
        3,
      )} (≈0 ⇒ mic captured silence)`,
      peak < 0.01 ? "err" : "dim",
    );
    startBusy();
    setStatus("transcribing…", "loading");
    try {
      await ensureReady();
      fullDiagStart = performance.now();
      worker.postMessage(
        { type: "transcribeFull", samples: samples.buffer, langId: langId(), beamWidth: parseInt(els.beamWidth.value) },
        [samples.buffer],
      );
    } catch (err) {
      setStatus("error", "error");
      log("ERROR: " + (err.message || err), "err");
      finishBusy();
    }
  }
});

let live = false;
let liveSegmenting = false;
let segActive = false;
let liveBatch = [];
let liveBatchLen = 0;
let pendingBlocks = 0;
let preRoll = [];
let preRollLen = 0;
let liveSegments = [];
let liveVad = null;

const LIVE_FLUSH = 3200;
const MAX_PENDING_BLOCKS = 24;
const PRE_ROLL_SAMPLES = SR * 0.2;
const VAD_ONSET_MS = 120;
const VAD_HANGOVER_MS = 650;
const VAD_NOISE_MULT = 3.0;
const VAD_RELEASE_MULT = 1.8;
const VAD_MIN_RMS = 0.005;
const VAD_FLOOR_ALPHA = 0.02;

function createEnergyVAD({ onSpeechStart, onSpeechEnd }) {
  const ONSET = (SR * VAD_ONSET_MS) / 1000;
  const HANGOVER = (SR * VAD_HANGOVER_MS) / 1000;
  let floor = null;
  let active = false;
  let onset = 0;
  let silence = 0;

  const start = () => {
    active = true;
    onset = 0;
    silence = 0;
    onSpeechStart();
  };

  const end = () => {
    active = false;
    onset = 0;
    silence = 0;
    onSpeechEnd();
  };

  return {
    get active() {
      return active;
    },
    process(level, nSamples) {
      if (floor == null) floor = level;
      if (active) {
        const releaseThr = Math.max(
          floor * VAD_RELEASE_MULT,
          VAD_MIN_RMS * 0.6,
        );
        silence = level < releaseThr ? silence + nSamples : 0;
        if (silence >= HANGOVER) end();
        return;
      }
      const speechThr = Math.max(floor * VAD_NOISE_MULT, VAD_MIN_RMS);
      if (level < speechThr)
        floor = (1 - VAD_FLOOR_ALPHA) * floor + VAD_FLOOR_ALPHA * level;
      onset = level > speechThr ? onset + nSamples : 0;
      if (onset >= ONSET) start();
    },
  };
}

function stopLiveUI() {
  live = false;
  segActive = false;
  els.liveBtn.textContent = "● Start listening";
  els.liveBtn.classList.remove("is-stop");
  els.liveBtn.classList.add("is-start");
  els.liveDot.classList.remove("online");
  els.liveLevel.style.width = "0%";
  els.liveBtn.disabled = false;
}

function postLiveChunk() {
  if (liveBatchLen === 0) return;
  const buf = new Float32Array(liveBatchLen);
  let o = 0;
  for (const f of liveBatch) {
    buf.set(f, o);
    o += f.length;
  }
  liveBatch = [];
  liveBatchLen = 0;
  if (pendingBlocks >= MAX_PENDING_BLOCKS) {
    log("dropping audio — decode can't keep up", "err");
    return;
  }
  pendingBlocks++;
  worker.postMessage({ type: "streamAudio", samples: buf.buffer }, [
    buf.buffer,
  ]);
}

function keepPreRoll(frame) {
  preRoll.push(frame);
  preRollLen += frame.length;
  while (preRollLen - preRoll[0].length >= PRE_ROLL_SAMPLES) {
    preRollLen -= preRoll[0].length;
    preRoll.shift();
  }
}

let liveCurText = "";
let liveCurLang = null;

function renderLive(partialText, partialLang) {
  liveCurText = partialText;
  liveCurLang = partialLang;
  const lines = liveSegments.map((s) => formatLine(s.text, s.lang));
  const all = [...lines, formatLine(partialText, partialLang)]
    .filter(Boolean)
    .join("\n");
  els.transcript.textContent = all || "—";
  els.transcript.classList.toggle("is-empty", !all);
  setLangChip(
    partialLang ||
      (liveSegments.length ? liveSegments[liveSegments.length - 1].lang : null),
  );
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function commitLiveSegment(text, lang) {
  if (text) {
    liveSegments.push({ text, lang });
    log(`segment committed — ${text.length} chars`, "dim");
  }
  renderLive("", null);
}

function openSegment() {
  if (segActive) return;
  segActive = true;
  liveBatch = [];
  liveBatchLen = 0;
  pendingBlocks = 0;
  worker.postMessage({ type: "streamStart", langId: langId(), beamWidth: parseInt(els.beamWidth.value) });
  streamDiagStart = performance.now();
  for (const f of preRoll) {
    liveBatch.push(f);
    liveBatchLen += f.length;
  }
  setStatus("transcribing…", "loading");
}

function closeSegment() {
  if (!segActive) return;
  segActive = false;
  postLiveChunk();
  worker.postMessage({ type: "streamEnd" });
  setStatus("listening…", "online");
}

els.liveBtn.addEventListener("click", async () => {
  if (live) {
    live = false;
    await stopMic();
    if (segActive) closeSegment();
    liveBatch = [];
    liveBatchLen = 0;
    preRoll = [];
    preRollLen = 0;
    stopLiveUI();
    return;
  }
  if (busy) return;
  els.liveBtn.disabled = true;
  try {
    await startMic();
  } catch (err) {
    log("mic error: " + micErrorMessage(err), "err");
    setStatus("mic error", "error");
    els.liveBtn.disabled = false;
    return;
  }

  live = true;
  liveSegmenting = true;
  segActive = false;
  liveBatch = [];
  liveBatchLen = 0;
  pendingBlocks = 0;
  preRoll = [];
  preRollLen = 0;
  liveSegments = [];
  setTranscript("", null);
  els.liveBtn.textContent = "■ Stop";
  els.liveBtn.classList.replace("is-start", "is-stop");
  els.liveBtn.disabled = false;
  els.liveDot.classList.add("online");
  setStatus("loading model…", "loading");

  const useVad = els.vadToggle.checked;

  if (useVad) {
    liveVad = createEnergyVAD({
      onSpeechStart: () => { if (live) openSegment(); },
      onSpeechEnd: () => { if (live) closeSegment(); },
    });
  } else {
    openSegment();
  }

  let ready = false;
  onFrame = (frame) => {
    normalizeAudio(frame);
    setLevel(els.liveLevel, frame);
    if (useVad && ready) liveVad.process(rms(frame), frame.length);
    if (segActive) {
      liveBatch.push(frame);
      liveBatchLen += frame.length;
      if (liveBatchLen >= LIVE_FLUSH) postLiveChunk();
    }
    keepPreRoll(frame);
  };

  try {
    await ensureReady();
    if (!live) return;
    ready = true;
    setStatus("listening…", "online");
  } catch (err) {
    log("model load failed: " + err.message, "err");
    setStatus("error", "error");
    await stopMic();
    stopLiveUI();
  }
});

// ── tab audio capture ──

let tabCapture = false;
let tabSegmenting = false;
let tabSegActive = false;
let tabBatch = [];
let tabBatchLen = 0;
let tabPending = 0;
let tabFramesIn = 0;
let tabFramesOut = 0;
let tabPreRoll = [];
let tabPreRollLen = 0;
let tabSegments = [];
let tabCurText = "";
let tabCurLang = null;
let tabVad = null;
let tabCtx = null;
let tabNode = null;
let tabStream = null;
let tabSource = null;

function stopTabUI() {
  tabCapture = false;
  tabSegActive = false;
  els.tabBtn.textContent = "● Capture tab audio";
  els.tabBtn.classList.remove("is-stop");
  els.tabBtn.classList.add("is-start");
  els.tabDot.classList.remove("online");
  els.tabLevel.style.width = "0%";
  els.tabBtn.disabled = false;
}

function postTabChunk() {
  if (tabBatchLen === 0) return;
  const buf = new Float32Array(tabBatchLen);
  let o = 0;
  for (const f of tabBatch) { buf.set(f, o); o += f.length; }
  tabBatch = [];
  tabBatchLen = 0;
  if (tabPending >= MAX_PENDING_BLOCKS) {
    log(`dropping ${buf.length} tab samples — engine backlog ${tabPending}`, "err");
    return;
  }
  tabPending++;
  tabFramesOut += buf.length;
  worker.postMessage({ type: "streamAudio", samples: buf.buffer }, [buf.buffer]);
}

function keepTabPreRoll(frame) {
  tabPreRoll.push(frame);
  tabPreRollLen += frame.length;
  while (tabPreRollLen - tabPreRoll[0].length >= PRE_ROLL_SAMPLES) {
    tabPreRollLen -= tabPreRoll[0].length;
    tabPreRoll.shift();
  }
}

function renderTab(partialText, partialLang) {
  tabCurText = partialText;
  tabCurLang = partialLang;
  const lines = tabSegments.map((s) => formatLine(s.text, s.lang));
  const all = [...lines, formatLine(partialText, partialLang)].filter(Boolean).join("\n");
  els.transcript.textContent = all || "—";
  els.transcript.classList.toggle("is-empty", !all);
  setLangChip(partialLang || (tabSegments.length ? tabSegments[tabSegments.length - 1].lang : null));
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function commitTabSegment(text, lang) {
  if (text) {
    tabSegments.push({ text, lang });
    log(`tab segment committed — ${text.length} chars`, "dim");
  }
  renderTab("", null);
}

function openTabSegment() {
  if (tabSegActive) return;
  tabSegActive = true;
  tabBatch = [];
  tabBatchLen = 0;
  tabPending = 0;
  worker.postMessage({ type: "streamStart", langId: langId(), beamWidth: parseInt(els.beamWidth.value) });
  for (const f of tabPreRoll) {
    tabBatch.push(f);
    tabBatchLen += f.length;
  }
}

function closeTabSegment() {
  if (!tabSegActive) return;
  tabSegActive = false;
  postTabChunk();
  worker.postMessage({ type: "streamEnd" });
}

async function stopTabCapture() {
  if (!tabCapture) return;
  tabCapture = false;
  tabSegmenting = false;
  try {
    tabNode && tabNode.disconnect();
    tabSource && tabSource.disconnect();
  } catch {}
  if (tabCtx) { try { await tabCtx.close(); } catch {} }
  if (tabStream) tabStream.getTracks().forEach((t) => t.stop());
  tabCtx = tabNode = tabStream = tabSource = null;
  if (tabSegActive) closeTabSegment();
  tabBatch = []; tabBatchLen = 0; tabPreRoll = []; tabPreRollLen = 0;
  const durSec = tabFramesIn / SR;
  log(`tab capture ended — ${durSec.toFixed(1)} s captured, ${tabFramesOut} samples sent to engine`, "dim");
  tabFramesIn = 0; tabFramesOut = 0;
  stopTabUI();
}

els.tabBtn.addEventListener("click", async () => {
  if (tabCapture) { await stopTabCapture(); return; }
  if (busy) return;
  els.tabBtn.disabled = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      audio: true,
      video: { width: 1, height: 1 },
    });
    // user cancelled
    if (!stream) { els.tabBtn.disabled = false; return; }
  } catch (err) {
    if (err.name === "NotAllowedError" || err.name === "AbortError") {
      log("tab picker cancelled", "dim");
    } else {
      log("tab capture error: " + micErrorMessage(err), "err");
      setStatus("error", "error");
    }
    els.tabBtn.disabled = false;
    return;
  }
  const audioTrack = stream.getAudioTracks()[0];
  if (!audioTrack) {
    log("selected tab has no audio track", "err");
    stream.getTracks().forEach((t) => t.stop());
    els.tabBtn.disabled = false;
    return;
  }

  try {
    tabCtx = new AudioContext({ sampleRate: SR });
    await tabCtx.audioWorklet.addModule(new URL("./mic-processor.js", import.meta.url));
    tabSource = tabCtx.createMediaStreamSource(stream);
    tabNode = new AudioWorkletNode(tabCtx, "mic-processor");
    tabNode.port.onmessage = (e) => {
      const frame = e.data;
      normalizeAudio(frame);
      tabFramesIn += frame.length;
      setLevel(els.tabLevel, frame);
      if (tabUseVad && tabCapture) tabVad.process(rms(frame), frame.length);
      if (tabSegActive) {
        tabBatch.push(frame);
        tabBatchLen += frame.length;
        if (tabBatchLen >= LIVE_FLUSH) postTabChunk();
      }
      keepTabPreRoll(frame);
    };
    tabSource.connect(tabNode);
    // NOT connecting to destination — tab audio already plays from its source
    tabStream = stream;

    tabCapture = true;
    tabSegmenting = true;
    tabSegActive = false;
    tabBatch = []; tabBatchLen = 0; tabPending = 0;
    tabPreRoll = []; tabPreRollLen = 0;
    tabSegments = [];
    setTranscript("", null);
    els.tabBtn.textContent = "■ Stop";
    els.tabBtn.classList.replace("is-start", "is-stop");
    els.tabBtn.disabled = false;
    els.tabDot.classList.add("online");
    setStatus("loading model…", "loading");

    const tabUseVad = els.vadToggle.checked;

    if (tabUseVad) {
      tabVad = createEnergyVAD({
        onSpeechStart: () => { if (tabCapture) openTabSegment(); },
        onSpeechEnd: () => { if (tabCapture) closeTabSegment(); },
      });
    } else {
      openTabSegment();
    }

    let ready = false;
    // patch onmessage to also handle model-ready for tab
    const origOnMsg = worker.onmessage;
    const tabCb = (e) => {
      const m = e.data;
      if (m.type === "ready" && !ready) {
        ready = true;
        setStatus("listening…", "online");
        const label = { TURBO: "80ms", FAST: "160ms", BALANCED: "320ms", NORMAL: "560ms", HIGH: "1120ms" }[m.profile] || m.profile;
        log(`tab capture active (encoder on ${m.encoderEP || "webgpu"}, ${label})`, "ok");
      }
      if (m.type === "partial" && tabSegmenting) {
        renderTab(m.text, m.lang);
      }
      if (m.type === "final" && tabSegmenting) {
        commitTabSegment(m.text, m.lang);
      }
    };
    worker.addEventListener("message", tabCb);

    // clean up listener when tab stops
    const origStopCb = stopTabCapture;
    const oldStop = stopTabCapture;
    stopTabCapture = async () => {
      worker.removeEventListener("message", tabCb);
      stopTabCapture = oldStop;
      await oldStop();
    };

    ensureReady().catch((err) => {
      log("model load failed: " + err.message, "err");
      if (!ready) {
        setStatus("error", "error");
        stopTabCapture();
      }
    });
  } catch (err) {
    log("tab setup error: " + (err.message || err), "err");
    stream.getTracks().forEach((t) => t.stop());
    stopTabUI();
  }
});

els.stripTag.addEventListener("change", () => {
  if (liveSegmenting) renderLive(liveCurText, liveCurLang);
  else setTranscript(lastText, lastLang);
});

log(
  "Ready. Pick a mode. First run downloads ~750 MB (cached on-device after — re-open is instant).",
  "dim",
);
log(
  "Serve over http://localhost or https:// (module worker + mic require it — file:// won't work).",
  "dim",
);
