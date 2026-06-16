import { AsrEngine } from "@jeffpeng3/nemotron-asr-core";

const post = (m, t) => self.postMessage(m, t || []);

let engine = null;

function makeEngine(profile, beamWidth) {
  return new AsrEngine({
    progress(label, loaded, total, cached) {
      post({ type: "progress", label, loaded, total, cached: !!cached });
    },
    status(detail) {
      post({ type: "status", detail });
    },
    partial(text, lang, progress) {
      post({ type: "partial", text, lang, progress });
    },
    ep(encoder, ep, note) {
      post({ type: "ep", encoder, ep, note });
    },
    perf({ stats, profile }) {
      post({ type: "perf", stats, profile });
    },
  }, { profile: profile || "NORMAL", beamWidth: beamWidth || 1 });
}

function postPerf() {
  const stats = engine.getPerfStats();
  post({ type: "perf", stats });
}

let sess = null;

async function handle(m) {
  switch (m.type) {
    case "init": {
      const profile = m.profile || "NORMAL";
      const bw = m.beamWidth || 1;
      if (!engine) {
        engine = makeEngine(profile, bw);
      } else {
        engine._beamWidth = bw;
        post({ type: "status", detail: `beam width: ${bw}${bw > 1 ? " (beam search)" : " (greedy)"}` });
        if (engine.profile !== profile) {
          await engine.switchProfile(profile);
        }
      }
      if (engine.ready) {
        post({ type: "ready", encoderEP: engine.encoderEP, profile: engine.profile });
        postPerf();
        break;
      }
      engine.init().then(() => {
        post({ type: "ready", encoderEP: engine.encoderEP, profile: engine.profile });
        postPerf();
      }, (err) => {
        throw err;
      });
      break;
    }
    case "perfStats": {
      postPerf();
      break;
    }
    case "transcribeFull": {
      if (!engine) break;
      if (m.beamWidth != null) {
        engine._beamWidth = m.beamWidth;
        post({ type: "status", detail: `beam width: ${m.beamWidth}${m.beamWidth > 1 ? " (beam search)" : " (greedy)"}` });
      }
      await (engine.ready ? Promise.resolve() : engine.init());
      const result = await engine.transcribe(
        new Float32Array(m.samples),
        m.langId,
      );
      post({
        type: "final",
        text: result.text,
        lang: result.lang,
        tokens: result.tokens,
        timing: result.timing,
      });
      postPerf();
      break;
    }
    case "streamStart": {
      if (!engine) break;
      if (m.beamWidth != null) {
        engine._beamWidth = m.beamWidth;
        post({ type: "status", detail: `beam width: ${m.beamWidth}${m.beamWidth > 1 ? " (beam search)" : " (greedy)"}` });
      }
      await (engine.ready ? Promise.resolve() : engine.init());
      sess = engine.session(m.langId);
      post({ type: "stream-ready" });
      break;
    }
    case "streamAudio": {
      if (!sess) break;
      const result = await sess.feed(new Float32Array(m.samples));
      if (result) {
        post({ type: "partial", text: result.text, lang: result.lang });
      }
      post({ type: "stream-tick" });
      break;
    }
    case "streamEnd": {
      if (!sess) break;
      const result = await sess.end();
      sess = null;
      if (result && result.text) {
        post({
          type: "final",
          text: result.text,
          lang: result.lang,
          tokens: result.tokens,
          timing: result.timing,
        });
      }
      break;
    }
    case "benchmark": {
      if (!engine) engine = makeEngine("HIGH", 1);
      await (engine.ready ? Promise.resolve() : engine.init());
      const samp = m.samples ? new Float32Array(m.samples) : null;
      const t0 = performance.now();
      const results = await engine.benchmark({
        duration: m.duration ?? 10,
        samples: samp,
        beamWidths: m.beamWidths || [1],
      });
      const totalMs = performance.now() - t0;
      post({ type: "benchmark", results, totalMs });
      break;
    }
    case "clearCache": {
      if (engine) await engine.clearCache();
      break;
    }
  }
}



let chain = Promise.resolve();
self.onmessage = (e) => {
  chain = chain
    .then(() => handle(e.data))
    .catch((err) => {
      sess = null;
      post({
        type: "error",
        message: (err && (err.stack || err.message)) || String(err),
      });
    });
};

self.onerror = (msg) =>
  post({ type: "error", message: `worker error: ${msg}` });

self.onunhandledrejection = (e) =>
  post({
    type: "error",
    message: `unhandled: ${(e && e.reason && (e.reason.message || e.reason)) || "unknown"}`,
  });
