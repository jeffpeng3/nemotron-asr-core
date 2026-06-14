# nemotron-asr-core

In-browser speech recognition using NVIDIA's Nemotron 3.5 (FastConformer-RNNT)
via `onnxruntime-web` on WebGPU. Fully client-side — no server needed.

```
npm install nemotron-asr-core
```

## Live Demo

A static demo app is included in the repo under `example/`. To run it:

```bash
python3 -m http.server 8000
# open http://localhost:8000/example/
```

`file://` won't work — the demo uses module workers and mic access.

## Features

- **5 latency profiles** — 80 ms to 1120 ms, trade latency for accuracy
- **Streaming & full-audio** — mic capture or file transcription
- **Beam search decoding** (configurable 1–5 beams)
- **Multilingual** — auto-detect or pick from 20+ language IDs
- **WebGPU accelerated** — falls back to WASM (CPU) on demand

## Usage

```js
import { AsrEngine } from "nemotron-asr-core";

// Callbacks for UI updates
const engine = new AsrEngine({
  progress(label, loaded, total, cached) {
    console.log(`${label}: ${loaded}/${total}`);
  },
  status(detail) {
    console.log(detail);
  },
  partial(text, lang, progress) {
    console.log(`partial (${(progress * 100).toFixed(0)}%): ${text}`);
  },
  ep(encoder, provider, note) {
    console.log(`encoder: ${provider}${note ? ` (${note})` : ""}`);
  },
});

// Download model weights (~690 MB, cached on-device)
await engine.init();

// ── Full audio transcription ──
const result = await engine.transcribe(samples, 101);
// samples: Float32Array of 16 kHz PCM
// 101 = auto-detect language

console.log(result.text);
// { text: "hello world <en-US>", lang: "en-US", tokens: 12, timing: { ... } }

// ── Streaming (mic / tab capture) ──
const session = engine.session(101);

// push chunks as they arrive
for (const chunk of audioChunks) {
  const partial = await session.feed(chunk);
  if (partial) console.log(partial.text);
}

const final = await session.end();
console.log(final.text);

// ── Benchmark ──
const results = await engine.benchmark({ duration: 10 });
for (const r of results) {
  console.log(`${r.profile} RTF ${r.rtf.toFixed(3)}`);
}
```

## Latency Profiles

| Profile   | Latency | Encoder |
|-----------|---------|---------|
| `TURBO`   | 80 ms   | encoder_80ms.onnx |
| `FAST`    | 160 ms  | encoder_160ms.onnx |
| `BALANCED`| 320 ms  | encoder_320ms.onnx |
| `NORMAL`  | 560 ms  | encoder_560ms.onnx |
| `HIGH`    | 1120 ms | encoder_1120ms.onnx |

Lower latency = fewer context frames = less accurate. Choose the profile
that fits your use case.

```js
await engine.switchProfile("HIGH");  // highest accuracy
```

## API

### `new AsrEngine(callbacks?, options?)`

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `profile` | `"NORMAL"` | Initial latency profile |
| `beamWidth` | `1` | Beam search width (1 = greedy) |
| `ensureCPU` | `false` | Force encoder to WASM (skip WebGPU) |

**Callbacks:**

| Callback | Arguments | Description |
|----------|-----------|-------------|
| `progress` | `(label, loaded, total, cached?)` | Model download progress |
| `status` | `(detail)` | Status messages |
| `partial` | `(text, lang, progress?)` | Partial transcription result |
| `ep` | `(isEncoder, provider, note?)` | Execution provider selection |

### `session(langId)` → `Session`

Create a streaming session.

| Method | Returns | Description |
|--------|---------|-------------|
| `feed(samples)` | `{text, lang} \| null` | Push audio chunk, get partial result |
| `end()` | `{text, lang, tokens, timing}` | Finalize and get complete result |

### `benchmark(opts?)` → `BenchmarkProfileResult[]`

Test all profiles and return RTF (Real-Time Factor) measurements.

**Options:** `{ profiles?, duration?, langId?, warmup?, forceAll?, samples? }`

### `switchProfile(name)`

Switch latency profile at runtime (reloads encoder).

### `clearCache()`

Remove cached model weights.

### `getPerfStats()` → `Record<string, {ms, calls, avg}>`

Performance statistics.

## Language IDs

Pass `101` for auto-detect. Use `0` for English, `4` for Chinese, etc.
See `LANG_TO_ID` and `langId()` exports for the full list.

## Architecture

```
app.js (main thread)  ←→  worker.js (Web Worker)  ←→  Hugging Face / Cache API
                            └── AsrEngine
                                  ├── Mel filterbank + FFT
                                  ├── Encoder (WebGPU or WASM)
                                  ├── Decoder + Joint (WASM)
                                  └── RNN-T beam search
```

- Model weights (~690 MB) are fetched once from Hugging Face and cached
  via the Cache API (Service Worker).
- All inference runs off the main thread via a Web Worker.
- The encoder runs on WebGPU when available; decoder + joint always run
  on WASM (CPU) for compatibility.

## Requirements

- **Browser**: Chrome 113+ / Edge 113+ (WebGPU), or any browser with
  WebAssembly. Safari 18+ on iOS.
- **HTTP origin**: `file://` won't work — module workers and mic access
  require `http://` or `https://`.
- **GPU**: ~690 MB of GPU-accessible memory. Integrated GPUs with
  < 690 MB of VRAM will page weights over PCIe (slow).
