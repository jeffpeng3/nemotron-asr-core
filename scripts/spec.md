# ONNX Model I/O Specification

Based on the dynamo single encoder (`export-single-encoder-dynamo.py`, right_context=13).
I/O names match the reference `onnx-community` model for compatibility.

---

## Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| `SR` | 16000 | Sample rate (Hz) |
| `N_FFT` | 512 | FFT size |
| `HOP` | 160 | Hop length (10ms) |
| `WIN` | 400 | Window length (25ms) |
| `N_MELS` | 128 | Mel filterbank channels |
| `CACHE_FRAMES` | 9 | Pre-encode cache (past context frames) |
| `ENC_IN` | 65 | Total encoder input frames (profile-dependent; 17–121) |
| `right_context` | 13 | Attention look-ahead (single value for all profiles) |
| `left_context` | 56 | Cache channel dimension |
| `conv_context` | 8 | Cache time width |
| `subsampling_factor` | 8 | Encoder temporal subsampling |
| `LAYERS` | 24 | Encoder hidden layers |
| `D_MODEL` | 1024 | Encoder hidden size |
| `DEC_LAYERS` | 2 | Decoder LSTM layers |
| `DEC_HID` | 640 | Decoder hidden size |
| `VOCAB` | 13088 | Vocabulary size |
| `BLANK` | 13087 | Blank token ID |
| `MAX_SYM` | 10 | Max symbols per frame |

---

## 1. Encoder (`encoder.onnx`)

### Inputs

| ONNX Name | Shape | Type | dynamic_axes | Description |
|-----------|-------|------|-------------|-------------|
| `audio_signal` | `[B, 128, T]` | float32 | `B`: batch, `T`: time (dynamic, 17–121) | JS passes `[1, ENC_IN, 128]`; internal Transpose maps `[B,T,128]`→`[B,128,T]` |
| `length` | `[B]` | int64 | `B`: batch | Valid frame count per sample |
| `cache_last_channel` | `[B, 24, 56, 1024]` | float32 | `B`: batch, 3rd: cache_channel_time | LSTM-style left context cache |
| `cache_last_time` | `[B, 24, 1024, 8]` | float32 | `B`: batch, 4th: cache_time_width | LSTM-style time axis cache |
| `cache_last_channel_len` | `[B]` | int64 | `B`: batch | Cache length (0 on first call) |
| `lang_id` | `[B]` | int64 | `B`: batch | Language ID (101 = auto) |

**Notes:**
- ONNX declares `audio_signal` as `['batch', 128, 'time']` (mel first). At runtime JS passes `[1, ENC_IN, 128]`. The graph's first Transpose op converts input layout, so ORT accepts either by total element count.
- `audio_signal` dim 2 (`time`) is dynamic (range 17–121, mapped to profiles: TURBO=17, FAST=25, BALANCED=41, NORMAL=65, HIGH=121).

### Outputs

| ONNX Name | Shape | Type | dynamic_axes | Description |
|-----------|-------|------|-------------|-------------|
| `outputs` | `[B, 1024, T_out]` | float32 | `B`: batch, `T_out`: time | Encoder output frames; JS reads as `dims[1]` for T_out, uses stride `D_MODEL=1024` |
| `cache_last_channel_next` | `[B, 24, 56, 1024]` | float32 | `B`: batch, 3rd: cache_channel_time | Updated cache for next call |
| `cache_last_time_next` | `[B, 24, 1024, 8]` | float32 | `B`: batch, 4th: cache_time_width | Updated time cache |
| `cache_last_channel_len_next` | `[B]` | int64 | `B`: batch | Updated cache length |

**Notes:**
- Output `encoded_lengths` may also be present in the graph but is **NOT used** by JS. JS determines output length via `outputs.dims[1]`.
- Output layout: internal Transpose converts `[B, 1024, T_out]` → `[B, T_out, 1024]` at runtime.
- `cache_last_channel_len_next` — note the order: `len_next`, NOT `next_len`.

### Chunk sizes per profile

A single encoder (right_context=13) handles all latency profiles. The encoder input length varies by profile; `right_context` caps the attention look-ahead but does not force a minimum input size:

| Profile | NEW_FRAMES | ENC_IN (9 + NEW_FRAMES) | chunk_size_ms | chunk_samples |
|---------|------------|-------------------------|---------------|---------------|
| TURBO | 8 | 17 | 80 | 1280 |
| FAST | 16 | 25 | 160 | 2560 |
| BALANCED | 32 | 41 | 320 | 5120 |
| NORMAL | 56 | 65 | 560 | 8960 |
| HIGH | 112 | 121 | 1120 | 17920 |

All profiles share the same cache dimensions (56, 8), I/O names, and dynamic axes.

---

## 2. Decoder (`decoder.onnx`)

### Inputs

| ONNX Name | Shape | Type | Description |
|-----------|-------|------|-------------|
| `targets` | `[B, 1]` | int64 | Token ID to decode |
| `h_in` | `[2, B, 640]` | float32 | LSTM hidden state (DEC_LAYERS, B, DEC_HID) |
| `c_in` | `[2, B, 640]` | float32 | LSTM cell state (DEC_LAYERS, B, DEC_HID) |

### Outputs

| ONNX Name | Shape | Type | Description |
|-----------|-------|------|-------------|
| `h_out` | `[2, B, 640]` | float32 | Updated LSTM hidden state |
| `c_out` | `[2, B, 640]` | float32 | Updated LSTM cell state |
| `decoder_output` | `[B, 640]` | float32 | Decoder output for joint step |

---

## 3. Joint (`joint.onnx`)

### Inputs

| ONNX Name | Shape | Type | Description |
|-----------|-------|------|-------------|
| `encoder_output` | `[B, 1, 1024]` | float32 | Single encoder frame |
| `decoder_output` | `[B, 1, 640]` | float32 | Single decoder output |

### Outputs

| ONNX Name | Shape | Type | Description |
|-----------|-------|------|-------------|
| `joint_output` | `[B, 1, 1, 13088]` | float32 | Logits over vocabulary (VOCAB=13088) |
