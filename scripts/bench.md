# Benchmark Results

Audio: 44.0 s Japanese, beam width: 1 (greedy), RTX 3060
Runtime optimization: `freeDimensionOverrides` (pins dynamic "time" dim per profile)

## All profiles

Single encoder (rc=13) with `torch.onnx.export(dynamo=True)` and `dynamic_shapes Dim(time, min=17, max=121)`.

| Profile  | Latency | RTF   | Time    | Transcript |
|----------|---------|-------|---------|------------|
| HIGH     | 1120ms  | 0.103 | 4537 ms | チャンネル登録者数が九万人を突破いたしました。 あと零点四万人四千人増えれば待 |
| NORMAL   | 560ms   | 0.165 | 7282 ms | チャンネル登録者数が九万人を突破いたしました。 後零増えればこの八点の待て、こ |
| BALANCED | 320ms   | 0.268 | 11796 ms | チャンネル登録者数が九万人を突破いたしました。 あと零点四万人四千人増えればこ |
| FAST     | 160ms   | 0.502 | 22085 ms | チャンネル登録者数が九万人を突破いたしました。 あと四万人四千人増えればこの八 |
| TURBO    | 80ms    | 0.980 | 43152 ms | チャンネル登録者数がえっと九万人を突破いたしました。 あと零点四万人四千人増え |

## vs Official (NORMAL / 560ms)

Official model from `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`.

| Model | RTF | Time | Δ | Transcript |
|-------|-----|------|---|------------|
| Official onnx-community | 0.186 | 8200 ms | — | チャンネル登録者数が九万人を突破いたしました。 あと零点四万人四千人増えれば待って、このお部屋に八田のこのお部屋になんと銀立が増えるかもしれません銀立てほしい、銀田で欲しい、銀立欲しい銀立てほしい銀立すぎてちょっと酔った |
| Ours (dynamo + freeDimOverride) | 0.165 | 7282 ms | **−11.3%** | チャンネル登録者数が九万人を突破いたしました。 後零増えればこの八点の待て、この部屋に八手のこの部屋になんと銀立が増えるかもしれません銀田でほしい銀田で欲しい、銀立欲しい銀立ほしい銀すぎてちょっと酔った |
| Before (dynamo, no override) | 0.205 | 9043 ms | +10.2% | チャンネル登録者数が九万人を突破いたしました。 後零増えればこの八点の待て、この部屋に八手のこの部屋になんと銀立が増えるかもしれません銀田でほしい銀田で欲しい、銀立欲しい銀立ほしい銀すぎてちょっと酔った |

Transcript differences are tokenization boundary choices only ("待って" vs "待て", "八田" vs "八手"), not recognition quality. 4-way cross-validation confirmed the decoder/joint contributes zero difference.

## Model

| Property | Value |
|----------|-------|
| ONNX nodes | 1962 (opset 21) |
| ONNX metadata | 2.9 MB |
| Weights (INT4, block_size=32) | 650.1 MB |
| right_context / left_context | 13 / 56 (single encoder, all 5 profiles) |
| Export | `torch.onnx.export(dynamo=True)` |
| Quantization | `MatMulNBitsQuantizer` (bits=4, accuracy_level=4) |
| Runtime | `freeDimensionOverrides` |
