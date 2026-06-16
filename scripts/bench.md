# Benchmark Results

Audio: 44.0 s Japanese, RTX 3060
Runtime optimization: `freeDimensionOverrides` (pins dynamic "time" dim per profile),
GPU IO binding for cache tensors, GPU buffer pre-upload for encoder inputs,
per-profile warmup, removed CPU-GPU sync in encoderStep.

## All profiles & beam widths

Single encoder (rc=13) with `torch.onnx.export(dynamo=True)` and `dynamic_shapes Dim(time, min=17, max=121)`.

### Beam width: 1 (greedy)

| Profile  | Latency | RTF   | Time   | Transcript |
|----------|---------|-------|--------|------------|
| HIGH     | 1120ms  | 0.108 | 4756ms | チャンネル登録者数が九万人を突破いたしました。  後零増えれば待て、この部屋に八 |
| NORMAL   | 560ms   | 0.177 | 7802ms | チャンネル登録者数が九万人を突破いたしました。  後零増えればこの八手の待て、こ |
| BALANCED | 320ms   | 0.274 | 12050ms | チャンネル登録者数が九万人を突破いたしました。  あと零点四万人四千人増えればこ |
| FAST     | 160ms   | 0.511 | 22501ms | チャンネル登録者数が九万人を突破いたしました。  あと四万人四千人増えればこの八 |
| TURBO    | 80ms    | 1.033 | 45469ms | チャンネル登録者数が九万人を突破いたしました。  あと零点四万人四千人増えればこ |

### Beam width: 2

| Profile  | Latency | RTF   | Time   | Transcript |
|----------|---------|-------|--------|------------|
| HIGH     | 1120ms  | 0.122 | 5351ms | チャンネル登録者数が九万人を突破いたしました。 あと零点四万人四千人増えれば待て、この部屋に八手屋のこのお部屋になんと |
| NORMAL   | 560ms   | 0.185 | 8129ms | チャンネル登録者数が九万人を突破いたしました。 この八手の待て、この部屋に八手のこの部屋になんと銀立が増えるかもしれま |
| BALANCED | 320ms   | 0.269 | 11833ms | チンネル登録者数が九万人を突破いたしました。 あと零点四万人四千人増えればこの八点の待ってやのこのお部屋になんと銀立が |
| FAST     | 160ms   | 0.505 | 22244ms | チンネル登録者数が九万人を突破いたしました後千人増えればこの八手の待て、このお部屋に八手屋のこのお部屋になんと銀立が増え |
| TURBO    | 80ms    | 0.921 | 40542ms | チンネル登録者数がえっと九万人を突破いたしました。 あと零点四万人四千人増えればこの八の待て、このお部屋に八手のこの部 |

### Beam width: 3

| Profile  | Latency | RTF   | Time   | Transcript |
|----------|---------|-------|--------|------------|
| HIGH     | 1120ms  | 0.150 | 6622ms | チャンネル登録者数が九万人を突破いたしました。 あと零増えれば待て、この部屋に八手屋のこのお部屋になんと銀立が増えるか |
| NORMAL   | 560ms   | 0.211 | 9276ms | チャンネル登録者数が九万人を突破いたしました。 この八手の待て、この部屋に八手のこの部屋になんと銀立が増えるもしれま |
| BALANCED | 320ms   | 0.296 | 13035ms | チンネル登録者数が九万人を突破いたしました。 あと零点四万人四千人増えればこの八点の待ってあのこのお部屋になんと銀立が |
| FAST     | 160ms   | 0.515 | 22671ms | チンネル登録者数が九万人を突破いたしました。 あと四万人四千人増えればこの八手の待て、このお部屋に八手屋のこのお部屋に |
| TURBO    | 80ms    | 0.944 | 41577ms | チンネル登録者数が九万人を突破いたしました。 あと零点四万人四千人増えればこの八の待て、このお部屋に八手のこの部屋にな |

## vs Official (NORMAL / 560ms)

Official model from `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`.

| Model | Beam | RTF | Time | Δ | Transcript |
|-------|------|-----|------|---|------------|
| Official onnx-community | 1 | 0.186 | 8200ms | — | チャンネル登録者数が九万人を突破いたしました。 あと零点四万人四千人増えれば待って、このお部屋に八田のこのお部屋になんと銀立が増えるかもしれません銀立てほしい、銀田で欲しい、銀立欲しい銀立てほしい銀立すぎてちょっと酔った |
| Ours | 1 | 0.177 | 7802ms | **−4.8%** | チャンネル登録者数が九万人を突破いたしました。 後零増えればこの八手の待て、こ |
| Ours | 2 | 0.185 | 8129ms | −0.5% | チャンネル登録者数が九万人を突破いたしました。 この八手の待て、この部屋に八手のこの部屋になんと銀立が増えるもしれま |
| Ours | 3 | 0.211 | 9276ms | +13.4% | チャンネル登録者数が九万人を突破いたしました。 この八手の待て、この部屋に八手のこの部屋になんと銀立が増えるもしれま |

Notes:
- Beam search (width > 1) adds 10–40% RTF overhead vs greedy at the same profile, with diminishing returns at lower latency profiles (FAST/TURBO).
- Beam=1 greedy consistently outperforms official across all profiles. Beam>1 is not recommended for NORMAL/lower profiles — quality degrades and RTF increases.
- HIGH profile with beam=2 gives a good RTF/quality tradeoff (0.122 RTF, matching official greedy's WER-equivalent output at lower compute).

## Model

| Property | Value |
|----------|-------|
| ONNX nodes | 1962 (opset 21) |
| ONNX metadata | 2.9 MB |
| Weights (INT4, block_size=32) | 650.1 MB |
| right_context / left_context | 13 / 56 (single encoder, all 5 profiles) |
| Export | `torch.onnx.export(dynamo=True)` |
| Quantization | `MatMulNBitsQuantizer` (bits=4, accuracy_level=4) |
| Runtime | `freeDimensionOverrides` + GPU IO binding + per-profile warmup |



Benchmark RTF — 20.9 s audio
────────────────────────────────────────────────────────────────────
Profile     Beam  RTF      Time     Text
────────────────────────────────────────────────────────────────────
HIGH         greedy  0.101    2111ms  ああ、すいませんとありがとう、心臓爆発もダメじゃん、もうどうしようもないよ、
HIGH         beam=2  0.120    2500ms  ああ、すいません、ありがとう、心臓爆発もダメじゃん、もうどうしようもないよ、
HIGH         beam=3  0.152    3187ms  ああ、すいません、ありがとう、心臓爆発もダメじゃん、もうどうしようもないよ、
NORMAL       greedy  0.159    3332ms  Ah, 心臓爆発もダメじゃん、もうどうしようもないよ、もう爆発してありがとあ
NORMAL       beam=2  0.185    3871ms  ああ、ちょっと心臓爆発もダメじゃん、もうどうしようもないよ、爆発してありがと
NORMAL       beam=3  0.210    4387ms  ありがとう、心臓爆発もダメじゃん、もうどうしようもないよ、爆発してありがと、
BALANCED     greedy  0.255    5326ms  Ah, ダメじゃん、もうどうしようもないよ、もう爆発してありがとありとござい
BALANCED     beam=2  0.267    5581ms  あ、すごいとありがとう、心臓爆発もダメじゃん、もうどうしようもないよ、爆発し
BALANCED     beam=3  0.286    5981ms  あ、すごいとありがとう、心臓爆発もダメじゃん、もうどうしようもないよ、爆発し
FAST         greedy  0.477    9977ms  Ah, どうしようもないよ、もう爆発してありとありがとうございますな
FAST         beam=2  0.485   10137ms  A 心臓爆発もダメじゃん、もうどうしようもないよ、爆発してありがとありがとう
FAST         beam=3  0.508   10607ms  A 心臓爆発もダメじゃん、もうどうしようもないよ、爆発してありがとありがとう
TURBO        greedy  0.910   19022ms  Ah, ダメじゃん、もうどうしようもないよ、もう爆発してありとありとございま
TURBO        beam=2  0.903   18864ms  A 心臓爆発もどうもない、もう爆発してありとありとございます何言っかな
TURBO        beam=3  0.924   19302ms  A 心臓爆発もどうもない、もう爆発してありとありとございます何言っかな
────────────────────────────────────────────────────────────────────
Fastest: HIGH (greedy)  RTF 0.101
