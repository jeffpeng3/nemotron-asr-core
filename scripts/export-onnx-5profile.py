#!/usr/bin/env python3
"""
Export Nemotron 3.5 ASR Streaming Multilingual 0.6B to ONNX with 5 encoder profiles.

Produces:
  - encoder_80ms.onnx / encoder_160ms.onnx / encoder_320ms.onnx / encoder_560ms.onnx / encoder_1120ms.onnx
  - decoder.onnx
  - joint.onnx
  - genai_config.json
  - audio_processor_config.json

Usage:
    uv run python scripts/export-onnx-5profile.py
    uv run python scripts/export-onnx-5profile.py --no-quant   (skip INT4)
    uv run python scripts/export-onnx-5profile.py --output-dir build/mymodels
"""

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

# Force CPU-only — export does not need CUDA and VRAM is limited.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nemo.collections.asr as nemo_asr
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b"

N_MELS = 128
SUBSAMPLING = 8
N_LAYERS = 24
D_MODEL = 1024
CONV_CONTEXT = 8
PRE_ENCODE_CACHE = 9
NUM_PROMPTS = 128
DEC_LAYERS = 2
DEC_HID = 640
LEFT_CONTEXT = 56
OPTSET = 20  # legacy dynamo=False 支援到 20；INT4 QOperator 不需 21

PROFILES = {
    "80ms":   {"right_context": 0,  "new_frames": 8},
    "160ms":  {"right_context": 1,  "new_frames": 16},
    "320ms":  {"right_context": 3,  "new_frames": 32},
    "560ms":  {"right_context": 6,  "new_frames": 56},
    "1120ms": {"right_context": 13, "new_frames": 112},
}

ENCODER_INPUT_NAMES = [
    "audio_signal", "length",
    "cache_last_channel", "cache_last_time", "cache_last_channel_len",
    "lang_id",
]
ENCODER_OUTPUT_NAMES = [
    "outputs", "encoded_lengths",
    "cache_last_channel_next", "cache_last_time_next",
    "cache_last_channel_len_next",
]

DECODER_INPUT_NAMES = ["targets", "h_in", "c_in"]
DECODER_OUTPUT_NAMES = ["decoder_output", "h_out", "c_out"]

JOINT_INPUT_NAMES = ["encoder_output", "decoder_output"]
JOINT_OUTPUT_NAMES = ["joint_output"]


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class StreamingEncoderWrapper(nn.Module):
    def __init__(self, encoder, prompt_kernel):
        super().__init__()
        self.encoder = encoder
        self.prompt_kernel = prompt_kernel

    def forward(self, audio_signal, length,
                cache_last_channel, cache_last_time, cache_last_channel_len,
                lang_id):
        audio_signal = audio_signal.transpose(1, 2)
        encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next = \
            self.encoder.forward_for_export(
                audio_signal=audio_signal,
                length=length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )
        encoded = encoded.transpose(1, 2)
        onehot = F.one_hot(lang_id, num_classes=NUM_PROMPTS).to(encoded.dtype)
        prompt = onehot.unsqueeze(1).expand(-1, encoded.shape[1], -1)
        concat = torch.cat([encoded, prompt], dim=-1)
        encoded = self.prompt_kernel(concat).to(encoded.dtype)
        return encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next


class StatefulDecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.decoder._rnnt_export = True

    def forward(self, targets, h_in, c_in):
        g, states = self.decoder.predict(y=targets, state=(h_in, c_in), add_sos=False)
        h_out, c_out = states
        g = g.transpose(1, 2)
        return g, h_out, c_out


class JointWrapper(nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, encoder_output, decoder_output):
        return self.joint.joint(encoder_output, decoder_output)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name):
    print(f"  Loading NeMo model: {model_name}")
    import time
    t0 = time.time()

    # Workaround for NeMo HF cache bug — download the .nemo archive directly
    # instead of using from_pretrained() which chokes on extra cache files.
    if model_name.endswith(".nemo"):
        nemo_path = model_name
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        files = list_repo_files(model_name)
        nemo_files = [f for f in files if f.endswith(".nemo")]
        if not nemo_files:
            raise RuntimeError(f"No .nemo file in {model_name!r}")
        if len(nemo_files) > 1:
            raise RuntimeError(f"Multiple .nemo files in {model_name!r}: {nemo_files}")
        nemo_path = hf_hub_download(repo_id=model_name, filename=nemo_files[0])

    asr_model = nemo_asr.models.ASRModel.restore_from(nemo_path)
    asr_model = asr_model.cpu()
    asr_model.eval()
    t = time.time() - t0
    print(f"  [OK] Loaded in {t:.0f}s")
    return asr_model


# ---------------------------------------------------------------------------
# Encoder export
# ---------------------------------------------------------------------------

def export_encoder(asr_model, profile_name, profile, output_dir):
    new_frames = profile["new_frames"]
    right_context = profile["right_context"]
    enc_in = new_frames + PRE_ENCODE_CACHE
    chunk_encoded = new_frames // SUBSAMPLING

    asr_model.encoder.set_default_att_context_size([LEFT_CONTEXT, right_context])
    asr_model.set_export_config({"cache_support": True})

    wrapper = StreamingEncoderWrapper(asr_model.encoder, asr_model.prompt_kernel)
    wrapper.eval()

    dummy = (
        torch.randn(1, enc_in, N_MELS),
        torch.tensor([enc_in], dtype=torch.int64),
        torch.zeros(1, N_LAYERS, LEFT_CONTEXT, D_MODEL),
        torch.zeros(1, N_LAYERS, D_MODEL, CONV_CONTEXT),
        torch.zeros(1, dtype=torch.int64),
        torch.full((1,), 101, dtype=torch.int64),
    )

    output_path = output_dir / f"encoder_{profile_name}.onnx"
    tmp_path = output_path.with_suffix(".tmp.onnx")

    dyn = {

        "audio_signal": {0: "batch", 1: "time"},
        "length": {0: "batch"},
        "cache_last_channel": {0: "batch", 2: "cache_channel_time"},
        "cache_last_time": {0: "batch", 3: "cache_time_width"},
        "cache_last_channel_len": {0: "batch"},
        "lang_id": {0: "batch"},
        "outputs": {0: "batch", 1: "time"},
        "encoded_lengths": {0: "batch"},
        "cache_last_channel_next": {0: "batch", 2: "cache_channel_time"},
        "cache_last_time_next": {0: "batch", 3: "cache_time_width"},
        "cache_last_channel_len_next": {0: "batch"},
    }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(tmp_path),
            input_names=ENCODER_INPUT_NAMES,
            output_names=ENCODER_OUTPUT_NAMES,
            opset_version=OPTSET,
            dynamic_axes=dyn,
            dynamo=False,
        )

    del wrapper
    gc.collect()

    # Load model with external data, add metadata, re-save with consolidated .data
    model = onnx.load(str(tmp_path), load_external_data=True)
    for k, v in {
        "profile": profile_name,
        "chunk_size_ms": str(int(new_frames * 10)),
        "right_context": str(right_context),
        "left_context": str(LEFT_CONTEXT),
    }.items():
        meta = model.metadata_props.add()
        meta.key = k
        meta.value = v

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
    )
    # Remove individual external-data files left by torch.onnx.export
    for f in output_dir.iterdir():
        if f.name.startswith("encoder.") or f.name.startswith("onnx__") or f.name.startswith("Constant_") or f.name.startswith("prompt_kernel."):
            f.unlink()
    if tmp_path.exists():
        tmp_path.unlink()
    size_mb = output_path.stat().st_size / (1024 * 1024)
    data_path = output_path.with_suffix(output_path.suffix + ".data")
    data_mb = data_path.stat().st_size / (1024 * 1024) if data_path.exists() else 0
    print(f"  [OK] {output_path.name} ({size_mb:.1f} MB + {data_mb:.1f} MB data, FP32)")
    return output_path


# ---------------------------------------------------------------------------
# INT4 quantization
# ---------------------------------------------------------------------------

def quantize_encoder_int4(onnx_path, output_dir):
    print(f"  Quantizing {onnx_path.name} to INT4 ...")
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    # Load FP32 model with external data into memory, clear old external
    # references so save_model_to_file writes fresh data only.
    model = onnx.load(str(onnx_path), load_external_data=True)
    for init in model.graph.initializer:
        if len(init.external_data) > 0 or init.data_location != 0:
            init.ClearField('external_data')
            init.ClearField('data_location')

    q = MatMulNBitsQuantizer(
        model=model,
        block_size=32,
        is_symmetric=True,
        accuracy_level=4,
    )
    q.process()

    # Remove old .data before saving to avoid stale content
    old_data = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if old_data.exists():
        old_data.unlink()
    onnx_path.unlink()

    q.model.save_model_to_file(
        str(onnx_path),
        use_external_data_format=True,
    )

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    data_mb = old_data.stat().st_size / (1024 * 1024) if old_data.exists() else 0
    print(f"  [OK] INT4: {onnx_path.name} ({size_mb:.1f} MB + {data_mb:.1f} MB data)")
    return onnx_path


# ---------------------------------------------------------------------------
# Decoder / Joint export
# ---------------------------------------------------------------------------

def export_decoder(asr_model, output_dir):
    wrapper = StatefulDecoderWrapper(asr_model.decoder)
    wrapper.eval()

    dummy = (
        torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(DEC_LAYERS, 1, DEC_HID),
        torch.zeros(DEC_LAYERS, 1, DEC_HID),
    )

    path = output_dir / "decoder.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        input_names=DECODER_INPUT_NAMES,
        output_names=DECODER_OUTPUT_NAMES,
        opset_version=OPTSET,
        dynamic_axes={
            "targets": {0: "batch"},
            "h_in": {1: "batch"},
            "c_in": {1: "batch"},
            "decoder_output": {0: "batch"},
            "h_out": {1: "batch"},
            "c_out": {1: "batch"},
        },
        dynamo=False,
    )
    # Re-save with external data so JS can load via external_data API.
    data_path = output_dir / "decoder.onnx.data"
    if data_path.exists():
        data_path.unlink()
    model = onnx.load(str(path))
    for t in model.graph.initializer:
        if t.external_data:
            t.external_data.Clear()
            t.data_location = 0
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder.onnx.data",
    )
    total_mb = (path.stat().st_size + data_path.stat().st_size) / (1024 * 1024)
    print(f"  [OK] decoder.onnx + decoder.onnx.data ({total_mb:.1f} MB, FP32)")


def export_joint(asr_model, output_dir):
    wrapper = JointWrapper(asr_model.joint)
    wrapper.eval()

    dummy = (
        torch.randn(1, 1, D_MODEL),
        torch.randn(1, 1, DEC_HID),
    )

    path = output_dir / "joint.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        input_names=JOINT_INPUT_NAMES,
        output_names=JOINT_OUTPUT_NAMES,
        opset_version=OPTSET,
        dynamic_axes={
            "encoder_output": {0: "batch"},
            "decoder_output": {0: "batch"},
            "joint_output": {0: "batch"},
        },
        dynamo=False,
    )
    # Re-save with external data so JS can load via external_data API.
    data_path = output_dir / "joint.onnx.data"
    if data_path.exists():
        data_path.unlink()
    model = onnx.load(str(path))
    for t in model.graph.initializer:
        if t.external_data:
            t.external_data.Clear()
            t.data_location = 0
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="joint.onnx.data",
    )
    total_mb = (path.stat().st_size + data_path.stat().st_size) / (1024 * 1024)
    print(f"  [OK] joint.onnx + joint.onnx.data ({total_mb:.1f} MB, FP32)")
    return path


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_configs(output_dir, asr_model):
    joint = asr_model.joint
    vocab_size = joint.num_classes_with_blank
    blank_id = vocab_size - 1

    genai_config = {
        "model": {
            "type": "nemotron_speech",
            "vocab_size": vocab_size,
            "num_mels": N_MELS,
            "fft_size": 512,
            "hop_length": 160,
            "win_length": 400,
            "preemph": 0.97,
            "log_eps": 5.96046448e-08,
            "subsampling_factor": SUBSAMPLING,
            "left_context": LEFT_CONTEXT,
            "conv_context": CONV_CONTEXT,
            "pre_encode_cache_size": PRE_ENCODE_CACHE,
            "sample_rate": 16000,
            "chunk_samples": 8960,
            "blank_id": blank_id,
            "max_symbols_per_step": 10,
            "encoder": {
                "filename": "encoder.onnx",
                "hidden_size": D_MODEL,
                "num_hidden_layers": N_LAYERS,
                "inputs": {
                    "audio_features": "audio_signal",
                    "input_lengths": "length",
                    "cache_last_channel": "cache_last_channel",
                    "cache_last_time": "cache_last_time",
                    "cache_last_channel_len": "cache_last_channel_len",
                    "lang_id": "lang_id",
                },
                "outputs": {
                    "encoder_outputs": "outputs",
                    "output_lengths": "encoded_lengths",
                    "cache_last_channel_next": "cache_last_channel_next",
                    "cache_last_time_next": "cache_last_time_next",
                    "cache_last_channel_len_next": "cache_last_channel_len_next",
                },
            },
            "decoder": {
                "filename": "decoder.onnx",
                "hidden_size": DEC_HID,
                "num_hidden_layers": DEC_LAYERS,
                "inputs": {
                    "targets": "targets",
                    "lstm_hidden_state": "h_in",
                    "lstm_cell_state": "c_in",
                },
                "outputs": {
                    "outputs": "decoder_output",
                    "lstm_hidden_state": "h_out",
                    "lstm_cell_state": "c_out",
                },
            },
            "joiner": {
                "filename": "joint.onnx",
                "inputs": {
                    "encoder_outputs": "encoder_output",
                    "decoder_outputs": "decoder_output",
                },
                "outputs": {
                    "logits": "joint_output",
                },
            },
        },
    }

    with open(output_dir / "genai_config.json", "w") as f:
        json.dump(genai_config, f, indent=2)
    print("  [OK] genai_config.json")

    audio_config = {
        "model_type": "speech_features",
        "audio_params": {
            "sample_rate": 16000,
            "n_fft": 512,
            "hop_length": 160,
            "n_mels": N_MELS,
            "window_length": 400,
            "window_type": "hann",
            "fmin": 0,
            "fmax": 8000,
            "dither": 1e-05,
            "preemphasis": 0.97,
            "log_zero_guard_type": "add",
            "log_zero_guard_value": 1e-10,
            "normalize": "NA",
            "center": True,
            "mag_power": 2.0,
        },
    }

    with open(output_dir / "audio_processor_config.json", "w") as f:
        json.dump(audio_config, f, indent=2)
    print("  [OK] audio_processor_config.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--output-dir", default="build/onnx_models")
    parser.add_argument("--no-quant", action="store_true", help="Skip INT4 quantization")
    parser.add_argument("--test", action="store_true", help="Only export 80ms profile for testing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(" Nemotron 3.5 ASR → ONNX export (5 profiles)")
    print("=" * 55)

    print(f"\nModel: {args.model}")
    asr_model = load_model(args.model)

    print(f"\nExporting encoder profiles ...")
    for name, profile in PROFILES.items():
        print(f"  [{name}] right_context={profile['right_context']}, "
              f"new_frames={profile['new_frames']}, "
              f"enc_in={profile['new_frames'] + PRE_ENCODE_CACHE}")
        fp32_path = export_encoder(asr_model, name, profile, output_dir)
        if not args.no_quant:
            quantize_encoder_int4(fp32_path, output_dir)
        gc.collect()

    print(f"\nExporting decoder (FP32, shared) ...")
    export_decoder(asr_model, output_dir)

    print(f"\nExporting joint (FP32, shared) ...")
    export_joint(asr_model, output_dir)

    print(f"\nGenerating configs ...")
    generate_configs(output_dir, asr_model)

    print(f"\n{'=' * 55}")
    print(f" Output: {output_dir.resolve()}")
    print(f"{'=' * 55}")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:40s} {size_mb:>8.1f} MB")
        elif f.is_dir():
            print(f"  {f.name}/")

    # Verify dynamic axes
    print(f"\nVerifying dynamic axes ...")
    for f in sorted(output_dir.iterdir()):
        if f.suffix == ".onnx" and f.name.startswith("encoder"):
            model = onnx.load(str(f), load_external_data=False)
            for inp in model.graph.input:
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        pass
            print(f"  {f.name}: OK")


if __name__ == "__main__":
    main()
