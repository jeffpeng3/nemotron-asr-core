#!/usr/bin/env python3
"""
Apply onnxsim and/or ORT transformer optimizer to a quantized encoder.onnx.
Usage:
  uv run python3 scripts/optimize-onnx.py
  uv run python3 scripts/optimize-onnx.py --with-ort
"""
import argparse
import shutil
import time
from pathlib import Path

import onnx

INPUT_DIR = Path("build/onnx_models-dynamo")
INPUT_ONNX = INPUT_DIR / "encoder.onnx"
BACKUP_ONNX = INPUT_ONNX.with_stem(INPUT_ONNX.stem + ".preopt")
BACKUP_DATA = BACKUP_ONNX.with_suffix(INPUT_ONNX.suffix + ".data")


def apply_onnxsim(model):
    import onnxsim
    print("  Running onnxsim ...")
    t0 = time.time()
    model_sim, ok = onnxsim.simplify(model)
    dt = time.time() - t0
    print(f"  onnxsim took {dt:.1f}s, success={ok}")
    if ok and model_sim:
        return model_sim
    print("  onnxsim returned no result, keeping original")
    return model


def apply_ort_optimizer(model):
    from onnxruntime.transformers import optimizer as ort_opt
    print("  Running ORT transformer optimizer ...")
    t0 = time.time()
    opt = ort_opt.optimize_model(
        model,
        model_type="bert",
        num_heads=0,
        hidden_size=0,
        optimization_options=ort_opt.OptimizationOptions(
            enable_layer_norm_fusion=True,
            enable_gelu_fusion=False,
            enable_attention_fusion=False,
            enable_skip_layer_norm_fusion=False,
            enable_bias_skip_layer_norm_fusion=False,
            enable_embed_layer_norm=False,
            enable_qdq=False,
        ),
        use_gpu=False,
    )
    opt_model = opt.model
    dt = time.time() - t0
    print(f"  ORT optimizer took {dt:.1f}s")
    return opt_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ort", action="store_true", help="Also run ORT transformer optimizer")
    args = parser.parse_args()

    if not INPUT_ONNX.exists():
        print(f"[FAIL] {INPUT_ONNX} not found")
        return

    if not BACKUP_ONNX.exists():
        print(f"  Backing up current model to {BACKUP_ONNX.name} ...")
        shutil.copy2(INPUT_ONNX, BACKUP_ONNX)
        data = INPUT_ONNX.with_suffix(INPUT_ONNX.suffix + ".data")
        if data.exists():
            shutil.copy2(data, BACKUP_DATA)
    else:
        print(f"  Backup already exists at {BACKUP_ONNX.name}")

    print(f"  Loading {INPUT_ONNX.name} ...")
    model = onnx.load(str(INPUT_ONNX), load_external_data=True)
    print(f"  Before: {len(model.graph.node)} nodes")

    model = apply_onnxsim(model)

    if args.with_ort:
        tmp = INPUT_DIR / "encoder.tmp.onnx"
        onnx.save_model(model, str(tmp), save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=tmp.name + ".data")
        model = apply_ort_optimizer(str(tmp))
        for f in [tmp, tmp.with_suffix(tmp.suffix + ".data")]:
            if f.exists():
                f.unlink()

    after_nodes = len(model.graph.node)
    saved = len(onnx.load(str(INPUT_ONNX), load_external_data=False).graph.node) - after_nodes
    print(f"  After:  {after_nodes} nodes ({saved} fewer)")

    print(f"  Saving to {INPUT_ONNX.name} ...")
    data_path = INPUT_ONNX.with_suffix(INPUT_ONNX.suffix + ".data")
    if data_path.exists():
        data_path.unlink()
    INPUT_ONNX.unlink()
    onnx.save_model(
        model,
        str(INPUT_ONNX),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=INPUT_ONNX.name + ".data",
    )
    total_mb = (INPUT_ONNX.stat().st_size + data_path.stat().st_size) / (1024 * 1024)
    print(f"  [OK] Saved ({total_mb:.1f} MB)")


if __name__ == "__main__":
    main()
