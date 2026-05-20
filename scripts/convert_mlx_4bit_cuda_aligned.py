#!/usr/bin/env python3
"""
Convert Qwen3-TTS to MLX int4 with the same quantization coverage as CUDA BNB NF4.

CUDA uses keep_lm_head_in_base_dtype=True, which skips:
  - talker.text_projection.linear_fc1
  - talker.text_projection.linear_fc2
  - talker.codec_head

Result: 247 quantized layers (same as CUDA), 3 bf16-kept layers.
"""

import argparse
import copy
from pathlib import Path

import mlx.nn as nn
from mlx_audio.convert import (
    copy_model_files,
    detect_model_domain,
    get_model_class,
    get_model_path,
    get_model_type,
    load_config,
    load_weights,
)
from mlx_lm.utils import quantize_model, save_config, save_model

# Kept in bf16 — mirrors CUDA keep_lm_head_in_base_dtype=True
CUDA_ALIGNED_SKIP = [
    "text_projection",  # talker.text_projection.linear_fc1 / linear_fc2
    "codec_head",       # talker.codec_head
    # Already skipped by model_quant_predicate, listed for explicitness:
    "codec_embedding",
    "text_embedding",
    "speech_tokenizer",
    "speaker_encoder",
]


def cuda_aligned_predicate(path: str, module) -> bool:
    if not hasattr(module, "to_quantized"):
        return False
    if not hasattr(module, "weight"):
        return False
    if module.weight.shape[-1] % 64 != 0:
        return False
    return not any(pat in path for pat in CUDA_ALIGNED_SKIP)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model",
        default="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        help="HF repo ID or local path to source bf16 model",
    )
    parser.add_argument(
        "--output-dir",
        default="~/Workspace/models/qwen3_tts_vllm_benchmark/mlx_4bit_cuda_aligned/Qwen3-TTS-12Hz-1.7B-Base-4bit-cuda-aligned",
        help="Directory to save the converted model",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    output_path = Path(args.output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Resolving model path: {args.hf_model}")
    model_path = get_model_path(args.hf_model)
    config = load_config(model_path)

    print("[2/5] Loading model (bare weights, no TTS pipeline wrappers)")
    domain = detect_model_domain(config, model_path)
    model_type = get_model_type(config, model_path, domain)
    model_class = get_model_class(model_type, domain)
    model_config = (
        model_class.ModelConfig.from_dict(config)
        if hasattr(model_class, "ModelConfig")
        else config
    )
    if hasattr(model_config, "model_path"):
        model_config.model_path = model_path

    weights = load_weights(model_path)
    model = model_class.Model(model_config)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))

    print(f"[3/5] Applying int4 quantization (group_size={args.group_size}, bits={args.bits})")
    print("      Kept in bf16 (CUDA-aligned): text_projection x2, codec_head x1")

    quantized_layers, skipped_layers = [], []

    def tracking_predicate(path, module):
        result = cuda_aligned_predicate(path, module)
        if hasattr(module, "to_quantized") and hasattr(module, "weight"):
            (quantized_layers if result else skipped_layers).append(path)
        return result

    quantized_weights, quantized_config = quantize_model(
        model,
        copy.deepcopy(config),
        args.group_size,
        args.bits,
        mode="affine",
        quant_predicate=tracking_predicate,
    )

    bf16_kept = [p for p in skipped_layers if any(x in p for x in ("text_projection", "codec_head"))]
    print(f"      Quantized : {len(quantized_layers)} layers")
    print(f"      Skipped   : {len(skipped_layers)} layers")
    print(f"      bf16 kept : {bf16_kept}")

    print("[4/5] Copying supporting files (tokenizer, speech_tokenizer/, config templates)")
    copy_model_files(model_path, output_path)

    print("[5/5] Saving quantized weights and updated config")
    quant_meta = {
        "group_size": args.group_size,
        "bits": args.bits,
        "mode": "affine",
        "cuda_aligned": True,
        "bf16_kept": bf16_kept,
    }
    quantized_config["quantization"] = quant_meta
    quantized_config["quantization_config"] = quant_meta
    quantized_config["model_type"] = model_type

    save_model(output_path, model, donate_model=True)
    save_config(quantized_config, config_path=output_path / "config.json")

    print(f"\nConversion complete.")
    print(f"  Output      : {output_path}")
    print(f"  Quantized   : {len(quantized_layers)} layers  (matches CUDA n_quantized=247)")
    print(f"  bf16 kept   : {len(bf16_kept)} layers  (matches CUDA n_skipped=3)")


if __name__ == "__main__":
    main()
