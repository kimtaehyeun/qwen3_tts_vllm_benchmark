"""
MLX-Audio based Qwen3-TTS inference runner for Apple Silicon Macs.

Uses mlx-audio's native Qwen3-TTS implementation (qwen3_tts model type).
No vLLM, no PyTorch, no CUDA required. Runs purely on Apple MLX framework.

Run as: python -m src.benchmark_mlx --config configs/qwen3_tts_mlx.yaml --output-dir <dir>
"""
import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

from src.audio_utils import get_audio_duration
from src.benchmark_manifest import (
    _percentile,
    _write_json,
    _write_jsonl,
    compute_group_summary,
)
from src.config import load_config
from src.manifest_loader import BenchmarkSample, load_manifest, save_validated_manifest
from src.metrics_collector import MetricsCollector
from src.tts_client import compute_steady_state_audio_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-TTS on Apple Silicon via MLX-Audio"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", required=True, help="Output directory for this run")
    parser.add_argument("--manifest", help="Override manifest CSV path")
    parser.add_argument("--audio-root", help="Override audio root directory")
    parser.add_argument("--limit", type=int, help="Limit number of manifest rows")
    parser.add_argument("--batch-size", type=int, help="Batch size (currently 1; mlx-audio generates one at a time)")
    parser.add_argument("--warmup-requests", type=int, help="Number of warmup samples")
    parser.add_argument("--save-audio", dest="save_audio", action="store_true")
    parser.add_argument("--no-save-audio", dest="save_audio", action="store_false")
    parser.add_argument("--max-tokens", type=int, help="Max tokens per segment")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int)
    parser.set_defaults(save_audio=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _expand(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return str(Path(value).expanduser())


def merge_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    config["model"] = dict(config.get("model", {}))
    config["manifest"] = dict(config.get("manifest", {}))
    config["benchmark"] = dict(config.get("benchmark", {}))
    config["mlx_runtime"] = dict(config.get("mlx_runtime", {}))
    config["run"] = dict(config.get("run", {}))

    if args.manifest:
        config["manifest"]["path"] = args.manifest
    if args.audio_root:
        config["manifest"]["audio_root"] = args.audio_root
    if args.limit is not None:
        config["benchmark"]["limit"] = args.limit
    if args.batch_size is not None:
        config["benchmark"]["batch_size"] = args.batch_size
    if args.warmup_requests is not None:
        config["benchmark"]["warmup_requests"] = args.warmup_requests
    if args.save_audio is not None:
        config["benchmark"]["save_audio"] = args.save_audio
    if args.max_tokens is not None:
        config["mlx_runtime"]["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        config["mlx_runtime"]["temperature"] = args.temperature
    if args.top_p is not None:
        config["mlx_runtime"]["top_p"] = args.top_p
    if args.top_k is not None:
        config["mlx_runtime"]["top_k"] = args.top_k
    if args.seed is not None:
        config["run"]["seed"] = args.seed

    for key in ("path", "audio_root"):
        if config["manifest"].get(key):
            config["manifest"][key] = _expand(config["manifest"][key])

    return config


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mlx_model(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Load Qwen3-TTS via mlx-audio. Downloads from HF hub if not cached."""
    try:
        from mlx_audio.tts.utils import load_model  # noqa: PLC0415
    except ImportError as exc:
        print("ERROR: mlx-audio is not installed.")
        print("Run: pip install mlx-audio")
        raise SystemExit(1) from exc

    try:
        import importlib.metadata as _im
        mlx_audio_version = _im.version("mlx-audio")
    except Exception:
        try:
            import mlx_audio  # noqa: F401
            mlx_audio_version = getattr(mlx_audio, "__version__", "unknown")
        except Exception:
            mlx_audio_version = "unknown"

    model_cfg = config.get("model", {})
    model_id: str = model_cfg.get("model_id", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
    # Support local paths with ~ expansion
    model_path = str(Path(model_id).expanduser()) if not model_id.startswith("mlx-community/") and not model_id.startswith("Qwen/") else model_id

    # Read quantization metadata from config.json if it's a local quantized model
    quant_bits = None
    quant_group_size = None
    quant_mode = None
    cuda_aligned = False
    bf16_kept: list = []
    local_config_path = Path(model_path) / "config.json" if Path(model_path).exists() else None
    if local_config_path and local_config_path.exists():
        import json as _json
        with open(local_config_path) as _f:
            _cfg = _json.load(_f)
        _qcfg = _cfg.get("quantization", {})
        quant_bits = _qcfg.get("bits")
        quant_group_size = _qcfg.get("group_size")
        quant_mode = _qcfg.get("mode")
        cuda_aligned = _qcfg.get("cuda_aligned", False)
        bf16_kept = _qcfg.get("bf16_kept", [])

    dtype_label = f"int{quant_bits}" if quant_bits else "bfloat16"
    print(f"Loading model '{model_path}' via mlx-audio {mlx_audio_version} (quant={dtype_label}) ...")
    t0 = time.perf_counter()
    model = load_model(model_path)
    load_time_sec = time.perf_counter() - t0
    print(f"Model loaded in {load_time_sec:.1f}s")

    sample_rate = getattr(model, "sample_rate", 24000)

    load_meta: Dict[str, Any] = {
        "runtime_env": "mlx_audio_direct_mac",
        "runtime_engine": f"mlx_audio.tts ({mlx_audio_version})",
        "model_id": model_id,
        "model_tag": model_id.split("/")[-1],
        "device": "mlx",
        "dtype": dtype_label,
        "attn_implementation": "mlx",
        "load_time_sec": load_time_sec,
        "model_cache_dir": str(Path("~/.cache/huggingface/hub").expanduser()),
        "model_root_dir": model_path if Path(model_path).exists() else None,
        "quantization": f"int{quant_bits}" if quant_bits else "none",
        "quant_method": quant_mode or "none",
        "quant_group_size": quant_group_size,
        "bit_width": dtype_label,
        "llm_bit": quant_bits,
        "mtp_bit": quant_bits,  # code_predictor Linear layers quantized same as LLM
        "llm_quant_method": quant_mode or "none",
        "mtp_quant_method": quant_mode or "none",
        "cuda_aligned": cuda_aligned,
        "bf16_kept": bf16_kept,
        "compute_dtype": dtype_label,
        "base_dtype": dtype_label,
        "sample_rate": sample_rate,
    }
    return model, load_meta


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def generate_one(
    model: Any,
    sample: BenchmarkSample,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, int]:
    """Generate audio for a single sample using mlx-audio's Qwen3-TTS."""
    runtime = config.get("mlx_runtime", {})
    text = sample.target_text or ""
    ref_audio_path = sample.resolved_ref_audio_path
    ref_text = sample.ref_text or ""

    results = list(model.generate(
        text=text,
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        temperature=float(runtime.get("temperature", 0.9)),
        max_tokens=int(runtime.get("max_tokens", 512)),
        top_k=int(runtime.get("top_k", 50)),
        top_p=float(runtime.get("top_p", 1.0)),
        repetition_penalty=float(runtime.get("repetition_penalty", 1.05)),
        stream=False,
        verbose=False,
    ))

    if not results:
        raise RuntimeError("mlx-audio generate() returned no results")

    result = results[0]
    audio_np = ensure_mono_float32(np.array(result.audio))
    sample_rate = int(getattr(result, "sample_rate", 24000))
    return audio_np, sample_rate


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def build_record_base(
    sample: BenchmarkSample,
    run_id: str,
    batch_id: int,
    load_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "request_id": sample.pair_id,
        "row_index": sample.row_index,
        "subset": sample.subset,
        "pair_id": sample.pair_id,
        "concurrency": 1,
        "batch_id": batch_id,
        "success": False,
        "error_type": None,
        "error_message": None,
        "ref_audio_path": sample.ref_audio_path,
        "ref_audio_relpath": sample.ref_audio_relpath,
        "resolved_ref_audio_path": sample.resolved_ref_audio_path,
        "ref_duration_sec": sample.ref_duration_sec,
        "ref_text": sample.ref_text,
        "target_text": sample.target_text,
        "target_text_len": len(sample.target_text or ""),
        "target_audio_path": sample.target_audio_path,
        "target_audio_relpath": sample.target_audio_relpath,
        "target_duration_sec": sample.target_duration_sec,
        "request_start_time_iso": None,
        "request_end_time_iso": None,
        "end_to_end_latency_sec": None,
        "time_to_first_byte_sec": None,
        "time_to_first_audio_chunk_sec": None,
        "output_audio_path": None,
        "output_audio_bytes": None,
        "audio_duration_sec": None,
        "rtf": None,
        "end_to_end_rtf": None,
        "audio_throughput_sec_per_sec": None,
        "end_to_end_audio_throughput_sec_per_sec": None,
        "post_first_audio_chunk_latency_sec": None,
        "steady_state_audio_throughput_sec_per_sec": None,
        "steady_state_streaming_rtf": None,
        "steady_state_metric_status": "request_failed",
        "audio_bytes_per_sec": None,
        "device": load_meta.get("device"),
        "dtype": load_meta.get("dtype"),
        "runtime_env": load_meta.get("runtime_env"),
        "runtime_engine": load_meta.get("runtime_engine"),
        "model_id": load_meta.get("model_id"),
        "model_cache_dir": load_meta.get("model_cache_dir"),
        "quant_method": load_meta.get("quant_method"),
        "bit_width": load_meta.get("bit_width"),
        "llm_bit": load_meta.get("llm_bit"),
        "mtp_bit": load_meta.get("mtp_bit"),
        "llm_quant_method": load_meta.get("llm_quant_method"),
        "mtp_quant_method": load_meta.get("mtp_quant_method"),
        "compute_dtype": load_meta.get("compute_dtype"),
        "attn_implementation": load_meta.get("attn_implementation"),
        "configured_batch_size": 1,
    }


def build_output_audio_path(audio_root: str, sample: BenchmarkSample) -> str:
    subset = sample.subset.replace(os.sep, "_")
    path = os.path.join(audio_root, subset, f"{sample.pair_id}_mlx.wav")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def run_warmup(
    model: Any,
    samples: List[BenchmarkSample],
    config: Dict[str, Any],
    warmup_count: int,
) -> int:
    warmup_samples = [s for s in samples if s.valid][:warmup_count]
    ok = 0
    for sample in tqdm(warmup_samples, desc="warmup", unit="sample"):
        try:
            generate_one(model, sample, config)
            ok += 1
        except Exception as exc:
            print(f"warmup failed: {type(exc).__name__}: {exc}")
    return ok


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_generation(
    model: Any,
    samples: List[BenchmarkSample],
    config: Dict[str, Any],
    output_dir: str,
    run_id: str,
    load_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    benchmark = config.get("benchmark", {})
    save_audio = bool(benchmark.get("save_audio", True))
    audio_root = os.path.join(output_dir, "audio")

    records: List[Dict[str, Any]] = []
    valid_samples = [s for s in samples if s.valid]

    for idx, sample in enumerate(
        tqdm(valid_samples, desc="mlx inference", unit="sample"), start=1
    ):
        start_iso = datetime.now(timezone.utc).isoformat()
        t_start = time.perf_counter()
        try:
            wav, sample_rate = generate_one(model, sample, config)
            elapsed = time.perf_counter() - t_start

            record = build_record_base(sample, run_id, idx, load_meta)
            record["request_start_time_iso"] = start_iso
            record["request_end_time_iso"] = datetime.now(timezone.utc).isoformat()
            record["success"] = True
            record["end_to_end_latency_sec"] = elapsed

            if save_audio:
                output_audio_path = build_output_audio_path(audio_root, sample)
                sf.write(output_audio_path, wav, sample_rate)
                record["output_audio_path"] = output_audio_path
                try:
                    record["output_audio_bytes"] = os.path.getsize(output_audio_path)
                except OSError:
                    record["output_audio_bytes"] = None
                record["audio_duration_sec"] = get_audio_duration(output_audio_path)
            else:
                record["audio_duration_sec"] = float(len(wav)) / float(sample_rate) if sample_rate else None

            dur = record["audio_duration_sec"]
            lat = elapsed
            if dur and dur > 0 and lat > 0:
                record["rtf"] = lat / dur
                record["end_to_end_rtf"] = lat / dur
                record["audio_throughput_sec_per_sec"] = dur / lat
                record["end_to_end_audio_throughput_sec_per_sec"] = dur / lat
                if record.get("output_audio_bytes") is not None:
                    record["audio_bytes_per_sec"] = record["output_audio_bytes"] / lat

            steady = compute_steady_state_audio_metrics(
                True,
                record["end_to_end_latency_sec"],
                record["time_to_first_audio_chunk_sec"],
                record["audio_duration_sec"],
            )
            record.update(steady)
            records.append(record)

        except Exception as exc:
            elapsed_err = time.perf_counter() - t_start
            record = build_record_base(sample, run_id, idx, load_meta)
            record["request_start_time_iso"] = start_iso
            record["request_end_time_iso"] = datetime.now(timezone.utc).isoformat()
            record["error_type"] = type(exc).__name__
            record["error_message"] = str(exc)
            record["end_to_end_latency_sec"] = elapsed_err
            records.append(record)
            print(f"sample {idx} ({sample.pair_id}) failed: {type(exc).__name__}: {exc}")
            error_dir = os.path.join(output_dir, "logs")
            os.makedirs(error_dir, exist_ok=True)
            with open(os.path.join(error_dir, f"{run_id}_sample_{idx:04d}_error.txt"), "w") as fh:
                fh.write(traceback.format_exc())

    return records


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_mlx_summary(
    run_id: str,
    records: List[Dict[str, Any]],
    bench_start: float,
    bench_end: float,
    load_meta: Dict[str, Any],
    memory_summary: Dict[str, Any],
    mlx_env: Dict[str, Any],
    warmup_ok: int,
    run_elapsed_sec: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    group = compute_group_summary(records, bench_start, bench_end, 1, {})
    success_records = [r for r in records if r.get("success")]
    failed_records = [r for r in records if not r.get("success")]

    return {
        "run_id": run_id,
        "total_rows": len(records),
        "ok_rows": len(success_records),
        "error_rows": len(failed_records),
        "success_rate": group.get("success_rate"),
        "latency_mean_sec": group.get("latency_mean_sec"),
        "latency_p50_sec": group.get("latency_p50_sec"),
        "latency_p95_sec": group.get("latency_p95_sec"),
        "rtf_mean": group.get("rtf_mean"),
        "rtf_p95": group.get("rtf_p95"),
        "audio_duration_mean_sec": group.get("audio_duration_mean_sec"),
        "audio_duration_total_sec": group.get("audio_duration_total_sec"),
        "requests_per_sec": group.get("requests_per_sec"),
        "successful_requests_per_sec": group.get("successful_requests_per_sec"),
        "process_ram_peak_mb": memory_summary.get("process_rss_peak_mb"),
        "system_ram_used_peak_mb": memory_summary.get("system_ram_peak_mb"),
        "device": load_meta.get("device"),
        "dtype": load_meta.get("dtype"),
        "attn_implementation": load_meta.get("attn_implementation"),
        "mlx_version": mlx_env.get("mlx_version"),
        "mlx_audio_version": mlx_env.get("mlx_audio_version"),
        "python_version": mlx_env.get("python_version"),
        "platform": mlx_env.get("platform"),
        "model_id": load_meta.get("model_id"),
        "model_load_time_sec": load_meta.get("load_time_sec"),
        "runtime_env": load_meta.get("runtime_env"),
        "runtime_engine": load_meta.get("runtime_engine"),
        "quantization": load_meta.get("quantization"),
        "quant_method": load_meta.get("quant_method"),
        "quant_group_size": load_meta.get("quant_group_size"),
        "bit_width": load_meta.get("bit_width"),
        "llm_bit": load_meta.get("llm_bit"),
        "mtp_bit": load_meta.get("mtp_bit"),
        "llm_quant_method": load_meta.get("llm_quant_method"),
        "mtp_quant_method": load_meta.get("mtp_quant_method"),
        "cuda_aligned": load_meta.get("cuda_aligned", False),
        "bf16_kept": load_meta.get("bf16_kept", []),
        "configured_batch_size": 1,
        "warmup_ok": warmup_ok,
        "run_elapsed_sec": run_elapsed_sec,
        "gen_sec_mean": group.get("latency_mean_sec"),
        "gen_sec_median": group.get("latency_median_sec"),
        "gen_sec_p95": group.get("latency_p95_sec"),
        "audio_sec_mean": group.get("audio_duration_mean_sec"),
    }


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

def describe_mlx_environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    try:
        import mlx.core as mx_core
        info["mlx_version"] = getattr(mx_core, "__version__", "unknown")
    except Exception:
        info["mlx_version"] = None

    try:
        import importlib.metadata as _meta
        info["mlx_audio_version"] = _meta.version("mlx-audio")
    except Exception:
        try:
            import mlx_audio
            info["mlx_audio_version"] = getattr(mlx_audio, "__version__", "unknown")
        except Exception:
            info["mlx_audio_version"] = None

    try:
        import mlx_lm
        info["mlx_lm_version"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        info["mlx_lm_version"] = None

    return info


def build_run_settings(
    config: Dict[str, Any],
    run_id: str,
    output_dir: str,
    samples: List[BenchmarkSample],
    load_meta: Dict[str, Any],
    mlx_env: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_id,
        "run_order": 1,
        "model_id": load_meta.get("model_id"),
        "model_tag": load_meta.get("model_tag"),
        "manifest_path": config.get("manifest", {}).get("path"),
        "manifest_size": len(samples),
        "seed": config.get("run", {}).get("seed"),
        "warmup_count": config.get("benchmark", {}).get("warmup_requests"),
        "sampling_config": {
            "max_tokens": config.get("mlx_runtime", {}).get("max_tokens"),
            "temperature": config.get("mlx_runtime", {}).get("temperature"),
            "top_p": config.get("mlx_runtime", {}).get("top_p"),
            "top_k": config.get("mlx_runtime", {}).get("top_k"),
            "repetition_penalty": config.get("mlx_runtime", {}).get("repetition_penalty"),
            "stream": False,
        },
        "runtime_env": load_meta.get("runtime_env"),
        "runtime_engine": load_meta.get("runtime_engine"),
        "target_device": load_meta.get("device"),
        "base_dtype": load_meta.get("base_dtype"),
        "attn_implementation": load_meta.get("attn_implementation"),
        "quantization": load_meta.get("quantization"),
        "model_cache_dir": load_meta.get("model_cache_dir"),
        "hardware": {
            "platform": mlx_env.get("platform"),
        },
        "software": {
            "python_version": mlx_env.get("python_version"),
            "platform": mlx_env.get("platform"),
            "mlx_version": mlx_env.get("mlx_version"),
            "mlx_audio_version": mlx_env.get("mlx_audio_version"),
        },
        "output_dir": os.path.abspath(output_dir),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = merge_config(load_config(args.config), args)

    output_dir = str(Path(args.output_dir).expanduser())
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    set_seed(config.get("run", {}).get("seed"))

    mlx_env = describe_mlx_environment()
    print(
        f"mlx {mlx_env.get('mlx_version')}  "
        f"mlx-audio {mlx_env.get('mlx_audio_version')}  "
        f"platform={mlx_env.get('platform', '?')[:40]}"
    )

    manifest_cfg = config.get("manifest", {})
    benchmark_cfg = config.get("benchmark", {})

    samples = load_manifest(
        manifest_cfg["path"],
        audio_root=manifest_cfg.get("audio_root"),
        path_prefix_from=manifest_cfg.get("path_prefix_from"),
        path_prefix_to=manifest_cfg.get("path_prefix_to"),
        subset_filter=benchmark_cfg.get("subset"),
        limit=benchmark_cfg.get("limit"),
        seed=config.get("run", {}).get("seed"),
    )
    save_validated_manifest(samples, os.path.join(output_dir, "manifest_validated.csv"))

    run_name = config.get("run", {}).get("name") or "qwen3_tts_mlx"
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    overall_start = time.perf_counter()
    model, load_meta = load_mlx_model(config)

    run_settings = build_run_settings(config, run_id, output_dir, samples, load_meta, mlx_env)
    _write_json(os.path.join(output_dir, "mlx_run_settings.json"), run_settings)
    _write_json(os.path.join(output_dir, f"{run_id}_run_settings.json"), run_settings)

    warmup_count = int(benchmark_cfg.get("warmup_requests", 0) or 0)
    warmup_ok = run_warmup(model, samples, config, warmup_count)

    collector = MetricsCollector(
        sample_interval=float(benchmark_cfg.get("metrics_sample_interval_sec", 0.5) or 0.5),
        process_pid=os.getpid(),
    )
    collector.start()
    bench_start = time.perf_counter()
    records = run_generation(model, samples, config, output_dir, run_id, load_meta)
    bench_end = time.perf_counter()
    collector.stop()

    memory_summary = collector.export_summary()
    _write_json(os.path.join(output_dir, "memory_summary.json"), memory_summary)
    if collector.samples:
        pd.DataFrame(collector.samples).to_csv(
            os.path.join(output_dir, "ram_timeseries.csv"), index=False
        )

    run_elapsed = time.perf_counter() - overall_start
    summary = build_mlx_summary(
        run_id, records, bench_start, bench_end,
        load_meta, memory_summary, mlx_env,
        warmup_ok, run_elapsed, config,
    )

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": os.path.abspath(args.config),
        "config": config,
        "output_dir": os.path.abspath(output_dir),
        "manifest_path": manifest_cfg.get("path"),
        "manifest_row_count": len(samples),
        "cli": " ".join(sys.argv),
    }

    _write_json(os.path.join(output_dir, "metadata.json"), metadata)
    _write_jsonl(os.path.join(output_dir, "requests.jsonl"), records)
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "requests.csv"), index=False)
    _write_json(os.path.join(output_dir, "summary.json"), [summary])
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    pd.DataFrame(records).to_csv(
        os.path.join(output_dir, f"{run_id}_inference_manifest.csv"), index=False
    )
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"{run_id}_run_summary.csv"), index=False
    )
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "all_runs_summary.csv"), index=False
    )

    ok_count = sum(1 for r in records if r.get("success"))
    print(
        f"\nMLX benchmark complete."
        f"  ok={ok_count}/{len(records)}"
        f"  elapsed={run_elapsed:.1f}s"
        f"\nResults saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
