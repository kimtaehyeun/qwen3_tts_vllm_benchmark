"""
Local Mac direct PyTorch/HuggingFace runtime for Qwen3-TTS inference.

Supports cuda/mps/cpu device selection with appropriate dtype.
Does NOT use vLLM, vLLM-Omni, bitsandbytes, or FlashAttention2.
"""
import argparse
import gc
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download as _hf_snapshot_download
except ImportError:
    _hf_snapshot_download = None

from src.audio_utils import get_audio_duration
from src.benchmark_manifest import (
    _percentile,
    _write_json,
    _write_jsonl,
    compute_group_summary,
)
from src.config import load_config
from src.local_device import (
    clear_device_cache,
    describe_local_torch_environment,
    get_best_device,
    get_recommended_dtype,
)
from src.manifest_loader import BenchmarkSample, load_manifest, save_validated_manifest
from src.metrics_collector import MetricsCollector
from src.tts_client import compute_steady_state_audio_metrics

_MB = 1024 ** 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-TTS on local Mac with direct PyTorch/HuggingFace runtime"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", required=True, help="Output directory for this run")
    parser.add_argument("--manifest", help="Override manifest CSV path")
    parser.add_argument("--audio-root", help="Override audio root directory")
    parser.add_argument("--limit", type=int, help="Limit number of manifest rows")
    parser.add_argument("--batch-size", type=int, help="Samples per generation call")
    parser.add_argument("--warmup-requests", type=int, help="Number of warmup samples")
    parser.add_argument("--save-audio", dest="save_audio", action="store_true")
    parser.add_argument("--no-save-audio", dest="save_audio", action="store_false")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu / auto")
    parser.add_argument("--dtype", default=None, help="float32 / float16 / bfloat16 / auto")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--do-sample", dest="do_sample", action="store_true")
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--seed", type=int)
    parser.set_defaults(save_audio=None, do_sample=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _expand_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return str(Path(value).expanduser())


def merge_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    config["model"] = dict(config.get("model", {}))
    config["manifest"] = dict(config.get("manifest", {}))
    config["benchmark"] = dict(config.get("benchmark", {}))
    config["torch_runtime"] = dict(config.get("torch_runtime", {}))
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
    if args.device is not None:
        config["torch_runtime"]["device"] = args.device
    if args.dtype is not None:
        config["torch_runtime"]["dtype"] = args.dtype
    if args.max_new_tokens is not None:
        config["torch_runtime"]["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        config["torch_runtime"]["temperature"] = args.temperature
    if args.top_p is not None:
        config["torch_runtime"]["top_p"] = args.top_p
    if args.top_k is not None:
        config["torch_runtime"]["top_k"] = args.top_k
    if args.do_sample is not None:
        config["torch_runtime"]["do_sample"] = args.do_sample
    if args.seed is not None:
        config["run"]["seed"] = args.seed

    # Expand ~ in all path fields
    for key in ("cache_dir",):
        if config["model"].get(key):
            config["model"][key] = _expand_path(config["model"][key])
    for key in ("path", "audio_root"):
        if config["manifest"].get(key):
            config["manifest"][key] = _expand_path(config["manifest"][key])
    for key in ("output_root", "log_root"):
        if config["benchmark"].get(key):
            config["benchmark"][key] = _expand_path(config["benchmark"][key])

    return config


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_model_path(model_id: str, cache_dir: Path) -> Path:
    if _hf_snapshot_download is None:
        raise ImportError(
            "huggingface_hub is required. Run: pip install huggingface_hub"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = _hf_snapshot_download(
        repo_id=model_id, resume_download=True, cache_dir=str(cache_dir)
    )
    root_dir = Path(local_path)
    speech_dir = root_dir / "speech_tokenizer"
    speech_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("preprocessor_config.json", "config.json"):
        src = root_dir / filename
        dst = speech_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return root_dir


def _try_load_and_move_model(
    root_dir: Path,
    dtype: torch.dtype,
    target_device: torch.device,
    attn_implementation: str,
) -> Any:
    """Load Qwen3TTSModel on CPU then move to target_device."""
    from qwen_tts import Qwen3TTSModel  # noqa: PLC0415

    wrapper = Qwen3TTSModel.from_pretrained(
        str(root_dir),
        device_map="cpu",
        dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    wrapper.model = wrapper.model.to(target_device)
    wrapper.model.eval()
    return wrapper


def load_local_model(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Load Qwen3-TTS model with Mac-appropriate device and dtype.

    For MPS: tries float16 first, falls back to float32 on failure.
    """
    try:
        import qwen_tts  # noqa: F401
    except ImportError as exc:
        print("ERROR: qwen-tts is not installed.")
        print("Run: pip install qwen-tts")
        print("Or: bash scripts/setup_mac_local.sh")
        raise SystemExit(1) from exc

    model_cfg = config.get("model", {})
    runtime = config.get("torch_runtime", {})

    model_id: str = model_cfg.get("model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    cache_dir = Path(
        model_cfg.get("cache_dir", "~/Workspace/models/qwen3_tts_vllm_benchmark/hf_cache")
    ).expanduser()
    requested_device: str = runtime.get("device", "auto") or "auto"
    requested_dtype: str = runtime.get("dtype", "auto") or "auto"
    attn_implementation: str = runtime.get("attn_implementation", "sdpa") or "sdpa"

    target_device = get_best_device(requested_device)
    target_dtype = get_recommended_dtype(target_device, requested_dtype)

    print(f"Target device: {target_device}  dtype: {target_dtype}")
    print(f"Resolving model '{model_id}' from cache_dir={cache_dir}")

    load_started = time.perf_counter()
    root_dir = _resolve_model_path(model_id, cache_dir)

    tts_wrapper = None
    actual_dtype = target_dtype

    if target_device.type == "mps" and target_dtype == torch.float16:
        try:
            tts_wrapper = _try_load_and_move_model(root_dir, torch.float16, target_device, attn_implementation)
        except Exception as exc:
            print(f"float16 on MPS failed ({type(exc).__name__}: {exc}), retrying with float32")
            gc.collect()
            tts_wrapper = None
            actual_dtype = torch.float32

    if tts_wrapper is None:
        tts_wrapper = _try_load_and_move_model(root_dir, actual_dtype, target_device, attn_implementation)

    try:
        actual_device = next(tts_wrapper.model.parameters()).device
    except StopIteration:
        actual_device = target_device

    try:
        tts_wrapper.device = actual_device
    except Exception:
        pass

    load_time_sec = time.perf_counter() - load_started

    load_meta: Dict[str, Any] = {
        "runtime_env": "torch_hf_direct_mac",
        "runtime_engine": "qwen_tts.Qwen3TTSModel",
        "model_id": model_id,
        "model_tag": model_id.split("/")[-1],
        "device": str(actual_device),
        "dtype": str(actual_dtype).replace("torch.", ""),
        "attn_implementation": attn_implementation,
        "load_time_sec": load_time_sec,
        "model_cache_dir": str(cache_dir),
        "model_root_dir": str(root_dir),
        "quantization": "none",
        "quant_method": "none",
        "bit_width": str(actual_dtype).replace("torch.", ""),
        "llm_bit": None,
        "mtp_bit": None,
        "llm_quant_method": "none",
        "mtp_quant_method": "none",
        "compute_dtype": str(actual_dtype).replace("torch.", ""),
        "base_dtype": str(actual_dtype).replace("torch.", ""),
    }

    print(f"Model loaded in {load_time_sec:.1f}s  device={actual_device}  dtype={actual_dtype}")
    return tts_wrapper, load_meta


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def guess_language(text: Optional[str]) -> str:
    value = text or ""
    for ch in value:
        if "가" <= ch <= "힣":
            return "Korean"
    return "Auto"


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def unpack_generation_result(result: Any) -> Tuple[List[np.ndarray], int]:
    if isinstance(result, (tuple, list)) and len(result) == 2:
        audio, sample_rate = result
    else:
        audio = result
        sample_rate = 24000

    if isinstance(audio, list):
        raw_wavs = audio
    elif torch.is_tensor(audio):
        arr = audio.detach().cpu().numpy()
        if arr.ndim == 1:
            raw_wavs = [arr]
        else:
            raw_wavs = [arr[i] for i in range(arr.shape[0])]
    else:
        arr = np.array(audio)
        if arr.ndim == 1:
            raw_wavs = [arr]
        else:
            raw_wavs = [arr[i] for i in range(arr.shape[0])]

    wavs = [ensure_mono_float32(np.array(wav).squeeze()) for wav in raw_wavs]
    if not wavs:
        raise RuntimeError("Generated audio is empty")
    return wavs, int(sample_rate)


def generate_batch(
    tts_wrapper: Any,
    samples: List[BenchmarkSample],
    config: Dict[str, Any],
) -> Tuple[List[np.ndarray], int]:
    runtime = config.get("torch_runtime", {})
    texts = [s.target_text or "" for s in samples]
    languages = [guess_language(s.target_text) for s in samples]
    ref_audios = [s.resolved_ref_audio_path for s in samples]
    ref_texts = [s.ref_text or "" for s in samples]

    with torch.inference_mode():
        result = tts_wrapper.generate_voice_clone(
            text=texts,
            language=languages,
            ref_audio=ref_audios,
            ref_text=ref_texts,
            non_streaming_mode=bool(runtime.get("non_streaming_mode", True)),
            max_new_tokens=int(runtime.get("max_new_tokens", 512)),
            do_sample=bool(runtime.get("do_sample", True)),
            top_k=int(runtime.get("top_k", 50)),
            top_p=float(runtime.get("top_p", 0.95)),
            temperature=float(runtime.get("temperature", 1.0)),
            use_cache=bool(runtime.get("use_cache", True)),
        )
    return unpack_generation_result(result)


# ---------------------------------------------------------------------------
# Record building
# ---------------------------------------------------------------------------

def build_record_base(
    sample: BenchmarkSample,
    run_id: str,
    batch_id: int,
    batch_size: int,
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
        "configured_batch_size": batch_size,
    }


def build_output_audio_path(audio_root: str, sample: BenchmarkSample, suffix: str) -> str:
    subset = sample.subset.replace(os.sep, "_")
    path = os.path.join(audio_root, subset, f"{sample.pair_id}_{suffix}.wav")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def run_warmup(
    tts_wrapper: Any,
    samples: List[BenchmarkSample],
    config: Dict[str, Any],
    warmup_count: int,
    batch_size: int,
) -> int:
    warmup_samples = [s for s in samples if s.valid][:warmup_count]
    ok = 0
    if not warmup_samples:
        return ok
    batches = [
        warmup_samples[i: i + batch_size]
        for i in range(0, len(warmup_samples), batch_size)
    ]
    for batch in tqdm(batches, desc="warmup", unit="batch"):
        try:
            generate_batch(tts_wrapper, batch, config)
            ok += len(batch)
        except Exception as exc:
            print(f"warmup failed: {type(exc).__name__}: {exc}")
    clear_device_cache(get_best_device(config.get("torch_runtime", {}).get("device", "auto")))
    return ok


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_generation(
    tts_wrapper: Any,
    samples: List[BenchmarkSample],
    config: Dict[str, Any],
    output_dir: str,
    run_id: str,
    load_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    benchmark = config.get("benchmark", {})
    batch_size = int(benchmark.get("batch_size", 1) or 1)
    save_audio = bool(benchmark.get("save_audio", True))
    audio_root = os.path.join(output_dir, "audio")

    records: List[Dict[str, Any]] = []
    valid_samples = [s for s in samples if s.valid]
    batches = [
        valid_samples[i: i + batch_size]
        for i in range(0, len(valid_samples), batch_size)
    ]

    for batch_id, batch in enumerate(
        tqdm(batches, desc="local mac inference", unit="batch"), start=1
    ):
        batch_start_iso = datetime.now(timezone.utc).isoformat()
        batch_start = time.perf_counter()
        try:
            wavs, sample_rate = generate_batch(tts_wrapper, batch, config)
            elapsed = time.perf_counter() - batch_start

            if len(wavs) != len(batch):
                raise RuntimeError(
                    f"Output count mismatch: outputs={len(wavs)}, inputs={len(batch)}"
                )

            amortized_latency = elapsed / len(batch) if batch else None

            for sample, wav in zip(batch, wavs):
                record = build_record_base(sample, run_id, batch_id, batch_size, load_meta)
                record["request_start_time_iso"] = batch_start_iso
                record["request_end_time_iso"] = datetime.now(timezone.utc).isoformat()
                record["success"] = True
                record["end_to_end_latency_sec"] = amortized_latency
                record["batch_end_to_end_latency_sec"] = elapsed
                record["batch_size_actual"] = len(batch)

                if save_audio:
                    output_audio_path = build_output_audio_path(audio_root, sample, "local_mac")
                    sf.write(output_audio_path, wav, sample_rate)
                    record["output_audio_path"] = output_audio_path
                    try:
                        record["output_audio_bytes"] = os.path.getsize(output_audio_path)
                    except OSError:
                        record["output_audio_bytes"] = None
                    record["audio_duration_sec"] = get_audio_duration(output_audio_path)
                else:
                    record["audio_duration_sec"] = (
                        float(len(wav)) / float(sample_rate) if sample_rate else None
                    )

                dur = record["audio_duration_sec"]
                lat = amortized_latency
                if dur and dur > 0 and lat and lat > 0:
                    record["rtf"] = lat / dur
                    record["end_to_end_rtf"] = record["rtf"]
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
            elapsed_err = time.perf_counter() - batch_start
            for sample in batch:
                record = build_record_base(sample, run_id, batch_id, batch_size, load_meta)
                record["request_start_time_iso"] = batch_start_iso
                record["request_end_time_iso"] = datetime.now(timezone.utc).isoformat()
                record["error_type"] = type(exc).__name__
                record["error_message"] = str(exc)
                record["end_to_end_latency_sec"] = elapsed_err
                records.append(record)
            print(f"batch {batch_id} failed: {type(exc).__name__}: {exc}")
            error_dir = os.path.join(output_dir, "logs")
            os.makedirs(error_dir, exist_ok=True)
            error_path = os.path.join(
                error_dir, f"{run_id}_batch_{batch_id:04d}_error.txt"
            )
            with open(error_path, "w", encoding="utf-8") as fh:
                fh.write(traceback.format_exc())

    return records


# ---------------------------------------------------------------------------
# Summary building
# ---------------------------------------------------------------------------

def build_mac_summary(
    run_id: str,
    records: List[Dict[str, Any]],
    bench_start: float,
    bench_end: float,
    load_meta: Dict[str, Any],
    memory_summary: Dict[str, Any],
    torch_env: Dict[str, Any],
    warmup_ok: int,
    run_elapsed_sec: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute summary dict with mac-specific fields, reusing compute_group_summary."""
    group = compute_group_summary(
        records,
        bench_start,
        bench_end,
        1,
        {"gpu_memory_summary": memory_summary},
    )
    batch_size = int(config.get("benchmark", {}).get("batch_size", 1) or 1)
    success_records = [r for r in records if r.get("success")]
    failed_records = [r for r in records if not r.get("success")]

    summary: Dict[str, Any] = {
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
        "mps_available": torch_env.get("mps_available"),
        "mps_built": torch_env.get("mps_built"),
        "torch_version": torch_env.get("torch_version"),
        "python_version": torch_env.get("python_version"),
        "platform": torch_env.get("platform"),
        "model_id": load_meta.get("model_id"),
        "model_load_time_sec": load_meta.get("load_time_sec"),
        "runtime_env": load_meta.get("runtime_env"),
        "runtime_engine": load_meta.get("runtime_engine"),
        "quant_method": load_meta.get("quant_method"),
        "bit_width": load_meta.get("bit_width"),
        "configured_batch_size": batch_size,
        "warmup_ok": warmup_ok,
        "run_elapsed_sec": run_elapsed_sec,
        "gen_sec_mean": group.get("latency_mean_sec"),
        "gen_sec_median": group.get("latency_median_sec"),
        "gen_sec_p95": group.get("latency_p95_sec"),
        "audio_sec_mean": group.get("audio_duration_mean_sec"),
    }
    return summary


# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

def build_run_settings(
    config: Dict[str, Any],
    run_id: str,
    output_dir: str,
    samples: List[BenchmarkSample],
    load_meta: Dict[str, Any],
    torch_env: Dict[str, Any],
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
            "max_new_tokens": config.get("torch_runtime", {}).get("max_new_tokens"),
            "temperature": config.get("torch_runtime", {}).get("temperature"),
            "top_p": config.get("torch_runtime", {}).get("top_p"),
            "top_k": config.get("torch_runtime", {}).get("top_k"),
            "do_sample": config.get("torch_runtime", {}).get("do_sample"),
            "non_streaming_mode": config.get("torch_runtime", {}).get("non_streaming_mode"),
            "use_cache": config.get("torch_runtime", {}).get("use_cache"),
            "batch_size": config.get("benchmark", {}).get("batch_size"),
        },
        "runtime_env": load_meta.get("runtime_env"),
        "runtime_engine": load_meta.get("runtime_engine"),
        "target_device": load_meta.get("device"),
        "base_dtype": load_meta.get("base_dtype"),
        "attn_implementation": load_meta.get("attn_implementation"),
        "quantization": load_meta.get("quantization"),
        "model_cache_dir": load_meta.get("model_cache_dir"),
        "hardware": {
            "mps_available": torch_env.get("mps_available"),
            "mps_built": torch_env.get("mps_built"),
            "cuda_available": torch_env.get("cuda_available"),
            "cuda_device_count": torch_env.get("cuda_device_count"),
            "cuda_device_names": torch_env.get("cuda_device_names"),
        },
        "software": {
            "python_version": torch_env.get("python_version"),
            "platform": torch_env.get("platform"),
            "torch_version": torch_env.get("torch_version"),
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
    torch.set_grad_enabled(False)

    torch_env = describe_local_torch_environment()
    print(
        f"torch {torch_env['torch_version']}  "
        f"mps_built={torch_env['mps_built']}  "
        f"mps_available={torch_env['mps_available']}"
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

    run_name = config.get("run", {}).get("name") or config.get("run", {}).get("run_name", "qwen3_tts_local_mac")
    run_id = f"{run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    overall_start = time.perf_counter()
    tts_wrapper, load_meta = load_local_model(config)

    run_settings = build_run_settings(config, run_id, output_dir, samples, load_meta, torch_env)
    _write_json(os.path.join(output_dir, "local_mac_run_settings.json"), run_settings)
    _write_json(os.path.join(output_dir, f"{run_id}_run_settings.json"), run_settings)

    batch_size = int(benchmark_cfg.get("batch_size", 1) or 1)
    warmup_count = int(benchmark_cfg.get("warmup_requests", 0) or 0)
    warmup_ok = run_warmup(tts_wrapper, samples, config, warmup_count, batch_size)

    collector = MetricsCollector(
        sample_interval=float(benchmark_cfg.get("metrics_sample_interval_sec", 0.5) or 0.5),
        process_pid=os.getpid(),
    )
    collector.start()
    bench_start = time.perf_counter()
    records = run_generation(tts_wrapper, samples, config, output_dir, run_id, load_meta)
    bench_end = time.perf_counter()
    collector.stop()

    memory_summary = collector.export_summary()
    _write_json(os.path.join(output_dir, "memory_summary.json"), memory_summary)
    if collector.samples:
        pd.DataFrame(collector.samples).to_csv(
            os.path.join(output_dir, "ram_timeseries.csv"), index=False
        )

    run_elapsed = time.perf_counter() - overall_start
    summary = build_mac_summary(
        run_id, records, bench_start, bench_end,
        load_meta, memory_summary, torch_env,
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
    pd.DataFrame(records).to_csv(
        os.path.join(output_dir, f"{run_id}_inference_manifest.partial.csv"), index=False
    )
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"{run_id}_run_summary.csv"), index=False
    )
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "all_runs_summary.csv"), index=False
    )

    del tts_wrapper
    clear_device_cache(get_best_device(config.get("torch_runtime", {}).get("device", "auto")))

    ok_count = sum(1 for r in records if r.get("success"))
    print(
        f"\nLocal Mac benchmark complete."
        f"  ok={ok_count}/{len(records)}"
        f"  elapsed={run_elapsed:.1f}s"
        f"\nResults saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
