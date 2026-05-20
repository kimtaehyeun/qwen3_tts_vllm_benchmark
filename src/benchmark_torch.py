import argparse
import gc
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from huggingface_hub import snapshot_download
from tqdm import tqdm

try:
    import bitsandbytes as bnb
except Exception:
    bnb = None

try:
    import transformers
except Exception:
    transformers = None

from src.audio_utils import get_audio_duration
from src.benchmark_manifest import _percentile, _write_json, _write_jsonl, compute_group_summary
from src.config import load_config
from src.env_info import gather_environment_info
from src.manifest_loader import BenchmarkSample, load_manifest, save_validated_manifest
from src.metrics_collector import MetricsCollector
from src.tts_client import compute_steady_state_audio_metrics


_MB = 1024 ** 2
LM_BACKBONE_PREFIXES = ('talker.model',)
LM_HEAD_PREFIXES = ('talker.codec_head', 'talker.text_projection')
MTP_PREFIXES = ('talker.code_predictor',)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Qwen3-TTS benchmark with direct PyTorch/Hugging Face runtime')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--quantization', choices=['none', 'bnb4'], default='none')
    parser.add_argument('--manifest')
    parser.add_argument('--audio-root')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--subset', action='append')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--warmup-requests', type=int)
    parser.add_argument('--timeout-sec', type=int)
    parser.add_argument('--save-audio', dest='save_audio', action='store_true')
    parser.add_argument('--no-save-audio', dest='save_audio', action='store_false')
    parser.add_argument('--attn-implementation')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--do-sample', dest='do_sample', action='store_true')
    parser.add_argument('--no-do-sample', dest='do_sample', action='store_false')
    parser.add_argument('--subtalker-dosample', action='store_true')
    parser.add_argument('--no-use-cache', dest='use_cache', action='store_false')
    parser.add_argument('--metrics-sample-interval-sec', type=float)
    parser.add_argument('--seed', type=int)
    parser.set_defaults(save_audio=None, do_sample=True, use_cache=True)
    return parser.parse_args()


def merge_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    config['manifest'] = dict(config.get('manifest', {}))
    config['benchmark'] = dict(config.get('benchmark', {}))
    config['server'] = dict(config.get('server', {}))
    config['tts'] = dict(config.get('tts', {}))
    config['torch_runtime'] = dict(config.get('torch_runtime', {}))

    if args.manifest:
        config['manifest']['path'] = args.manifest
    if args.audio_root:
        config['manifest']['audio_root'] = args.audio_root
    if args.limit is not None:
        config['benchmark']['limit'] = args.limit
    if args.subset is not None:
        config['benchmark']['subset'] = args.subset
    if args.batch_size is not None:
        config['benchmark']['batch_size'] = args.batch_size
    if args.warmup_requests is not None:
        config['benchmark']['warmup_requests'] = args.warmup_requests
    if args.save_audio is not None:
        config['benchmark']['save_audio'] = args.save_audio
    if args.metrics_sample_interval_sec is not None:
        config['benchmark']['metrics_sample_interval_sec'] = args.metrics_sample_interval_sec
    if args.seed is not None:
        config['run'] = dict(config.get('run', {}))
        config['run']['seed'] = args.seed

    config['torch_runtime']['quantization'] = args.quantization
    if args.attn_implementation is not None:
        config['torch_runtime']['attn_implementation'] = args.attn_implementation
    config['torch_runtime']['max_new_tokens'] = args.max_new_tokens
    config['torch_runtime']['temperature'] = args.temperature
    config['torch_runtime']['top_p'] = args.top_p
    config['torch_runtime']['top_k'] = args.top_k
    config['torch_runtime']['do_sample'] = args.do_sample
    config['torch_runtime']['subtalker_dosample'] = args.subtalker_dosample
    config['torch_runtime']['use_cache'] = args.use_cache
    return config


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


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def dtype_to_str(dtype: Any) -> str:
    if dtype is None:
        return 'unknown'
    return str(dtype).replace('torch.', '')


def has_prefix(name: str, prefixes: Tuple[str, ...]) -> bool:
    for prefix in prefixes:
        if name.startswith(prefix):
            return True
    return False


def detect_module_group(name: str) -> str:
    if has_prefix(name, LM_HEAD_PREFIXES):
        return 'lm_head'
    if has_prefix(name, LM_BACKBONE_PREFIXES):
        return 'lm_backbone'
    if has_prefix(name, MTP_PREFIXES):
        return 'mtp'
    return 'other'


def make_int4_layer(linear: nn.Linear, compute_dtype: torch.dtype) -> nn.Module:
    if bnb is None:
        raise ImportError('bitsandbytes is required for torch bnb4 quantization')
    new_mod = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=(linear.bias is not None),
        compute_dtype=compute_dtype,
        compress_statistics=True,
        quant_type='nf4',
    )
    new_mod.weight = bnb.nn.Params4bit(
        linear.weight.detach().cpu().clone(),
        requires_grad=False,
        quant_type='nf4',
        compress_statistics=True,
    )
    if linear.bias is not None:
        new_mod.bias = nn.Parameter(linear.bias.detach().cpu().clone(), requires_grad=False)
    return new_mod


def get_new_weight_dtype_repr(module: nn.Module) -> str:
    weight = getattr(module, 'weight', None)
    if weight is None:
        return 'unknown'
    dtype = getattr(weight, 'dtype', None)
    return dtype_to_str(dtype)


def apply_selective_bnb4_quantization(
    model: nn.Module,
    llm_bit: Optional[int],
    mtp_bit: Optional[int],
    compute_dtype: torch.dtype,
    keep_lm_head_in_base_dtype: bool = True,
) -> Dict[str, Any]:
    inventory: List[Dict[str, Any]] = []
    n_quantized = 0
    n_skipped = 0
    named_modules = list(model.named_modules())

    for name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        group = detect_module_group(name)
        target_bit = None
        if group == 'lm_backbone':
            target_bit = llm_bit
        elif group == 'mtp':
            target_bit = mtp_bit

        row: Dict[str, Any] = {
            'module_name': name,
            'module_group': group,
            'original_class': type(module).__name__,
            'new_class': type(module).__name__,
            'in_features': int(getattr(module, 'in_features', -1)),
            'out_features': int(getattr(module, 'out_features', -1)),
            'assigned_bit_width': target_bit if target_bit is not None else 'fp',
            'quantized': 0,
            'skipped_reason': '',
            'assigned_quant_method': 'none',
            'original_weight_dtype': get_new_weight_dtype_repr(module),
            'new_weight_dtype': get_new_weight_dtype_repr(module),
        }

        if group == 'lm_head' and keep_lm_head_in_base_dtype:
            row['skipped_reason'] = 'kept_in_base_dtype'
            inventory.append(row)
            n_skipped += 1
            continue
        if target_bit is None:
            row['skipped_reason'] = 'no_target_bit'
            inventory.append(row)
            n_skipped += 1
            continue
        if target_bit != 4:
            raise ValueError(f'Only bnb4 is supported by this runner, got bit={target_bit}')

        parent = model
        parts = name.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        new_layer = make_int4_layer(module, compute_dtype)
        setattr(parent, parts[-1], new_layer)

        row['quantized'] = 1
        row['new_class'] = type(new_layer).__name__
        row['new_weight_dtype'] = get_new_weight_dtype_repr(new_layer)
        row['assigned_quant_method'] = 'bnb_nf4'
        inventory.append(row)
        n_quantized += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary: Dict[str, Any] = {
        'n_quantized': n_quantized,
        'n_skipped': n_skipped,
        'n_linear_total': len(inventory),
        'n_quantized_lm_backbone': sum(1 for row in inventory if row['module_group'] == 'lm_backbone' and row['quantized']),
        'n_quantized_mtp': sum(1 for row in inventory if row['module_group'] == 'mtp' and row['quantized']),
        'n_quantized_lm_head': sum(1 for row in inventory if row['module_group'] == 'lm_head' and row['quantized']),
        'inventory': inventory,
    }
    return summary


def directory_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return round(total / _MB, 2)


def get_model_memory_footprint_mb(model: nn.Module) -> Optional[float]:
    try:
        if hasattr(model, 'get_memory_footprint'):
            value = model.get_memory_footprint()
            if isinstance(value, (int, float)):
                return round(float(value) / _MB, 2)
    except Exception:
        pass

    total = 0
    seen_ptrs = set()
    try:
        tensors = list(model.parameters()) + list(model.buffers())
        for tensor in tensors:
            storage = tensor.untyped_storage()
            ptr = storage.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            total += storage.nbytes()
        return round(total / _MB, 2)
    except Exception:
        return None


def resolve_local_model_path(model_id: str, cache_dir: Path) -> Path:
    local_path = snapshot_download(repo_id=model_id, resume_download=True, cache_dir=str(cache_dir))
    root_dir = Path(local_path)
    speech_dir = root_dir / 'speech_tokenizer'
    speech_dir.mkdir(parents=True, exist_ok=True)
    for filename in ['preprocessor_config.json', 'config.json']:
        src = root_dir / filename
        dst = speech_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return root_dir


def resolve_hf_hub_cache_dir(output_dir: str) -> Path:
    hub_cache = os.environ.get('HUGGINGFACE_HUB_CACHE') or os.environ.get('HF_HUB_CACHE')
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        return Path(hf_home).expanduser() / 'hub'
    return Path.home() / '.cache' / 'huggingface' / 'hub'


def describe_quantization(quantization: str) -> Dict[str, Any]:
    if quantization == 'bnb4':
        return {
            'quant_method': 'selective_mixed_precision',
            'bit_width': 'llm4_mtp4',
            'llm_bit': 4,
            'mtp_bit': 4,
            'llm_quant_method': 'bnb_nf4',
            'mtp_quant_method': 'bnb_nf4',
        }
    return {
        'quant_method': 'none',
        'bit_width': 'bf16',
        'llm_bit': None,
        'mtp_bit': None,
        'llm_quant_method': 'none',
        'mtp_quant_method': 'none',
    }


def resolve_attn_implementation(value: Optional[str]) -> str:
    if value and value != 'auto':
        return str(value)
    if importlib.util.find_spec('flash_attn') is not None:
        return 'flash_attention_2'
    return 'sdpa'


def load_torch_model(config: Dict[str, Any], output_dir: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:
        raise ImportError('qwen-tts is not installed. Run: pip install qwen-tts==0.1.1') from exc

    runtime = config.get('torch_runtime', {})
    server = config.get('server', {})
    model_id = server.get('model_id', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base')
    quantization = runtime.get('quantization', 'none')
    attn_implementation = resolve_attn_implementation(runtime.get('attn_implementation', 'auto'))
    if attn_implementation == 'flash_attention_2' and runtime.get('require_flash_attention', False):
        if importlib.util.find_spec('flash_attn') is None:
            raise ImportError(
                'flash_attn is required because torch_runtime.require_flash_attention=true. '
                'Run: bash scripts/install_flash_attention.sh configs/qwen3_tts_base.yaml'
            )
    compute_dtype = torch.bfloat16
    cache_dir = resolve_hf_hub_cache_dir(output_dir)
    load_started = time.perf_counter()
    root_dir = resolve_local_model_path(model_id, cache_dir)

    tts_wrapper = Qwen3TTSModel.from_pretrained(
        str(root_dir),
        device_map='cpu',
        dtype=compute_dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )

    quant_meta = describe_quantization(quantization)
    if quantization == 'bnb4':
        quant_report = apply_selective_bnb4_quantization(
            tts_wrapper.model,
            llm_bit=4,
            mtp_bit=4,
            compute_dtype=compute_dtype,
            keep_lm_head_in_base_dtype=True,
        )
    else:
        quant_report = apply_selective_bnb4_quantization(
            tts_wrapper.model,
            llm_bit=None,
            mtp_bit=None,
            compute_dtype=compute_dtype,
            keep_lm_head_in_base_dtype=True,
        )

    target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts_wrapper.model = tts_wrapper.model.to(target_device)
    tts_wrapper.model.eval()
    try:
        tts_wrapper.device = next(tts_wrapper.model.parameters()).device
    except StopIteration:
        tts_wrapper.device = torch.device(target_device)

    load_meta = {
        **quant_meta,
        'runtime_env': 'torch_hf_direct',
        'runtime_engine': 'qwen_tts.Qwen3TTSModel',
        'model_id': model_id,
        'model_tag': str(model_id).split('/')[-1],
        'base_dtype': 'bfloat16',
        'compute_dtype': dtype_to_str(compute_dtype),
        'attn_implementation': attn_implementation,
        'load_time_sec': time.perf_counter() - load_started,
        'model_memory_footprint_mb': get_model_memory_footprint_mb(tts_wrapper.model),
        'gpu_mem_after_load_mb': torch.cuda.memory_allocated() / _MB if torch.cuda.is_available() else None,
        'model_snapshot_disk_size_mb': directory_size_mb(root_dir),
        'disk_size_mb': directory_size_mb(root_dir),
        'model_root_dir': str(root_dir),
        'hf_cache_dir': str(cache_dir),
    }
    return tts_wrapper, quant_report, load_meta


def guess_language(text: Optional[str]) -> str:
    value = text or ''
    for ch in value:
        if '\uac00' <= ch <= '\ud7a3':
            return 'Korean'
    return 'Auto'


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
        raw_wavs = [arr] if arr.ndim == 1 else [arr[index] for index in range(arr.shape[0])]
    else:
        arr = np.array(audio)
        raw_wavs = [arr] if arr.ndim == 1 else [arr[index] for index in range(arr.shape[0])]
    wavs = [ensure_mono_float32(np.array(wav).squeeze()) for wav in raw_wavs]
    if not wavs:
        raise RuntimeError('Generated audio is empty')
    return wavs, int(sample_rate)


def generate_batch(tts_wrapper: Any, samples: List[BenchmarkSample], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int]:
    runtime = config.get('torch_runtime', {})
    texts = [sample.target_text or '' for sample in samples]
    languages = [guess_language(sample.target_text) for sample in samples]
    ref_audios = [sample.resolved_ref_audio_path for sample in samples]
    ref_texts = [sample.ref_text or '' for sample in samples]
    autocast_enabled = torch.cuda.is_available() and bool(runtime.get('autocast_bf16', True))
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
            result = tts_wrapper.generate_voice_clone(
                text=texts,
                language=languages,
                ref_audio=ref_audios,
                ref_text=ref_texts,
                non_streaming_mode=True,
                max_new_tokens=int(runtime.get('max_new_tokens', 512)),
                do_sample=bool(runtime.get('do_sample', True)),
                top_k=int(runtime.get('top_k', 50)),
                top_p=float(runtime.get('top_p', 0.95)),
                temperature=float(runtime.get('temperature', 1.0)),
                subtalker_dosample=bool(runtime.get('subtalker_dosample', False)),
                use_cache=bool(runtime.get('use_cache', True)),
            )
    return unpack_generation_result(result)


def audio_duration_from_array(wav: np.ndarray, sample_rate: int) -> Optional[float]:
    if sample_rate <= 0:
        return None
    return float(len(wav)) / float(sample_rate)


def build_output_audio_path(audio_root: str, sample: BenchmarkSample, suffix: str) -> str:
    subset = sample.subset.replace(os.sep, '_')
    path = os.path.join(audio_root, subset, f'{sample.pair_id}_{suffix}.wav')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def build_record_base(sample: BenchmarkSample, run_id: str, batch_id: int, batch_size: int, load_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'run_id': run_id,
        'request_id': sample.pair_id,
        'row_index': sample.row_index,
        'subset': sample.subset,
        'pair_id': sample.pair_id,
        'concurrency': 1,
        'batch_id': batch_id,
        'success': False,
        'error_type': None,
        'error_message': None,
        'http_status_code': None,
        'ref_audio_path': sample.ref_audio_path,
        'ref_audio_relpath': sample.ref_audio_relpath,
        'resolved_ref_audio_path': sample.resolved_ref_audio_path,
        'ref_duration_sec': sample.ref_duration_sec,
        'ref_text': sample.ref_text,
        'target_text': sample.target_text,
        'target_text_len': len(sample.target_text or ''),
        'target_audio_path': sample.target_audio_path,
        'target_audio_relpath': sample.target_audio_relpath,
        'target_duration_sec': sample.target_duration_sec,
        'request_start_time_iso': None,
        'request_end_time_iso': None,
        'end_to_end_latency_sec': None,
        'time_to_first_byte_sec': None,
        'time_to_first_audio_chunk_sec': None,
        'response_format': 'wav',
        'output_audio_path': None,
        'output_audio_bytes': None,
        'audio_duration_sec': None,
        'rtf': None,
        'end_to_end_rtf': None,
        'audio_throughput_sec_per_sec': None,
        'end_to_end_audio_throughput_sec_per_sec': None,
        'post_first_audio_chunk_latency_sec': None,
        'steady_state_audio_throughput_sec_per_sec': None,
        'steady_state_streaming_rtf': None,
        'steady_state_metric_status': 'request_failed',
        'audio_bytes_per_sec': None,
        'prompt_tokens': None,
        'completion_tokens': None,
        'total_tokens': None,
        'tokens_per_sec': None,
        'output_tokens_per_sec': None,
        'end_to_end_output_tokens_per_sec': None,
        'token_metric_status': 'unavailable_from_endpoint_or_metrics',
        'runtime_env': load_meta.get('runtime_env'),
        'runtime_engine': load_meta.get('runtime_engine'),
        'quant_method': load_meta.get('quant_method'),
        'bit_width': load_meta.get('bit_width'),
        'llm_bit': load_meta.get('llm_bit'),
        'mtp_bit': load_meta.get('mtp_bit'),
        'llm_quant_method': load_meta.get('llm_quant_method'),
        'mtp_quant_method': load_meta.get('mtp_quant_method'),
        'compute_dtype': load_meta.get('compute_dtype'),
        'attn_implementation': load_meta.get('attn_implementation'),
        'configured_batch_size': batch_size,
    }


def run_warmup(tts_wrapper: Any, samples: List[BenchmarkSample], config: Dict[str, Any], warmup_count: int, batch_size: int) -> int:
    warmup_samples = [sample for sample in samples if sample.valid][:warmup_count]
    ok = 0
    if not warmup_samples:
        return ok
    batches = [warmup_samples[index:index + batch_size] for index in range(0, len(warmup_samples), batch_size)]
    for batch in tqdm(batches, desc='warmup', unit='batch'):
        try:
            generate_batch(tts_wrapper, batch, config)
            ok += len(batch)
        except Exception as exc:
            print(f'warmup failed: {type(exc).__name__}: {exc}')
    return ok


def run_generation(
    tts_wrapper: Any,
    samples: List[BenchmarkSample],
    config: Dict[str, Any],
    output_dir: str,
    run_id: str,
    load_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    benchmark = config.get('benchmark', {})
    batch_size = int(benchmark.get('batch_size', 1) or 1)
    save_audio = bool(benchmark.get('save_audio', True))
    audio_root = os.path.join(output_dir, 'audio')
    records: List[Dict[str, Any]] = []
    valid_samples = [sample for sample in samples if sample.valid]
    batches = [valid_samples[index:index + batch_size] for index in range(0, len(valid_samples), batch_size)]

    for batch_id, batch in enumerate(tqdm(batches, desc='torch inference', unit='batch'), start=1):
        batch_start_iso = datetime.now(timezone.utc).isoformat()
        batch_start = time.perf_counter()
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            wavs, sample_rate = generate_batch(tts_wrapper, batch, config)
            elapsed = time.perf_counter() - batch_start
            if len(wavs) != len(batch):
                raise RuntimeError(f'Generated outputs mismatch: outputs={len(wavs)}, inputs={len(batch)}')
            amortized_latency = elapsed / len(batch) if batch else None
            for sample, wav in zip(batch, wavs):
                record = build_record_base(sample, run_id, batch_id, batch_size, load_meta)
                record['request_start_time_iso'] = batch_start_iso
                record['request_end_time_iso'] = datetime.now(timezone.utc).isoformat()
                record['success'] = True
                record['end_to_end_latency_sec'] = amortized_latency
                record['batch_end_to_end_latency_sec'] = elapsed
                record['batch_size_actual'] = len(batch)
                record['audio_duration_sec'] = audio_duration_from_array(wav, sample_rate)
                if save_audio:
                    output_audio_path = build_output_audio_path(audio_root, sample, 'torch')
                    sf.write(output_audio_path, wav, sample_rate)
                    record['output_audio_path'] = output_audio_path
                    try:
                        record['output_audio_bytes'] = os.path.getsize(output_audio_path)
                    except OSError:
                        record['output_audio_bytes'] = None
                    if record['audio_duration_sec'] is None:
                        record['audio_duration_sec'] = get_audio_duration(output_audio_path)
                if record['audio_duration_sec'] and record['audio_duration_sec'] > 0 and amortized_latency and amortized_latency > 0:
                    record['rtf'] = amortized_latency / record['audio_duration_sec']
                    record['end_to_end_rtf'] = record['rtf']
                    record['audio_throughput_sec_per_sec'] = record['audio_duration_sec'] / amortized_latency
                    record['end_to_end_audio_throughput_sec_per_sec'] = record['audio_throughput_sec_per_sec']
                    if record['output_audio_bytes'] is not None:
                        record['audio_bytes_per_sec'] = record['output_audio_bytes'] / amortized_latency
                steady = compute_steady_state_audio_metrics(
                    True,
                    record['end_to_end_latency_sec'],
                    record['time_to_first_audio_chunk_sec'],
                    record['audio_duration_sec'],
                )
                record.update(steady)
                records.append(record)
        except Exception as exc:
            for sample in batch:
                record = build_record_base(sample, run_id, batch_id, batch_size, load_meta)
                record['request_start_time_iso'] = batch_start_iso
                record['request_end_time_iso'] = datetime.now(timezone.utc).isoformat()
                record['error_type'] = type(exc).__name__
                record['error_message'] = str(exc)
                records.append(record)
            print(f'batch {batch_id} failed: {type(exc).__name__}: {exc}')
            error_dir = os.path.join(output_dir, 'logs')
            os.makedirs(error_dir, exist_ok=True)
            with open(os.path.join(error_dir, f'{run_id}_batch_{batch_id:04d}_error.txt'), 'w', encoding='utf-8') as f:
                f.write(traceback.format_exc())
    return records


def make_run_settings(config: Dict[str, Any], run_id: str, output_dir: str, samples: List[BenchmarkSample], load_meta: Dict[str, Any], quant_report: Dict[str, Any]) -> Dict[str, Any]:
    env = gather_environment_info()
    return {
        'saved_at': datetime.now(timezone.utc).isoformat(),
        'run_name': run_id,
        'run_order': 1,
        'model_id': load_meta.get('model_id'),
        'model_tag': load_meta.get('model_tag'),
        'manifest_path': config.get('manifest', {}).get('path'),
        'manifest_size': len(samples),
        'seed': config.get('run', {}).get('seed'),
        'warmup_count': config.get('benchmark', {}).get('warmup_requests'),
        'sampling_config': {
            'max_new_tokens': config.get('torch_runtime', {}).get('max_new_tokens'),
            'temperature': config.get('torch_runtime', {}).get('temperature'),
            'top_p': config.get('torch_runtime', {}).get('top_p'),
            'top_k': config.get('torch_runtime', {}).get('top_k'),
            'do_sample': config.get('torch_runtime', {}).get('do_sample'),
            'subtalker_dosample': config.get('torch_runtime', {}).get('subtalker_dosample'),
            'use_cache': config.get('torch_runtime', {}).get('use_cache'),
            'batch_size': config.get('benchmark', {}).get('batch_size'),
        },
        'runtime_env': load_meta.get('runtime_env'),
        'runtime_engine': load_meta.get('runtime_engine'),
        'target_device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'base_dtype': load_meta.get('base_dtype'),
        'batch_size': config.get('benchmark', {}).get('batch_size'),
        'quantization': {**load_meta, **quant_report},
        'hardware': {
            'gpu_names': env.get('gpu_names'),
            'gpu_count': env.get('gpu_count'),
            'cuda_available': env.get('cuda_available'),
            'cuda_visible_devices': env.get('cuda_visible_devices'),
            'nvidia_driver_version': env.get('nvidia_driver_version'),
        },
        'software': {
            'python_version': env.get('python_version'),
            'platform': env.get('platform') or platform.platform(),
            'torch_version': env.get('torch_version'),
            'transformers_version': getattr(transformers, '__version__', None) if transformers else None,
            'bitsandbytes_version': getattr(bnb, '__version__', None) if bnb else None,
        },
    }


def write_colab_summary(output_dir: str, run_id: str, records: List[Dict[str, Any]], group_summary: Dict[str, Any], load_meta: Dict[str, Any], warmup_ok: int, elapsed: float) -> Dict[str, Any]:
    success_records = [record for record in records if record.get('success')]
    failed_records = [record for record in records if not record.get('success')]
    summary = {
        'run_name': run_id,
        'run_order': 1,
        'model_id': load_meta.get('model_id'),
        'model_tag': load_meta.get('model_tag'),
        'quant_method': load_meta.get('quant_method'),
        'bit_width': load_meta.get('bit_width'),
        'llm_bit': load_meta.get('llm_bit'),
        'mtp_bit': load_meta.get('mtp_bit'),
        'llm_quant_method': load_meta.get('llm_quant_method'),
        'mtp_quant_method': load_meta.get('mtp_quant_method'),
        'compute_dtype': load_meta.get('compute_dtype'),
        'attn_implementation': load_meta.get('attn_implementation'),
        'runtime_env': load_meta.get('runtime_env'),
        'runtime_engine': load_meta.get('runtime_engine'),
        'configured_batch_size': group_summary.get('configured_batch_size'),
        'concurrency': 1,
        'warmup_count_done': warmup_ok,
        'warmup_ok': warmup_ok,
        'total_rows': len(records),
        'ok_rows': len(success_records),
        'error_rows': len(failed_records),
        'success_rate': group_summary.get('success_rate'),
        'run_elapsed_sec': elapsed,
        'gen_sec_mean': group_summary.get('latency_mean_sec'),
        'gen_sec_median': group_summary.get('latency_median_sec'),
        'gen_sec_p95': group_summary.get('latency_p95_sec'),
        'audio_sec_mean': group_summary.get('audio_duration_mean_sec'),
        'audio_sec_total': group_summary.get('audio_duration_total_sec'),
        'rtf_mean': group_summary.get('rtf_mean'),
        'rtf_p95': group_summary.get('rtf_p95'),
        'time_to_first_audio_chunk_p50_sec': group_summary.get('time_to_first_audio_chunk_p50_sec'),
        'time_to_first_audio_chunk_p95_sec': group_summary.get('time_to_first_audio_chunk_p95_sec'),
        'steady_state_streaming_rtf_mean': group_summary.get('steady_state_streaming_rtf_mean'),
        'steady_state_streaming_rtf_p95': group_summary.get('steady_state_streaming_rtf_p95'),
        'steady_state_audio_throughput_mean_sec_per_sec': group_summary.get('steady_state_audio_throughput_mean_sec_per_sec'),
        'requests_per_sec': group_summary.get('requests_per_sec'),
        'successful_requests_per_sec': group_summary.get('successful_requests_per_sec'),
        'load_time_sec': load_meta.get('load_time_sec'),
        'model_memory_footprint_mb': load_meta.get('model_memory_footprint_mb'),
        'gpu_mem_after_load_mb': load_meta.get('gpu_mem_after_load_mb'),
    }
    gpu_summary = group_summary.get('gpu_memory_summary') or {}
    if isinstance(gpu_summary, dict):
        summary.update({
            'peak_gpu_mem_mb': gpu_summary.get('gpu_memory_peak_mb'),
            'allocated_gpu_mem_mb': gpu_summary.get('gpu_memory_after_benchmark_mb'),
            'gpu_utilization_mean_percent': gpu_summary.get('gpu_utilization_mean_percent'),
            'gpu_utilization_peak_percent': gpu_summary.get('gpu_utilization_peak_percent'),
            'process_ram_mb': gpu_summary.get('process_rss_peak_mb'),
            'system_ram_peak_mb': gpu_summary.get('system_ram_peak_mb'),
        })
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, f'{run_id}_run_summary.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, 'all_runs_summary.csv'), index=False)
    return summary


def main() -> None:
    args = parse_args()
    config = merge_config(load_config(args.config), args)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'audio'), exist_ok=True)
    set_seed(config.get('run', {}).get('seed'))
    torch.set_grad_enabled(False)

    manifest_config = config.get('manifest', {})
    benchmark = config.get('benchmark', {})
    samples = load_manifest(
        manifest_config['path'],
        audio_root=manifest_config.get('audio_root'),
        path_prefix_from=manifest_config.get('path_prefix_from'),
        path_prefix_to=manifest_config.get('path_prefix_to'),
        subset_filter=benchmark.get('subset'),
        limit=benchmark.get('limit'),
        seed=config.get('run', {}).get('seed'),
    )
    save_validated_manifest(samples, os.path.join(args.output_dir, 'manifest_validated.csv'))

    quant_suffix = 'bnb4' if config.get('torch_runtime', {}).get('quantization') == 'bnb4' else 'bf16'
    run_id = f"qwen3_tts_reference_300_torch_{quant_suffix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    start_time = time.perf_counter()
    tts_wrapper, quant_report, load_meta = load_torch_model(config, args.output_dir)
    settings = make_run_settings(config, run_id, args.output_dir, samples, load_meta, quant_report)
    _write_json(os.path.join(args.output_dir, f'{run_id}_run_settings.json'), settings)

    batch_size = int(benchmark.get('batch_size', 1) or 1)
    warmup_count = int(benchmark.get('warmup_requests', 0) or 0)
    warmup_ok = run_warmup(tts_wrapper, samples, config, warmup_count, batch_size)

    collector = MetricsCollector(sample_interval=float(benchmark.get('metrics_sample_interval_sec', 0.2) or 0.2), process_pid=os.getpid())
    collector.start()
    bench_start = time.perf_counter()
    records = run_generation(tts_wrapper, samples, config, args.output_dir, run_id, load_meta)
    bench_end = time.perf_counter()
    collector.stop()

    memory_summary = collector.export_summary()
    _write_json(os.path.join(args.output_dir, 'memory_summary.json'), memory_summary)
    if collector.samples:
        pd.DataFrame(collector.samples).to_csv(os.path.join(args.output_dir, 'gpu_memory_timeseries.csv'), index=False)

    group_summary = compute_group_summary(records, bench_start, bench_end, 1, {'gpu_memory_summary': memory_summary})
    group_summary['configured_batch_size'] = batch_size
    group_summary['runtime_env'] = load_meta.get('runtime_env')
    group_summary['runtime_engine'] = load_meta.get('runtime_engine')
    group_summary['quant_method'] = load_meta.get('quant_method')
    group_summary['bit_width'] = load_meta.get('bit_width')
    group_summary['llm_bit'] = load_meta.get('llm_bit')
    group_summary['mtp_bit'] = load_meta.get('mtp_bit')
    group_summary['llm_quant_method'] = load_meta.get('llm_quant_method')
    group_summary['mtp_quant_method'] = load_meta.get('mtp_quant_method')
    group_summary['compute_dtype'] = load_meta.get('compute_dtype')
    group_summary['load_time_sec'] = load_meta.get('load_time_sec')
    group_summary['model_memory_footprint_mb'] = load_meta.get('model_memory_footprint_mb')
    group_summary['gpu_mem_after_load_mb'] = load_meta.get('gpu_mem_after_load_mb')

    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config_path': os.path.abspath(args.config),
        'config': config,
        'output_dir': os.path.abspath(args.output_dir),
        'cli': ' '.join(os.sys.argv),
    }
    _write_json(os.path.join(args.output_dir, 'metadata.json'), metadata)
    _write_jsonl(os.path.join(args.output_dir, 'requests.jsonl'), records)
    pd.DataFrame(records).to_csv(os.path.join(args.output_dir, 'requests.csv'), index=False)
    _write_json(os.path.join(args.output_dir, 'summary.json'), [group_summary])
    pd.DataFrame([group_summary]).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    pd.DataFrame(records).to_csv(os.path.join(args.output_dir, f'{run_id}_inference_manifest.csv'), index=False)
    pd.DataFrame(records).to_csv(os.path.join(args.output_dir, f'{run_id}_inference_manifest.partial.csv'), index=False)
    write_colab_summary(args.output_dir, run_id, records, group_summary, load_meta, warmup_ok, time.perf_counter() - start_time)

    del tts_wrapper
    cleanup_memory()
    print(f'Torch benchmark complete. Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
