import argparse
import asyncio
import glob
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import httpx
import pandas as pd
import yaml
from tqdm import tqdm

from src.audio_utils import ensure_directory
from src.config import load_config, validate_config
from src.env_info import gather_environment_info
from src.manifest_loader import BenchmarkSample, load_manifest, save_validated_manifest
from src.metrics_collector import MetricsCollector
from src.tts_client import RequestRecord, TTSClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Qwen3-TTS benchmark from a manifest')
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', help='Path to reference_manifest.csv')
    parser.add_argument('--audio-root', help='Audio root path for ref_audio_relpath')
    parser.add_argument('--api-base', help='Override API base URL')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--warmup-requests', type=int, help='Number of warmup requests')
    parser.add_argument('--concurrency', type=int, nargs='+', help='Concurrency values to benchmark')
    parser.add_argument('--save-audio', dest='save_audio', action='store_true', help='Save generated audio')
    parser.add_argument('--no-save-audio', dest='save_audio', action='store_false', help='Do not save generated audio')
    parser.add_argument('--response-format', help='Response format such as wav')
    parser.add_argument('--timeout-sec', type=int, help='Request timeout seconds')
    parser.add_argument('--sample-rate', type=int, help='Audio sample rate')
    parser.add_argument('--request-rate', type=float, help='Target request rate per second')
    parser.add_argument('--repeat-per-sample', type=int, help='Repeat each sample this many times')
    parser.add_argument('--batch-size', type=int, default=1, help='Logical batch size for grouping/reporting samples')
    parser.add_argument('--metrics-sample-interval-sec', type=float, help='GPU/process metrics sampling interval seconds')
    parser.add_argument('--subset', action='append', help='Subset filter: clean, noisy, numeric')
    parser.add_argument('--limit', type=int, help='Limit number of manifest rows for benchmark')
    parser.add_argument('--seed', type=int, help='Random seed for manifest ordering')
    parser.add_argument('--path-prefix-from', help='Replace path prefix from')
    parser.add_argument('--path-prefix-to', help='Replace path prefix to')
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode')
    parser.set_defaults(save_audio=None)
    return parser.parse_args()


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    config['manifest'] = dict(config.get('manifest', {}))
    config['tts'] = dict(config.get('tts', {}))
    config['benchmark'] = dict(config.get('benchmark', {}))
    config['server'] = dict(config.get('server', {}))

    if args.manifest:
        config['manifest']['path'] = args.manifest
    if args.audio_root:
        config['manifest']['audio_root'] = args.audio_root
    if args.path_prefix_from:
        config['manifest']['path_prefix_from'] = args.path_prefix_from
    if args.path_prefix_to:
        config['manifest']['path_prefix_to'] = args.path_prefix_to
    if args.api_base:
        config['server']['api_base'] = args.api_base
    if args.warmup_requests is not None:
        config['benchmark']['warmup_requests'] = args.warmup_requests
    if args.concurrency is not None:
        config['benchmark']['concurrency'] = args.concurrency
    if args.timeout_sec is not None:
        config['benchmark']['timeout_sec'] = args.timeout_sec
    if args.response_format is not None:
        config['tts']['response_format'] = args.response_format
    if args.sample_rate is not None:
        config['tts']['sample_rate'] = args.sample_rate
    if args.request_rate is not None:
        config['benchmark']['request_rate'] = args.request_rate
    if args.repeat_per_sample is not None:
        config['benchmark']['repeat_per_sample'] = args.repeat_per_sample
    if args.batch_size is not None:
        config['benchmark']['batch_size'] = args.batch_size
    if args.metrics_sample_interval_sec is not None:
        config['benchmark']['metrics_sample_interval_sec'] = args.metrics_sample_interval_sec
    if args.subset is not None:
        config['benchmark']['subset'] = args.subset
    if args.limit is not None:
        config['benchmark']['limit'] = args.limit
    if args.seed is not None:
        config['run'] = dict(config.get('run', {}))
        config['run']['seed'] = args.seed
    if args.stream:
        config['tts']['stream'] = True
    if args.save_audio is not None:
        config['benchmark']['save_audio'] = args.save_audio
    return config


def _ensure_output_dirs(output_dir: str) -> Dict[str, str]:
    paths = {
        'root': output_dir,
        'audio': os.path.join(output_dir, 'audio'),
        'prometheus': os.path.join(output_dir, 'prometheus'),
        'plots': os.path.join(output_dir, 'plots'),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _write_json(path: str, data: Any) -> None:
    ensure_directory(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(data), f, indent=2)


def _json_safe(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_safe(value) for value in data]
    if isinstance(data, float) and not math.isfinite(data):
        return None
    if hasattr(data, 'item'):
        return _json_safe(data.item())
    return data


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _finite_values(records: List[Dict[str, Any]], key: str, sort_values: bool = False) -> List[float]:
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if _is_finite_number(value):
            values.append(float(value))
    if sort_values:
        values.sort()
    return values


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_directory(path)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False, default=str, allow_nan=False) + '\n')


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _guess_qwen_language(text: Optional[str]) -> str:
    text = text or ''
    if any('\uac00' <= ch <= '\ud7a3' for ch in text):
        return 'Korean'
    if any('\u3040' <= ch <= '\u30ff' or '\u31f0' <= ch <= '\u31ff' for ch in text):
        return 'Japanese'
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        return 'Chinese'
    return 'Auto'


def _model_tag(model_id: Optional[str]) -> str:
    if not model_id:
        return 'unknown'
    name = str(model_id).split('/')[-1]
    for token in name.split('-'):
        if token.endswith('B') and any(ch.isdigit() for ch in token):
            return token
    return name


def _colab_sampling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    tts_config = config.get('tts', {})
    benchmark_config = config.get('benchmark', {})
    return {
        'temperature': tts_config.get('temperature'),
        'top_p': tts_config.get('top_p'),
        'top_k': tts_config.get('top_k'),
        'max_new_tokens': tts_config.get('max_new_tokens') or tts_config.get('max_tokens'),
        'do_sample': tts_config.get('do_sample'),
        'non_streaming_mode': not bool(tts_config.get('stream', False)),
        'subtalker_dosample': tts_config.get('subtalker_dosample'),
        'use_cache': tts_config.get('use_cache'),
        'batch_size': benchmark_config.get('batch_size'),
        'concurrency': benchmark_config.get('concurrency'),
        'request_rate': benchmark_config.get('request_rate'),
        'response_format': tts_config.get('response_format'),
        'sample_rate': tts_config.get('sample_rate'),
    }


def _vllm_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    server_config = config.get('server', {})
    quantization = server_config.get('quantization') or 'none'
    bit_width = server_config.get('bit_width')
    if bit_width is None:
        bit_width = 4 if quantization == 'bitsandbytes' else 'bf16'
    quant_method = quantization if quantization != 'none' else 'none'
    llm_bit = server_config.get('llm_bit')
    if llm_bit is None:
        llm_bit = bit_width if quantization != 'none' else None
    mtp_quant_method = server_config.get('mtp_quantization') or ('none' if quantization == 'none' else quant_method)
    mtp_bit = server_config.get('mtp_bit')
    if mtp_bit is None:
        mtp_bit = bit_width if mtp_quant_method != 'none' else None
    return {
        'runtime_env': 'vllm_omni_server',
        'runtime_engine': server_config.get('executable', 'vllm-omni'),
        'serving_mode': 'openai_compatible_http',
        'api_base': server_config.get('api_base'),
        'endpoint': config.get('tts', {}).get('endpoint'),
        'model_id': server_config.get('model_id'),
        'model_tag': _model_tag(server_config.get('model_id')),
        'base_dtype': 'bfloat16',
        'compute_dtype': 'bfloat16',
        'attn_implementation': 'vllm_FLASH_ATTN_v2',
        'quant_method': quant_method,
        'bit_width': bit_width,
        'llm_bit': llm_bit,
        'mtp_bit': mtp_bit,
        'llm_quant_method': quant_method,
        'mtp_quant_method': mtp_quant_method,
        'quantization_scope': server_config.get('quantization_scope'),
        'load_format': server_config.get('load_format'),
        'tensor_parallel_size': server_config.get('tensor_parallel_size'),
        'max_num_seqs': server_config.get('max_num_seqs'),
        'gpu_memory_utilization': server_config.get('gpu_memory_utilization'),
        'max_model_len': server_config.get('max_model_len'),
        'stage_configs_path': server_config.get('stage_configs_path'),
        'allowed_local_media_path': server_config.get('allowed_local_media_path'),
        'trust_remote_code': server_config.get('trust_remote_code'),
        'request_mode': config.get('tts', {}).get('request_mode'),
        'payload_mode': config.get('tts', {}).get('payload_mode'),
        'task_type': config.get('tts', {}).get('task_type'),
    }


def _write_prompt_list(path: str, samples: List[BenchmarkSample]) -> None:
    rows = []
    for sample in samples:
        rows.append({
            'pair_id': sample.pair_id,
            'target_text': sample.target_text,
            'ref_audio_path': sample.ref_audio_path,
            'ref_audio_relpath': sample.ref_audio_relpath,
            'resolved_ref_audio_path': sample.resolved_ref_audio_path,
            'ref_text': sample.ref_text,
            'subset': sample.subset,
        })
    _write_jsonl(path, rows)


def _write_run_settings(
    output_dir: str,
    run_id: str,
    config: Dict[str, Any],
    env: Dict[str, Any],
    manifest_samples: List[BenchmarkSample],
    prompt_list_path: str,
    cli: str,
) -> str:
    runtime = _vllm_runtime_config(config)
    server_config = config.get('server', {})
    benchmark_config = config.get('benchmark', {})
    payload = {
        'saved_at': _now_iso(),
        'run_name': run_id,
        'run_order': 1,
        'model_id': runtime['model_id'],
        'model_tag': runtime['model_tag'],
        'manifest_path': config.get('manifest', {}).get('path'),
        'manifest_size': len(manifest_samples),
        'prompt_list_path': prompt_list_path,
        'seed': config.get('run', {}).get('seed'),
        'warmup_count': benchmark_config.get('warmup_requests'),
        'sampling_config': _colab_sampling_config(config),
        'worker_tag': 'local_vllm',
        'worker_index': 0,
        'worker_count': 1,
        'runtime_env': runtime['runtime_env'],
        'target_device': 'cuda' if env.get('cuda_available') else 'cpu',
        'base_dtype': runtime['base_dtype'],
        'batch_size': benchmark_config.get('batch_size'),
        'concurrency': benchmark_config.get('concurrency'),
        'quantization': {
            'llm_bit': runtime['llm_bit'],
            'mtp_bit': runtime['mtp_bit'],
            'keep_lm_head_in_base_dtype': True,
            'attn_implementation': runtime['attn_implementation'],
            'quant_method': runtime['quant_method'],
            'bit_width': runtime['bit_width'],
            'llm_quant_method': runtime['llm_quant_method'],
            'mtp_quant_method': runtime['mtp_quant_method'],
            'compute_dtype': runtime['compute_dtype'],
        },
        'vllm_runtime': runtime,
        'server': server_config,
        'hardware': {
            'gpu_names': env.get('gpu_names'),
            'gpu_count': env.get('gpu_count'),
            'cuda_available': env.get('cuda_available'),
            'cuda_visible_devices': env.get('cuda_visible_devices'),
            'nvidia_driver_version': env.get('nvidia_driver_version'),
            'nvidia_smi': env.get('nvidia_smi'),
        },
        'software': {
            'python_version': env.get('python_version'),
            'platform': env.get('platform') or platform.platform(),
            'torch_version': env.get('torch_version'),
            'transformers_version': env.get('transformers_version'),
            'vllm_version': env.get('vllm_version'),
            'vllm_omni_version': env.get('vllm_omni_version'),
        },
        'cli': cli,
    }
    settings_path = os.path.join(output_dir, f'{run_id}_run_settings.json')
    _write_json(settings_path, payload)
    return settings_path


def _to_colab_manifest_rows(
    records: List[Dict[str, Any]],
    run_id: str,
    config: Dict[str, Any],
    settings_path: str,
    prompt_list_path: str,
) -> List[Dict[str, Any]]:
    runtime = _vllm_runtime_config(config)
    sampling = _colab_sampling_config(config)
    benchmark_config = config.get('benchmark', {})
    rows: List[Dict[str, Any]] = []
    for row in records:
        if row.get('concurrency') == 0 and row.get('error_type') == 'invalid_sample':
            status = 'invalid'
        elif row.get('success'):
            status = 'ok'
        else:
            status = 'error'
        latency = row.get('end_to_end_latency_sec')
        audio_sec = row.get('audio_duration_sec')
        rows.append({
            **row,
            'run_name': run_id,
            'run_order': 1,
            'model_id': runtime['model_id'],
            'model_tag': runtime['model_tag'],
            'quant_method': runtime['quant_method'],
            'bit_width': runtime['bit_width'],
            'llm_bit': runtime['llm_bit'],
            'mtp_bit': runtime['mtp_bit'],
            'llm_quant_method': runtime['llm_quant_method'],
            'mtp_quant_method': runtime['mtp_quant_method'],
            'compute_dtype': runtime['compute_dtype'],
            'seed': config.get('run', {}).get('seed'),
            'warmup_count': benchmark_config.get('warmup_requests'),
            'manifest_size': None,
            'temperature': sampling.get('temperature'),
            'top_p': sampling.get('top_p'),
            'top_k': sampling.get('top_k'),
            'max_new_tokens': sampling.get('max_new_tokens'),
            'do_sample': sampling.get('do_sample'),
            'non_streaming_mode': int(bool(sampling.get('non_streaming_mode'))),
            'subtalker_dosample': sampling.get('subtalker_dosample'),
            'use_cache': sampling.get('use_cache'),
            'configured_batch_size': benchmark_config.get('batch_size'),
            'batch_size_actual': None,
            'batch_index': (row.get('batch_id') - 1) if isinstance(row.get('batch_id'), int) else row.get('batch_id'),
            'load_time_sec': None,
            'model_memory_footprint_mb': None,
            'disk_size_mb': None,
            'model_snapshot_disk_size_mb': None,
            'gpu_mem_after_load_mb': None,
            'prompt_list_path': prompt_list_path,
            'run_settings_path': settings_path,
            'settings_path': settings_path,
            'runtime_env': runtime['runtime_env'],
            'runtime_engine': runtime['runtime_engine'],
            'attn_implementation': runtime['attn_implementation'],
            'request_mode': runtime['request_mode'],
            'payload_mode': runtime['payload_mode'],
            'task_type': runtime['task_type'],
            'generation_language': _guess_qwen_language(row.get('target_text')),
            'batch_mode': 'vllm_http_concurrent_requests',
            'fallback_used': 0,
            'status': status,
            'error': row.get('error_message'),
            'output_wav_path': row.get('output_audio_path'),
            'gen_sec': latency,
            'audio_sec': audio_sec,
            'gen_latency_sec': latency,
            'gen_duration_sec': audio_sec,
            'gen_rtf': row.get('rtf'),
        })
    manifest_size = len(rows)
    for row in rows:
        row['manifest_size'] = manifest_size
    return rows


def _write_partial_manifest(path: str, records: List[Dict[str, Any]], run_id: str, config: Dict[str, Any], settings_path: str, prompt_list_path: str) -> None:
    rows = _to_colab_manifest_rows(records, run_id, config, settings_path, prompt_list_path)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_colab_summary(
    output_dir: str,
    run_id: str,
    records: List[Dict[str, Any]],
    group_summary: Dict[str, Any],
    config: Dict[str, Any],
    settings_path: str,
    prompt_list_path: str,
    warmup_count_done: int,
    warmup_ok: int,
    run_elapsed_sec: float,
) -> Dict[str, Any]:
    runtime = _vllm_runtime_config(config)
    success_records = [r for r in records if r.get('success')]
    failed_records = [r for r in records if not r.get('success')]
    summary = {
        'run_name': run_id,
        'run_order': 1,
        'model_id': runtime['model_id'],
        'model_tag': runtime['model_tag'],
        'quant_method': runtime['quant_method'],
        'bit_width': runtime['bit_width'],
        'llm_bit': runtime['llm_bit'],
        'mtp_bit': runtime['mtp_bit'],
        'llm_quant_method': runtime['llm_quant_method'],
        'mtp_quant_method': runtime['mtp_quant_method'],
        'compute_dtype': runtime['compute_dtype'],
        'attn_implementation': runtime['attn_implementation'],
        'runtime_env': runtime['runtime_env'],
        'runtime_engine': runtime['runtime_engine'],
        'configured_batch_size': config.get('benchmark', {}).get('batch_size'),
        'concurrency': group_summary.get('concurrency'),
        'warmup_count_done': warmup_count_done,
        'warmup_ok': warmup_ok,
        'total_rows': len(records),
        'ok_rows': len(success_records),
        'error_rows': len(failed_records),
        'success_rate': group_summary.get('success_rate'),
        'run_elapsed_sec': run_elapsed_sec,
        'gen_sec_mean': group_summary.get('latency_mean_sec'),
        'gen_sec_median': group_summary.get('latency_median_sec'),
        'gen_sec_p90': group_summary.get('latency_p90_sec'),
        'gen_sec_p95': group_summary.get('latency_p95_sec'),
        'gen_sec_min': group_summary.get('latency_min_sec'),
        'gen_sec_max': group_summary.get('latency_max_sec'),
        'audio_sec_mean': group_summary.get('audio_duration_mean_sec'),
        'audio_sec_total': group_summary.get('audio_duration_total_sec'),
        'rtf_mean': group_summary.get('rtf_mean'),
        'rtf_median': group_summary.get('rtf_median'),
        'rtf_p90': group_summary.get('rtf_p90'),
        'rtf_p95': group_summary.get('rtf_p95'),
        'time_to_first_audio_chunk_p50_sec': group_summary.get('time_to_first_audio_chunk_p50_sec'),
        'time_to_first_audio_chunk_p95_sec': group_summary.get('time_to_first_audio_chunk_p95_sec'),
        'steady_state_streaming_rtf_mean': group_summary.get('steady_state_streaming_rtf_mean'),
        'steady_state_streaming_rtf_p95': group_summary.get('steady_state_streaming_rtf_p95'),
        'steady_state_audio_throughput_mean_sec_per_sec': group_summary.get('steady_state_audio_throughput_mean_sec_per_sec'),
        'requests_per_sec': group_summary.get('requests_per_sec'),
        'successful_requests_per_sec': group_summary.get('successful_requests_per_sec'),
        'prompt_list_path': prompt_list_path,
        'settings_path': settings_path,
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
    summary_csv = os.path.join(output_dir, f'{run_id}_run_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    return summary


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    k = (len(values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values[int(k)])
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return float(d0 + d1)


def compute_group_summary(records: List[Dict[str, Any]], bench_start: float, bench_end: float, concurrency: int, metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
    total = len(records)
    success_records = [r for r in records if r.get('success')]
    failed = total - len(success_records)
    bench_duration = bench_end - bench_start if bench_end > bench_start else None
    latency_values = _finite_values(success_records, 'end_to_end_latency_sec', sort_values=True)
    rtf_values = _finite_values(success_records, 'rtf', sort_values=True)
    audio_durations = _finite_values(success_records, 'audio_duration_sec')
    ttfb_values = _finite_values(success_records, 'time_to_first_byte_sec', sort_values=True)
    first_audio_values = _finite_values(success_records, 'time_to_first_audio_chunk_sec', sort_values=True)
    tokens_per_sec = _finite_values(success_records, 'tokens_per_sec')
    output_tokens_per_sec = _finite_values(success_records, 'output_tokens_per_sec')
    throughput_values = _finite_values(success_records, 'audio_throughput_sec_per_sec', sort_values=True)
    post_first_audio_chunk_latency_values = _finite_values(success_records, 'post_first_audio_chunk_latency_sec', sort_values=True)
    steady_state_audio_throughput_values = _finite_values(success_records, 'steady_state_audio_throughput_sec_per_sec', sort_values=True)
    steady_state_streaming_rtf_values = _finite_values(success_records, 'steady_state_streaming_rtf', sort_values=True)
    return {
        'concurrency': concurrency,
        'num_requests': total,
        'num_success': len(success_records),
        'num_failed': failed,
        'success_rate': len(success_records) / total if total else None,
        'latency_mean_sec': statistics.mean(latency_values) if latency_values else None,
        'latency_median_sec': statistics.median(latency_values) if latency_values else None,
        'latency_p50_sec': _percentile(latency_values, 50) if latency_values else None,
        'latency_p90_sec': _percentile(latency_values, 90) if latency_values else None,
        'latency_p95_sec': _percentile(latency_values, 95) if latency_values else None,
        'latency_p99_sec': _percentile(latency_values, 99) if latency_values else None,
        'latency_min_sec': min(latency_values) if latency_values else None,
        'latency_max_sec': max(latency_values) if latency_values else None,
        'rtf_mean': statistics.mean(rtf_values) if rtf_values else None,
        'rtf_median': statistics.median(rtf_values) if rtf_values else None,
        'rtf_p50': _percentile(rtf_values, 50) if rtf_values else None,
        'rtf_p90': _percentile(rtf_values, 90) if rtf_values else None,
        'rtf_p95': _percentile(rtf_values, 95) if rtf_values else None,
        'rtf_p99': _percentile(rtf_values, 99) if rtf_values else None,
        'end_to_end_rtf_mean': statistics.mean(rtf_values) if rtf_values else None,
        'end_to_end_rtf_p50': _percentile(rtf_values, 50) if rtf_values else None,
        'end_to_end_rtf_p95': _percentile(rtf_values, 95) if rtf_values else None,
        'audio_duration_mean_sec': statistics.mean(audio_durations) if audio_durations else None,
        'audio_duration_total_sec': sum(audio_durations) if audio_durations else None,
        'audio_throughput_mean_sec_per_sec': statistics.mean(throughput_values) if throughput_values else None,
        'end_to_end_audio_throughput_mean_sec_per_sec': statistics.mean(throughput_values) if throughput_values else None,
        'post_first_audio_chunk_latency_mean_sec': statistics.mean(post_first_audio_chunk_latency_values) if post_first_audio_chunk_latency_values else None,
        'post_first_audio_chunk_latency_p50_sec': _percentile(post_first_audio_chunk_latency_values, 50) if post_first_audio_chunk_latency_values else None,
        'post_first_audio_chunk_latency_p95_sec': _percentile(post_first_audio_chunk_latency_values, 95) if post_first_audio_chunk_latency_values else None,
        'steady_state_audio_throughput_mean_sec_per_sec': statistics.mean(steady_state_audio_throughput_values) if steady_state_audio_throughput_values else None,
        'steady_state_audio_throughput_p50_sec_per_sec': _percentile(steady_state_audio_throughput_values, 50) if steady_state_audio_throughput_values else None,
        'steady_state_audio_throughput_p95_sec_per_sec': _percentile(steady_state_audio_throughput_values, 95) if steady_state_audio_throughput_values else None,
        'steady_state_streaming_rtf_mean': statistics.mean(steady_state_streaming_rtf_values) if steady_state_streaming_rtf_values else None,
        'steady_state_streaming_rtf_p50': _percentile(steady_state_streaming_rtf_values, 50) if steady_state_streaming_rtf_values else None,
        'steady_state_streaming_rtf_p95': _percentile(steady_state_streaming_rtf_values, 95) if steady_state_streaming_rtf_values else None,
        'requests_per_sec': len(records) / bench_duration if bench_duration and bench_duration > 0 else None,
        'successful_requests_per_sec': len(success_records) / bench_duration if bench_duration and bench_duration > 0 else None,
        'time_to_first_byte_p50_sec': _percentile(ttfb_values, 50) if ttfb_values else None,
        'time_to_first_byte_p95_sec': _percentile(ttfb_values, 95) if ttfb_values else None,
        'time_to_first_audio_chunk_p50_sec': _percentile(first_audio_values, 50) if first_audio_values else None,
        'time_to_first_audio_chunk_p95_sec': _percentile(first_audio_values, 95) if first_audio_values else None,
        'tokens_per_sec_mean': statistics.mean(tokens_per_sec) if tokens_per_sec else None,
        'output_tokens_per_sec_mean': statistics.mean(output_tokens_per_sec) if output_tokens_per_sec else None,
        **metrics_summary,
    }


def _parse_prometheus_metrics(text: str) -> Dict[str, Any]:
    values = {}
    lines = [line for line in text.splitlines() if line and not line.startswith('#')]
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        try:
            value = float(parts[1])
        except Exception:
            continue
        values[key] = value
    return values


def _metrics_of_interest(parsed: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    lookup = {
        'running_requests': ['running_requests', 'vllm_request_running_current', 'request_running'],
        'waiting_requests': ['waiting_requests', 'vllm_request_waiting_current', 'request_waiting'],
        'prompt_token_throughput': ['prompt_token_throughput', 'vllm_prompt_token_throughput'],
        'generation_token_throughput': ['generation_token_throughput', 'vllm_generation_token_throughput'],
        'request_latency_histogram': ['request_latency_histogram'],
        'ttft': ['ttft', 'time_to_first_token'],
        'tpot': ['tpot', 'time_per_output_token'],
        'itl': ['itl', 'inter_token_latency'],
    }
    for name, keys in lookup.items():
        for key in keys:
            if key in parsed:
                result[name] = parsed[key]
                break
        else:
            result[name] = None
    return result


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _get_server_pid(repo_root: str) -> Optional[int]:
    pid_path = os.path.join(repo_root, 'logs', 'server.pid')
    if not os.path.exists(pid_path):
        return None
    try:
        with open(pid_path, 'r', encoding='utf-8') as f:
            pid = int(f.read().strip())
    except Exception:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _sanitize_subset_name(name: str) -> str:
    return name.replace(os.sep, '_').replace(' ', '_')


def _coerce_request_rate(value: Any) -> float:
    if value is None:
        return float('inf')
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'inf', '+inf', 'infinity', '+infinity'}:
            return float('inf')
    return float(value)


def _load_manifest_samples(config: Dict[str, Any], args: argparse.Namespace) -> List[BenchmarkSample]:
    manifest_config = config['manifest']
    subset_filter = None
    if args.subset:
        subset_filter = args.subset
    if 'subset' in config['benchmark'] and config['benchmark'].get('subset'):
        subset_filter = config['benchmark']['subset']
    return load_manifest(
        manifest_config['path'],
        audio_root=manifest_config.get('audio_root'),
        path_prefix_from=manifest_config.get('path_prefix_from'),
        path_prefix_to=manifest_config.get('path_prefix_to'),
        subset_filter=subset_filter,
        limit=config['benchmark'].get('limit'),
        seed=config['run'].get('seed') if config.get('run') else None,
    )


def _save_artifacts(output_dir: str, requests: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    _write_json(os.path.join(output_dir, 'metadata.json'), metadata)
    _write_jsonl(os.path.join(output_dir, 'requests.jsonl'), requests)
    pd.DataFrame(requests).to_csv(os.path.join(output_dir, 'requests.csv'), index=False)
    _write_json(os.path.join(output_dir, 'summary.json'), summary_rows)
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, 'summary.csv'), index=False)


def _copy_server_log(output_dir: str, repo_root: str) -> None:
    log_dir = os.path.join(repo_root, 'logs')
    source = os.path.join(log_dir, 'server.log')
    if not os.path.exists(source):
        candidates = sorted(glob.glob(os.path.join(log_dir, 'server_*.log')), key=os.path.getmtime, reverse=True)
        source = candidates[0] if candidates else source
    if os.path.exists(source):
        dest = os.path.join(output_dir, os.path.basename(source))
        with open(source, 'r', encoding='utf-8', errors='ignore') as source_file:
            with open(dest, 'w', encoding='utf-8') as dest_file:
                dest_file.write(source_file.read())
    inventory_candidates = sorted(
        glob.glob(os.path.join(log_dir, 'mtp_quant_inventory_*.json')),
        key=os.path.getmtime,
        reverse=True,
    )
    if os.path.exists(source):
        source_mtime = os.path.getmtime(source)
        inventory_candidates = [
            candidate for candidate in inventory_candidates
            if os.path.getmtime(candidate) >= source_mtime - 5.0
        ]
    if inventory_candidates:
        inventory_source = inventory_candidates[0]
        inventory_dest = os.path.join(output_dir, os.path.basename(inventory_source))
        with open(inventory_source, 'r', encoding='utf-8', errors='ignore') as source_file:
            with open(inventory_dest, 'w', encoding='utf-8') as dest_file:
                dest_file.write(source_file.read())


def _get_metrics(api_base: str, output_path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {'raw': None, 'parsed': None}
    url = f"{api_base.rstrip('/')}/metrics"
    try:
        response = httpx.get(url, timeout=30.0)
        if response.status_code == 200:
            raw = response.text
            result['raw'] = raw
            result['parsed'] = _parse_prometheus_metrics(raw)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(raw)
        else:
            result['raw'] = None
            result['parsed'] = None
    except Exception:
        result['raw'] = None
        result['parsed'] = None
    return result


def _build_request_audio_path(base_audio_dir: str, subset: str, pair_id: str, concurrency: int) -> str:
    safe_subset = _sanitize_subset_name(subset)
    dest_dir = os.path.join(base_audio_dir, safe_subset)
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, f'{pair_id}_c{concurrency}.wav')


def _logical_batch_id(sample_index: int, batch_size: int) -> int:
    normalized_batch_size = max(1, int(batch_size or 1))
    return (sample_index // normalized_batch_size) + 1


def _make_request_error_record(
    sample: BenchmarkSample,
    run_id: str,
    concurrency: int,
    batch_id: Optional[int],
    error_type: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        'run_id': run_id,
        'request_id': sample.pair_id,
        'row_index': sample.row_index,
        'subset': sample.subset,
        'pair_id': sample.pair_id,
        'concurrency': concurrency,
        'batch_id': batch_id,
        'success': False,
        'error_type': error_type,
        'error_message': error_message,
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
        'request_start_time_iso': datetime.now(timezone.utc).isoformat(),
        'request_end_time_iso': datetime.now(timezone.utc).isoformat(),
        'end_to_end_latency_sec': None,
        'time_to_first_byte_sec': None,
        'time_to_first_audio_chunk_sec': None,
        'response_format': None,
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
    }


def _start_warmup(client: TTSClient, warmup_samples: List[BenchmarkSample], run_id: str, output_base: str, concurrency: int, request_rate: float) -> None:
    if not warmup_samples:
        return
    print(f'Running warmup {len(warmup_samples)} samples at concurrency {concurrency}')
    asyncio.run(_run_requests(client, warmup_samples, run_id, output_base, concurrency, request_rate, warmup=True))


async def _run_requests(
    client: TTSClient,
    samples: List[BenchmarkSample],
    run_id: str,
    audio_root: str,
    concurrency: int,
    request_rate: float,
    warmup: bool = False,
    batch_id: Optional[int] = None,
    batch_size: int = 1,
    progress_desc: Optional[str] = None,
    records_sink: Optional[List[Dict[str, Any]]] = None,
    on_record: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = records_sink if records_sink is not None else []
    initial_record_count = len(records)
    request_interval = 1.0 / request_rate if request_rate and math.isfinite(request_rate) and request_rate > 0 else 0.0
    valid_items = [(index, sample) for index, sample in enumerate(samples) if sample.valid]
    if not valid_items:
        return []

    queue: asyncio.Queue = asyncio.Queue()
    for item in valid_items:
        queue.put_nowait(item)

    progress = None
    pacing_lock = asyncio.Lock()
    next_request_time = time.perf_counter()

    async def _pace_request_start() -> None:
        nonlocal next_request_time
        if request_interval <= 0:
            return
        async with pacing_lock:
            now = time.perf_counter()
            wait_sec = max(0.0, next_request_time - now)
            scheduled_start = max(now, next_request_time)
            next_request_time = scheduled_start + request_interval
        if wait_sec > 0:
            await asyncio.sleep(wait_sec)

    def _append_record(record: Dict[str, Any]) -> None:
        if warmup:
            return
        records.append(record)
        if on_record is not None:
            try:
                on_record(record)
            except Exception:
                pass

    worker_count = min(max(1, int(concurrency or 1)), len(valid_items))
    limits = httpx.Limits(
        max_connections=max(worker_count + 4, 8),
        max_keepalive_connections=max(worker_count, 4),
    )

    async def _request_worker(http_client: httpx.AsyncClient) -> None:
        while True:
            try:
                sample_index, sample = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            current_batch_id = batch_id if batch_id is not None else _logical_batch_id(sample_index, batch_size)
            try:
                await _pace_request_start()
                output_audio_path = None
                if not warmup and client.tts_config.get('save_audio', True):
                    output_audio_path = _build_request_audio_path(audio_root, sample.subset, sample.pair_id, concurrency)
                record = await client.request_sample(
                    sample,
                    run_id,
                    concurrency,
                    output_audio_path=output_audio_path,
                    batch_id=current_batch_id,
                    http_client=http_client,
                )
                _append_record(record.__dict__)
            except Exception as exc:
                _append_record(_make_request_error_record(
                    sample,
                    run_id,
                    concurrency,
                    current_batch_id,
                    type(exc).__name__,
                    str(exc),
                ))
            finally:
                if progress is not None:
                    progress.update(1)
                queue.task_done()

    if progress_desc and valid_items:
        progress = tqdm(
            total=len(valid_items),
            desc=progress_desc,
            unit='req',
            dynamic_ncols=True,
            file=sys.stdout,
        )
    try:
        async with httpx.AsyncClient(timeout=client.timeout_sec, limits=limits) as http_client:
            workers = [asyncio.create_task(_request_worker(http_client)) for _ in range(worker_count)]
            await asyncio.gather(*workers)
    finally:
        if progress is not None:
            progress.close()
    return records[initial_record_count:]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = merge_config_with_args(config, args)
    validate_config(config)
    os.makedirs(args.output_dir, exist_ok=True)
    output_paths = _ensure_output_dirs(args.output_dir)
    env = gather_environment_info()
    run_id = f"{config['run'].get('run_name', 'qwen3_tts_benchmark')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    manifest_samples = _load_manifest_samples(config, args)
    save_validated_manifest(manifest_samples, os.path.join(args.output_dir, 'manifest_validated.csv'))
    prompt_list_path = os.path.join(args.output_dir, f'{run_id}_prompt_list.jsonl')
    _write_prompt_list(prompt_list_path, manifest_samples)
    settings_path = _write_run_settings(
        args.output_dir,
        run_id,
        config,
        env,
        manifest_samples,
        prompt_list_path,
        ' '.join(os.sys.argv),
    )
    partial_manifest_path = os.path.join(args.output_dir, f'{run_id}_inference_manifest.partial.csv')
    final_manifest_path = os.path.join(args.output_dir, f'{run_id}_inference_manifest.csv')
    run_summary_path = os.path.join(args.output_dir, f'{run_id}_run_summary.csv')
    all_runs_summary_path = os.path.join(args.output_dir, 'all_runs_summary.csv')
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    valid_samples = [sample for sample in manifest_samples if sample.valid]
    invalid_samples = [sample for sample in manifest_samples if not sample.valid]

    invalid_records: List[Dict[str, Any]] = []
    for sample in invalid_samples:
        record = {
            'run_id': run_id,
            'request_id': sample.pair_id,
            'row_index': sample.row_index,
            'subset': sample.subset,
            'pair_id': sample.pair_id,
            'concurrency': 0,
            'batch_id': None,
            'success': False,
            'error_type': 'invalid_sample',
            'error_message': sample.error_message,
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
            'request_start_time_iso': datetime.now(timezone.utc).isoformat(),
            'request_end_time_iso': None,
            'end_to_end_latency_sec': None,
            'time_to_first_byte_sec': None,
            'time_to_first_audio_chunk_sec': None,
            'response_format': config['tts'].get('response_format'),
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
        }
        invalid_records.append(record)

    if not valid_samples and invalid_records:
        print('No valid manifest samples available for benchmark, only invalid sample records will be saved.')

    warmup_requests = config['benchmark'].get('warmup_requests', 0)
    concurrency_values = config['benchmark'].get('concurrency', [1])
    batch_size = config['benchmark'].get('batch_size', 1)
    request_rate = _coerce_request_rate(config['benchmark'].get('request_rate', float('inf')))
    repeat_per_sample = config['benchmark'].get('repeat_per_sample', 1)
    save_audio = config['benchmark'].get('save_audio', True)
    timeout_sec = config['benchmark'].get('timeout_sec', 300)
    api_base = config['server']['api_base']
    server_pid = _get_server_pid(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    subset_counts: Dict[str, int] = {}
    for sample in manifest_samples:
        subset_counts[sample.subset] = subset_counts.get(sample.subset, 0) + 1

    server_command = None
    if config['server'].get('model_id'):
        server_command = f"vllm serve {config['server'].get('model_id')} --host {config['server'].get('host', '0.0.0.0')} --port {config['server'].get('port', 8091)}"

    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config_path': os.path.abspath(args.config),
        'config': config,
        'manifest_path': config['manifest']['path'],
        'manifest_row_count': len(manifest_samples),
        'subset_counts': subset_counts,
        'api_base': api_base,
        'model_id': config['server'].get('model_id'),
        'batch_size': batch_size,
        'server_command': server_command,
        'output_dir': os.path.abspath(args.output_dir),
        'prompt_list_path': prompt_list_path,
        'settings_path': settings_path,
        'colab_compatible_partial_manifest_path': partial_manifest_path,
        'colab_compatible_final_manifest_path': final_manifest_path,
        'colab_compatible_run_summary_path': run_summary_path,
        'environment': env,
        'cli': ' '.join(os.sys.argv),
    }

    warmup_count_done = 0
    warmup_ok = 0
    if warmup_requests > 0:
        warmup_samples = valid_samples[:warmup_requests]
        warmup_count_done = len(warmup_samples)
        warmup_client = TTSClient(api_base, {**config['tts'], 'save_audio': False}, timeout_sec=timeout_sec)
        try:
            asyncio.run(_run_requests(
                warmup_client,
                warmup_samples,
                run_id,
                output_paths['audio'],
                concurrency=1,
                request_rate=request_rate,
                warmup=True,
                batch_size=batch_size,
                progress_desc='warmup',
            ))
            warmup_ok = len(warmup_samples)
        except Exception:
            error_path = os.path.join(log_dir, f'{run_id}_warmup_error.txt')
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(traceback.format_exc())

    all_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    colab_summary_rows: List[Dict[str, Any]] = []
    memory_all_samples: List[Dict[str, Any]] = []
    all_records.extend(invalid_records)
    if invalid_records:
        _write_partial_manifest(partial_manifest_path, all_records, run_id, config, settings_path, prompt_list_path)

    for concurrency in concurrency_values:
        print(f'Benchmarking with concurrency={concurrency}')
        metrics_before = _get_metrics(api_base, os.path.join(output_paths['prometheus'], f'prometheus_before_concurrency_{concurrency}.prom'))
        collector = MetricsCollector(sample_interval=config['benchmark'].get('metrics_sample_interval_sec', 0.2), process_pid=server_pid)
        collector.start()
        start_time = time.perf_counter()
        client = TTSClient(api_base, {**config['tts'], 'save_audio': save_audio}, timeout_sec=timeout_sec)
        batched_samples = []
        for _ in range(repeat_per_sample):
            batched_samples.extend(valid_samples)
        records: List[Dict[str, Any]] = []
        partial_every_n = int(config['benchmark'].get('write_partial_csv_every_n', 10) or 10)
        last_partial_count = 0

        def _on_record(_record: Dict[str, Any]) -> None:
            nonlocal last_partial_count
            if partial_every_n <= 0:
                return
            if len(records) - last_partial_count >= partial_every_n:
                _write_partial_manifest(partial_manifest_path, [*all_records, *records], run_id, config, settings_path, prompt_list_path)
                last_partial_count = len(records)

        try:
            asyncio.run(_run_requests(
                client,
                batched_samples,
                run_id,
                output_paths['audio'],
                concurrency,
                request_rate,
                warmup=False,
                batch_size=batch_size,
                progress_desc=f'c{concurrency}',
                records_sink=records,
                on_record=_on_record,
            ))
        except Exception:
            error_path = os.path.join(log_dir, f'{run_id}_c{concurrency}_error.txt')
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(traceback.format_exc())
        if records:
            _write_partial_manifest(partial_manifest_path, [*all_records, *records], run_id, config, settings_path, prompt_list_path)
        end_time = time.perf_counter()
        collector.stop()
        memory_all_samples.extend(collector.samples)
        metrics_after = _get_metrics(api_base, os.path.join(output_paths['prometheus'], f'prometheus_after_concurrency_{concurrency}.prom'))
        metrics_summary = {
            'metrics_before': metrics_before['parsed'],
            'metrics_after': metrics_after['parsed'],
            'raw_metrics_before': metrics_before['raw'],
            'raw_metrics_after': metrics_after['raw'],
            **_parse_prometheus_metrics(metrics_after['raw'] or ''),
        }
        group_summary = compute_group_summary(records, start_time, end_time, concurrency, {
            'gpu_memory_summary': collector.export_summary(),
            'prometheus_metrics_before': metrics_before['parsed'],
            'prometheus_metrics_after': metrics_after['parsed'],
        })
        runtime = _vllm_runtime_config(config)
        group_summary.update({
            'runtime_env': runtime.get('runtime_env'),
            'runtime_engine': runtime.get('runtime_engine'),
            'quant_method': runtime.get('quant_method'),
            'bit_width': runtime.get('bit_width'),
            'llm_bit': runtime.get('llm_bit'),
            'mtp_bit': runtime.get('mtp_bit'),
            'llm_quant_method': runtime.get('llm_quant_method'),
            'mtp_quant_method': runtime.get('mtp_quant_method'),
            'compute_dtype': runtime.get('compute_dtype'),
            'configured_batch_size': batch_size,
        })
        summary_rows.append(group_summary)
        colab_summary_rows.append(_write_colab_summary(
            args.output_dir,
            run_id,
            records,
            group_summary,
            config,
            settings_path,
            prompt_list_path,
            warmup_count_done,
            warmup_ok,
            end_time - start_time,
        ))
        all_records.extend(records)
        _write_partial_manifest(partial_manifest_path, all_records, run_id, config, settings_path, prompt_list_path)

    if memory_all_samples:
        memory_df = pd.DataFrame(memory_all_samples)
        memory_df.to_csv(os.path.join(args.output_dir, 'gpu_memory_timeseries.csv'), index=False)
        if not memory_df.empty:
            summary_memory = {
                'gpu_memory_peak_mb': memory_df['gpu_memory_used_mb'].dropna().max() if 'gpu_memory_used_mb' in memory_df else None,
                'gpu_memory_mean_mb': memory_df['gpu_memory_used_mb'].dropna().mean() if 'gpu_memory_used_mb' in memory_df else None,
                'gpu_memory_before_benchmark_mb': memory_df['gpu_memory_used_mb'].dropna().iloc[0] if 'gpu_memory_used_mb' in memory_df and not memory_df['gpu_memory_used_mb'].dropna().empty else None,
                'gpu_memory_after_benchmark_mb': memory_df['gpu_memory_used_mb'].dropna().iloc[-1] if 'gpu_memory_used_mb' in memory_df and not memory_df['gpu_memory_used_mb'].dropna().empty else None,
                'gpu_utilization_mean_percent': memory_df['gpu_utilization_percent'].dropna().mean() if 'gpu_utilization_percent' in memory_df else None,
                'gpu_utilization_peak_percent': memory_df['gpu_utilization_percent'].dropna().max() if 'gpu_utilization_percent' in memory_df else None,
                'process_rss_peak_mb': memory_df['process_rss_mb'].dropna().max() if 'process_rss_mb' in memory_df else None,
                'system_ram_peak_mb': memory_df['system_ram_used_mb'].dropna().max() if 'system_ram_used_mb' in memory_df else None,
            }
            _write_json(os.path.join(args.output_dir, 'memory_summary.json'), summary_memory)
        else:
            _write_json(os.path.join(args.output_dir, 'memory_summary.json'), {})
    else:
        _write_json(os.path.join(args.output_dir, 'memory_summary.json'), {})

    _save_artifacts(args.output_dir, all_records, summary_rows, metadata)
    _write_partial_manifest(final_manifest_path, all_records, run_id, config, settings_path, prompt_list_path)
    if colab_summary_rows:
        pd.DataFrame(colab_summary_rows).to_csv(run_summary_path, index=False)
        pd.DataFrame(colab_summary_rows).to_csv(all_runs_summary_path, index=False)
    _copy_server_log(args.output_dir, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f'Benchmark complete. Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
