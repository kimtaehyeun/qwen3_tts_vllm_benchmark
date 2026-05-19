import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare Qwen3-TTS benchmark result directories')
    parser.add_argument('--run-dir', action='append', required=True, help='Benchmark result directory. Can be repeated.')
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def _safe_first(df: pd.DataFrame, key: str) -> Optional[Any]:
    if key not in df.columns or df.empty:
        return None
    value = df[key].iloc[0]
    if pd.isna(value):
        return None
    return value


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_run_settings(run_dir: str) -> Dict[str, Any]:
    matches = glob.glob(os.path.join(run_dir, '*_run_settings.json'))
    if not matches:
        return {}
    return _load_json(matches[0])


def _nested_get(data: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_available(values: List[Any]) -> Optional[Any]:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        return value
    return None


def _detect_label(run_dir: str, summary: pd.DataFrame) -> str:
    settings = _load_run_settings(run_dir)
    runtime = str(_safe_first(summary, 'runtime_env') or '')
    quant = str(_safe_first(summary, 'quant_method') or '')
    llm_bit = _first_available([_safe_first(summary, 'llm_bit'), _nested_get(settings, ['quantization', 'llm_bit'])])
    mtp_bit = _first_available([_safe_first(summary, 'mtp_bit'), _nested_get(settings, ['quantization', 'mtp_bit'])])
    if not runtime:
        runtime = str(
            settings.get('runtime_env')
            or _nested_get(settings, ['vllm_runtime', 'runtime_env'])
            or _load_json(os.path.join(run_dir, 'metadata.json')).get('config', {}).get('server', {}).get('executable')
            or ''
        )
    if not quant:
        quant = str(_nested_get(settings, ['quantization', 'quant_method']) or _nested_get(settings, ['vllm_runtime', 'quant_method']) or '')
    if 'torch' in runtime:
        runtime_name = 'torch'
    elif 'vllm' in runtime:
        runtime_name = 'vllm'
    else:
        runtime_name = 'unknown'
    if quant in {'none', '', 'nan'}:
        quant_name = 'bf16'
    elif str(llm_bit) == '4.0' or str(llm_bit) == '4' or str(mtp_bit) == '4.0' or str(mtp_bit) == '4':
        quant_name = 'llm4_mtp4'
    else:
        quant_name = quant
    return f'{runtime_name}_{quant_name}'


def summarize_run(run_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(run_dir, 'summary.csv')
    requests_path = os.path.join(run_dir, 'requests.csv')
    memory_path = os.path.join(run_dir, 'memory_summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    requests = pd.read_csv(requests_path) if os.path.exists(requests_path) else pd.DataFrame()
    memory = _load_json(memory_path)
    settings = _load_run_settings(run_dir)
    label = _detect_label(run_dir, summary)
    row = {
        'label': label,
        'run_dir': run_dir,
        'runtime_env': _first_available([_safe_first(summary, 'runtime_env'), settings.get('runtime_env'), _nested_get(settings, ['vllm_runtime', 'runtime_env'])]),
        'runtime_engine': _first_available([_safe_first(summary, 'runtime_engine'), settings.get('runtime_engine'), _nested_get(settings, ['vllm_runtime', 'runtime_engine'])]),
        'quant_method': _first_available([_safe_first(summary, 'quant_method'), _nested_get(settings, ['quantization', 'quant_method']), _nested_get(settings, ['vllm_runtime', 'quant_method'])]),
        'bit_width': _first_available([_safe_first(summary, 'bit_width'), _nested_get(settings, ['quantization', 'bit_width']), _nested_get(settings, ['vllm_runtime', 'bit_width'])]),
        'llm_bit': _first_available([_safe_first(summary, 'llm_bit'), _nested_get(settings, ['quantization', 'llm_bit']), _nested_get(settings, ['vllm_runtime', 'llm_bit'])]),
        'mtp_bit': _first_available([_safe_first(summary, 'mtp_bit'), _nested_get(settings, ['quantization', 'mtp_bit']), _nested_get(settings, ['vllm_runtime', 'mtp_bit'])]),
        'llm_quant_method': _first_available([_safe_first(summary, 'llm_quant_method'), _nested_get(settings, ['quantization', 'llm_quant_method']), _nested_get(settings, ['vllm_runtime', 'llm_quant_method'])]),
        'mtp_quant_method': _first_available([_safe_first(summary, 'mtp_quant_method'), _nested_get(settings, ['quantization', 'mtp_quant_method']), _nested_get(settings, ['vllm_runtime', 'mtp_quant_method'])]),
        'compute_dtype': _first_available([_safe_first(summary, 'compute_dtype'), _nested_get(settings, ['quantization', 'compute_dtype']), _nested_get(settings, ['vllm_runtime', 'compute_dtype'])]),
        'concurrency': _safe_first(summary, 'concurrency'),
        'configured_batch_size': _safe_first(summary, 'configured_batch_size'),
        'num_requests': _safe_first(summary, 'num_requests'),
        'num_success': _safe_first(summary, 'num_success'),
        'success_rate': _safe_first(summary, 'success_rate'),
        'latency_mean_sec': _safe_first(summary, 'latency_mean_sec'),
        'latency_p50_sec': _safe_first(summary, 'latency_p50_sec'),
        'latency_p95_sec': _safe_first(summary, 'latency_p95_sec'),
        'rtf_mean': _safe_first(summary, 'rtf_mean'),
        'rtf_p95': _safe_first(summary, 'rtf_p95'),
        'audio_throughput_mean_sec_per_sec': _safe_first(summary, 'audio_throughput_mean_sec_per_sec'),
        'steady_state_streaming_rtf_mean': _safe_first(summary, 'steady_state_streaming_rtf_mean'),
        'steady_state_audio_throughput_mean_sec_per_sec': _safe_first(summary, 'steady_state_audio_throughput_mean_sec_per_sec'),
        'requests_per_sec': _safe_first(summary, 'requests_per_sec'),
        'successful_requests_per_sec': _safe_first(summary, 'successful_requests_per_sec'),
        'load_time_sec': _safe_first(summary, 'load_time_sec'),
        'model_memory_footprint_mb': _safe_first(summary, 'model_memory_footprint_mb'),
        'gpu_mem_after_load_mb': _safe_first(summary, 'gpu_mem_after_load_mb'),
        'gpu_memory_peak_mb': memory.get('gpu_memory_peak_mb'),
        'gpu_memory_mean_mb': memory.get('gpu_memory_mean_mb'),
        'gpu_utilization_mean_percent': memory.get('gpu_utilization_mean_percent'),
        'gpu_utilization_peak_percent': memory.get('gpu_utilization_peak_percent'),
        'process_rss_peak_mb': memory.get('process_rss_peak_mb'),
    }
    if not requests.empty and 'audio_duration_sec' in requests.columns:
        success = requests[requests.get('success') == True]
        if not success.empty:
            row['audio_duration_total_sec_from_requests'] = success['audio_duration_sec'].dropna().sum()
    return row


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for run_dir in args.run_dir:
        rows.append(summarize_run(run_dir))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, 'comparison.csv'), index=False)
    with open(os.path.join(args.output_dir, 'comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
    print(df.to_string(index=False))
    print(f'Comparison saved to {args.output_dir}')


if __name__ == '__main__':
    main()
