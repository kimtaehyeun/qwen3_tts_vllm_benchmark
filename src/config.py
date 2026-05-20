import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RunConfig:
    run_name: str
    seed: Optional[int] = None


@dataclass
class ServerConfig:
    model_id: str
    api_base: str
    host: str = '0.0.0.0'
    port: int = 8091
    gpu_memory_utilization: Optional[float] = None
    tensor_parallel_size: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_model_len: Optional[int] = None
    trust_remote_code: bool = True
    stage_configs_path: Optional[str] = None
    extra_vllm_args: List[str] = None


@dataclass
class TTSConfig:
    endpoint: str
    task_type: Optional[str] = None
    payload_mode: str = 'qwen3_tts_base'
    request_mode: str = 'multipart_file'
    response_format: str = 'wav'
    sample_rate: int = 24000
    stream: bool = False
    language: Optional[str] = None
    voice: Optional[str] = None
    instructions: Optional[str] = None


@dataclass
class ManifestConfig:
    path: str
    audio_root: Optional[str] = None
    path_prefix_from: Optional[str] = None
    path_prefix_to: Optional[str] = None


@dataclass
class BenchmarkConfig:
    warmup_requests: int = 0
    concurrency: List[int] = None
    batch_size: int = 1
    request_rate: float = float('inf')
    timeout_sec: int = 300
    save_audio: bool = True
    metrics_sample_interval_sec: float = 0.5
    repeat_per_sample: int = 1


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError('Config file is empty')
    return config


def validate_config(config: Dict[str, Any]) -> None:
    required_sections = ['run', 'server', 'tts', 'manifest', 'benchmark']
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f'Missing config sections: {missing}')
    if 'model_id' not in config['server']:
        raise ValueError('server.model_id is required')
    if 'api_base' not in config['server']:
        raise ValueError('server.api_base is required')
    if 'endpoint' not in config['tts']:
        raise ValueError('tts.endpoint is required')
    if 'path' not in config['manifest']:
        raise ValueError('manifest.path is required')
    if 'concurrency' not in config['benchmark']:
        raise ValueError('benchmark.concurrency is required')


def merge_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = {**config}
    for category, values in overrides.items():
        if values is None:
            continue
        result.setdefault(category, {})
        if isinstance(values, dict):
            result[category].update({k: v for k, v in values.items() if v is not None})
        else:
            result[category] = values
    return result
