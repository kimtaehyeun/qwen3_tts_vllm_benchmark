import os
from typing import Any, Dict, List, Optional


def build_server_command(config: Dict[str, Any], extra_args: Optional[List[str]] = None) -> List[str]:
    server = config.get('server', {})
    model_id = server.get('model_id')
    if model_id is None:
        raise ValueError('server.model_id is required')
    command = ['vllm', 'serve', model_id]
    host = server.get('host')
    port = server.get('port')
    if host:
        command.extend(['--host', str(host)])
    if port is not None:
        command.extend(['--port', str(port)])
    gpu_memory_utilization = server.get('gpu_memory_utilization')
    if gpu_memory_utilization is not None:
        command.extend(['--gpu-memory-utilization', str(gpu_memory_utilization)])
    tensor_parallel_size = server.get('tensor_parallel_size')
    if tensor_parallel_size is not None:
        command.extend(['--tensor-parallel-size', str(tensor_parallel_size)])
    max_num_seqs = server.get('max_num_seqs')
    if max_num_seqs is not None:
        command.extend(['--max-num-seqs', str(max_num_seqs)])
    max_model_len = server.get('max_model_len')
    if max_model_len is not None:
        command.extend(['--max-model-len', str(max_model_len)])
    if server.get('trust_remote_code'):
        command.append('--trust-remote-code')
    stage_configs_path = server.get('stage_configs_path')
    if stage_configs_path:
        command.extend(['--stage-configs-path', str(stage_configs_path)])
    extra = server.get('extra_vllm_args') or []
    for arg in extra:
        command.append(str(arg))
    if extra_args:
        command.extend(extra_args)
    return command


def resolve_runpod_path(original_path: str, path_prefix_from: Optional[str], path_prefix_to: Optional[str]) -> str:
    if path_prefix_from and path_prefix_to and original_path.startswith(path_prefix_from):
        return original_path.replace(path_prefix_from, path_prefix_to, 1)
    return original_path
