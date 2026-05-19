import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from src.audio_utils import get_audio_duration, get_audio_duration_from_bytes, save_audio_bytes


def _finite_float(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float('inf'), float('-inf')):
        return None
    return number


def _safe_divide(numerator: Optional[Any], denominator: Optional[Any]) -> Optional[float]:
    numerator_value = _finite_float(numerator)
    denominator_value = _finite_float(denominator)
    if numerator_value is None or denominator_value is None or denominator_value == 0:
        return None
    return numerator_value / denominator_value


def compute_steady_state_audio_metrics(
    success: bool,
    end_to_end_latency_sec: Optional[float],
    time_to_first_audio_chunk_sec: Optional[float],
    audio_duration_sec: Optional[float],
) -> Dict[str, Optional[Any]]:
    result: Dict[str, Optional[Any]] = {
        'post_first_audio_chunk_latency_sec': None,
        'steady_state_audio_throughput_sec_per_sec': None,
        'steady_state_streaming_rtf': None,
        'steady_state_metric_status': None,
    }
    if not success:
        result['steady_state_metric_status'] = 'request_failed'
        return result

    end_to_end_latency = _finite_float(end_to_end_latency_sec)
    first_audio_chunk = _finite_float(time_to_first_audio_chunk_sec)
    audio_duration = _finite_float(audio_duration_sec)

    if first_audio_chunk is None:
        result['steady_state_metric_status'] = 'missing_time_to_first_audio_chunk'
        return result
    if audio_duration is None or audio_duration <= 0:
        result['steady_state_metric_status'] = 'invalid_audio_duration'
        return result
    if end_to_end_latency is None or end_to_end_latency <= first_audio_chunk:
        result['steady_state_metric_status'] = 'invalid_latency_range'
        return result

    post_first_audio_chunk_latency = end_to_end_latency - first_audio_chunk
    result['post_first_audio_chunk_latency_sec'] = post_first_audio_chunk_latency
    result['steady_state_audio_throughput_sec_per_sec'] = audio_duration / post_first_audio_chunk_latency
    result['steady_state_streaming_rtf'] = post_first_audio_chunk_latency / audio_duration
    result['steady_state_metric_status'] = 'ok'
    return result


@dataclass
class RequestRecord:
    run_id: str
    request_id: str
    row_index: int
    subset: str
    pair_id: str
    concurrency: int
    batch_id: Optional[int]
    success: bool
    error_type: Optional[str]
    error_message: Optional[str]
    http_status_code: Optional[int]
    ref_audio_path: Optional[str]
    ref_audio_relpath: Optional[str]
    resolved_ref_audio_path: Optional[str]
    ref_duration_sec: Optional[float]
    ref_text: Optional[str]
    target_text: Optional[str]
    target_text_len: int
    target_audio_path: Optional[str]
    target_audio_relpath: Optional[str]
    target_duration_sec: Optional[float]
    request_start_time_iso: str
    request_end_time_iso: Optional[str]
    end_to_end_latency_sec: Optional[float]
    time_to_first_byte_sec: Optional[float]
    time_to_first_audio_chunk_sec: Optional[float]
    response_format: Optional[str]
    output_audio_path: Optional[str]
    output_audio_bytes: Optional[int]
    audio_duration_sec: Optional[float]
    rtf: Optional[float]
    end_to_end_rtf: Optional[float]
    audio_throughput_sec_per_sec: Optional[float]
    end_to_end_audio_throughput_sec_per_sec: Optional[float]
    post_first_audio_chunk_latency_sec: Optional[float]
    steady_state_audio_throughput_sec_per_sec: Optional[float]
    steady_state_streaming_rtf: Optional[float]
    steady_state_metric_status: Optional[str]
    audio_bytes_per_sec: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    tokens_per_sec: Optional[float]
    output_tokens_per_sec: Optional[float]
    end_to_end_output_tokens_per_sec: Optional[float]
    token_metric_status: Optional[str]


class TTSClient:
    def __init__(self, api_base: str, tts_config: Dict[str, Any], timeout_sec: int = 300):
        self.api_base = api_base.rstrip('/')
        self.tts_config = tts_config
        self.timeout_sec = timeout_sec
        self.endpoint = tts_config.get('endpoint', '/v1/audio/speech')
        self.request_mode = tts_config.get('request_mode', 'multipart_file')
        self.payload_mode = tts_config.get('payload_mode', 'qwen3_tts_base')
        self.response_format = tts_config.get('response_format', 'wav')
        self.sample_rate = tts_config.get('sample_rate', 24000)
        self.stream = tts_config.get('stream', False)

    def _build_json_payload(self, sample: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.payload_mode in ('qwen3_tts_base', 'qwen3_tts_customvoice', 'qwen3_tts_voicedesign'):
            payload['input'] = sample.target_text
            payload['ref_text'] = sample.ref_text
            if self.request_mode == 'json_with_path':
                ref_audio = sample.resolved_ref_audio_path
                if ref_audio and not ref_audio.startswith(('http://', 'https://', 'data:', 'file://')):
                    ref_audio = 'file://' + os.path.abspath(ref_audio)
                payload['ref_audio'] = ref_audio
            payload['response_format'] = self.response_format
            payload['sample_rate'] = self.sample_rate
            payload['stream'] = self.stream
            if self.tts_config.get('task_type'):
                payload['task_type'] = self.tts_config['task_type']
            if self.tts_config.get('language'):
                payload['language'] = self.tts_config['language']
            if self.tts_config.get('voice'):
                payload['voice'] = self.tts_config['voice']
            if self.tts_config.get('instructions'):
                payload['instructions'] = self.tts_config['instructions']
        elif self.payload_mode == 'openai_audio_speech_compatible':
            payload['input'] = sample.target_text
            payload['response_format'] = self.response_format
            payload['sample_rate'] = self.sample_rate
            if self.tts_config.get('voice'):
                payload['voice'] = self.tts_config['voice']
        else:
            payload['input'] = sample.target_text
            payload['response_format'] = self.response_format
        return payload

    async def _fetch_response(self, client: httpx.AsyncClient, url: str, data: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start = time.perf_counter()
        time_to_first_byte = None
        time_to_first_chunk = None
        output_bytes = b''
        status_code = None
        headers = None
        response_json: Optional[Dict[str, Any]] = None
        error_message = None

        async with client.stream('POST', url, json=data if files is None else None, data=None if files is None else data, files=files, timeout=self.timeout_sec) as response:
            status_code = response.status_code
            headers = response.headers
            first_chunk_time = None
            async for chunk in response.aiter_bytes():
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                    time_to_first_chunk = first_chunk_time - start
                    time_to_first_byte = time_to_first_chunk
                output_bytes += chunk
            if headers and headers.get('content-type', '').startswith('application/json'):
                try:
                    response_json = json.loads(output_bytes.decode('utf-8', errors='ignore'))
                except Exception:
                    response_json = None
        if status_code is None:
            raise RuntimeError('No status code from response stream')
        return {
            'status_code': status_code,
            'headers': headers,
            'body': output_bytes,
            'json': response_json,
            'time_to_first_byte_sec': time_to_first_byte,
            'time_to_first_audio_chunk_sec': time_to_first_chunk,
        }

    def _extract_token_metrics(self, response_json: Optional[Dict[str, Any]]) -> Dict[str, Optional[Any]]:
        result = {
            'prompt_tokens': None,
            'completion_tokens': None,
            'total_tokens': None,
            'tokens_per_sec': None,
            'output_tokens_per_sec': None,
            'token_metric_status': None,
        }
        if response_json is None:
            result['token_metric_status'] = 'unavailable_from_endpoint_or_metrics'
            return result
        usage = response_json.get('usage') if isinstance(response_json, dict) else None
        if usage:
            result['prompt_tokens'] = usage.get('prompt_tokens')
            result['completion_tokens'] = usage.get('completion_tokens')
            result['total_tokens'] = usage.get('total_tokens')
            result['token_metric_status'] = 'from_endpoint_usage'
        else:
            result['token_metric_status'] = 'unavailable_from_endpoint_or_metrics'
        return result

    async def request_sample(self, sample: Any, run_id: str, concurrency: int, output_audio_path: Optional[str] = None, batch_id: Optional[int] = None) -> RequestRecord:
        request_id = sample.pair_id
        request_start = time.perf_counter()
        start_time_iso = datetime.now(timezone.utc).isoformat()
        record = RequestRecord(
            run_id=run_id,
            request_id=request_id,
            row_index=sample.row_index,
            subset=sample.subset,
            pair_id=sample.pair_id,
            concurrency=concurrency,
            batch_id=batch_id,
            success=False,
            error_type=None,
            error_message=None,
            http_status_code=None,
            ref_audio_path=sample.ref_audio_path,
            ref_audio_relpath=sample.ref_audio_relpath,
            resolved_ref_audio_path=sample.resolved_ref_audio_path,
            ref_duration_sec=sample.ref_duration_sec,
            ref_text=sample.ref_text,
            target_text=sample.target_text,
            target_text_len=len(sample.target_text or ''),
            target_audio_path=sample.target_audio_path,
            target_audio_relpath=sample.target_audio_relpath,
            target_duration_sec=sample.target_duration_sec,
            request_start_time_iso=start_time_iso,
            request_end_time_iso=None,
            end_to_end_latency_sec=None,
            time_to_first_byte_sec=None,
            time_to_first_audio_chunk_sec=None,
            response_format=self.response_format,
            output_audio_path=output_audio_path,
            output_audio_bytes=None,
            audio_duration_sec=None,
            rtf=None,
            end_to_end_rtf=None,
            audio_throughput_sec_per_sec=None,
            end_to_end_audio_throughput_sec_per_sec=None,
            post_first_audio_chunk_latency_sec=None,
            steady_state_audio_throughput_sec_per_sec=None,
            steady_state_streaming_rtf=None,
            steady_state_metric_status='request_failed',
            audio_bytes_per_sec=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            tokens_per_sec=None,
            output_tokens_per_sec=None,
            end_to_end_output_tokens_per_sec=None,
            token_metric_status=None,
        )

        if sample.resolved_ref_audio_path is None:
            record.error_type = 'missing_input'
            record.error_message = 'resolved_ref_audio_path is missing'
            return record

        url = f"{self.api_base.rstrip('/')}{self.endpoint}"
        payload = self._build_json_payload(sample)
        files = None
        data = None

        if self.request_mode == 'multipart_file':
            files = {}
            if self.payload_mode in ('qwen3_tts_base', 'qwen3_tts_customvoice', 'qwen3_tts_voicedesign'):
                data = payload
                files['ref_audio'] = ('ref_audio.wav', open(sample.resolved_ref_audio_path, 'rb'), 'audio/wav')
            elif self.payload_mode == 'openai_audio_speech_compatible':
                data = payload
                files['file'] = ('ref_audio.wav', open(sample.resolved_ref_audio_path, 'rb'), 'audio/wav')
            else:
                data = payload
                files['ref_audio'] = ('ref_audio.wav', open(sample.resolved_ref_audio_path, 'rb'), 'audio/wav')
        else:
            data = payload

        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            try:
                response = await self._fetch_response(client, url, data=data, files=files)
            except Exception as exc:
                record.error_type = type(exc).__name__
                record.error_message = str(exc)
                return record
            finally:
                if files:
                    for fileobj in files.values():
                        try:
                            fileobj[1].close()
                        except Exception:
                            pass

        record.http_status_code = response['status_code']
        record.time_to_first_byte_sec = response.get('time_to_first_byte_sec')
        record.time_to_first_audio_chunk_sec = response.get('time_to_first_audio_chunk_sec')

        if response['status_code'] >= 400:
            record.error_type = 'http_error'
            record.error_message = response['body'].decode('utf-8', errors='ignore')[:1024]
            return record

        output_bytes = response['body']
        if not output_bytes:
            record.error_type = 'empty_audio'
            record.error_message = 'response body contained no audio'
            return record

        record.output_audio_bytes = len(output_bytes)
        if output_audio_path:
            try:
                save_audio_bytes(output_audio_path, output_bytes)
                record.output_audio_path = output_audio_path
            except Exception as exc:
                record.error_type = 'save_audio_error'
                record.error_message = str(exc)
                return record

        record.success = True
        record.request_end_time_iso = datetime.now(timezone.utc).isoformat()
        record.end_to_end_latency_sec = time.perf_counter() - request_start

        record.audio_duration_sec = None
        if output_audio_path:
            record.audio_duration_sec = get_audio_duration(output_audio_path)
        if record.audio_duration_sec is None:
            record.audio_duration_sec = get_audio_duration_from_bytes(output_bytes)

        if record.audio_duration_sec is not None and record.audio_duration_sec > 0:
            record.rtf = record.end_to_end_latency_sec / record.audio_duration_sec
            record.end_to_end_rtf = record.rtf
            record.audio_throughput_sec_per_sec = record.audio_duration_sec / record.end_to_end_latency_sec if record.end_to_end_latency_sec else None
            record.end_to_end_audio_throughput_sec_per_sec = record.audio_throughput_sec_per_sec
            record.audio_bytes_per_sec = record.output_audio_bytes / record.end_to_end_latency_sec if record.end_to_end_latency_sec else None
        else:
            record.rtf = None
            record.end_to_end_rtf = None
            record.audio_throughput_sec_per_sec = None
            record.end_to_end_audio_throughput_sec_per_sec = None
            record.audio_bytes_per_sec = None

        steady_state_metrics = compute_steady_state_audio_metrics(
            record.success,
            record.end_to_end_latency_sec,
            record.time_to_first_audio_chunk_sec,
            record.audio_duration_sec,
        )
        record.post_first_audio_chunk_latency_sec = steady_state_metrics['post_first_audio_chunk_latency_sec']
        record.steady_state_audio_throughput_sec_per_sec = steady_state_metrics['steady_state_audio_throughput_sec_per_sec']
        record.steady_state_streaming_rtf = steady_state_metrics['steady_state_streaming_rtf']
        record.steady_state_metric_status = steady_state_metrics['steady_state_metric_status']

        token_metrics = self._extract_token_metrics(response['json'])
        record.prompt_tokens = token_metrics['prompt_tokens']
        record.completion_tokens = token_metrics['completion_tokens']
        record.total_tokens = token_metrics['total_tokens']
        record.token_metric_status = token_metrics['token_metric_status']

        if record.total_tokens is not None and record.end_to_end_latency_sec and record.end_to_end_latency_sec > 0:
            record.tokens_per_sec = record.total_tokens / record.end_to_end_latency_sec
        if record.completion_tokens is not None and record.end_to_end_latency_sec and record.end_to_end_latency_sec > 0:
            record.output_tokens_per_sec = record.completion_tokens / record.end_to_end_latency_sec
            record.end_to_end_output_tokens_per_sec = record.output_tokens_per_sec

        return record
