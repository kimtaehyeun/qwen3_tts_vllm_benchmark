import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp


@dataclass
class StreamingTTSResult:
    status_code: int
    headers: Dict[str, Any]
    response_body: bytes
    time_to_first_byte_sec: Optional[float]
    time_to_first_audio_chunk_sec: Optional[float]
    json_payload: Optional[Dict[str, Any]]


class StreamingTTSClient:
    def __init__(self, api_base: str, endpoint: str, timeout_sec: int = 300):
        self.api_base = api_base.rstrip('/')
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec

    def _build_request_data(self, payload: Dict[str, Any], files: Optional[Dict[str, Any]]) -> Any:
        if not files:
            return payload

        form = aiohttp.FormData()
        for key, value in payload.items():
            form.add_field(key, str(value))
        for key, value in files.items():
            if isinstance(value, tuple):
                filename = value[0]
                fileobj = value[1]
                content_type = value[2] if len(value) > 2 else 'application/octet-stream'
                form.add_field(key, fileobj, filename=filename, content_type=content_type)
            else:
                form.add_field(key, value)
        return form

    async def request_stream(self, payload: Dict[str, Any], files: Optional[Dict[str, Any]] = None) -> StreamingTTSResult:
        url = f"{self.api_base}{self.endpoint}"
        data = self._build_request_data(payload, files)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_sec)) as session:
            start = time.perf_counter()
            async with session.post(url, data=data) as response:
                first_chunk_time = None
                body = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter() - start
                    body.extend(chunk)
                json_payload = None
                if response.headers.get('Content-Type', '').startswith('application/json'):
                    try:
                        json_payload = json.loads(body.decode('utf-8', errors='ignore'))
                    except Exception:
                        json_payload = None
                return StreamingTTSResult(
                    status_code=response.status,
                    headers=dict(response.headers),
                    response_body=bytes(body),
                    time_to_first_byte_sec=first_chunk_time,
                    time_to_first_audio_chunk_sec=first_chunk_time,
                    json_payload=json_payload,
                )
