import csv
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False


def _safe_float(value: Optional[float]) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


class MetricsCollector:
    def __init__(self, sample_interval: float = 0.2, process_pid: Optional[int] = None, gpu_index: Optional[int] = 0):
        self.sample_interval = sample_interval
        self.process_pid = process_pid or os.getpid()
        self.gpu_index = gpu_index
        self.samples: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.process = psutil.Process(self.process_pid)
        self._nvml_initialized = False
        self.gpu_count = 0
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except Exception:
                self._nvml_initialized = False

    def start(self) -> None:
        self._stop_event.clear()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5)

    def _collect_gpu_metrics(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not self._nvml_initialized:
            return results
        for gpu_index in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power = None
                temperature = None
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power = None
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temperature = None
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8', errors='ignore')
                results.append(
                    {
                        'gpu_index': gpu_index,
                        'gpu_name': str(gpu_name),
                        'gpu_memory_used_mb': memory.used / 1024**2,
                        'gpu_memory_total_mb': memory.total / 1024**2,
                        'gpu_memory_free_mb': memory.free / 1024**2,
                        'gpu_utilization_percent': util.gpu,
                        'gpu_power_watts': power,
                        'gpu_temperature_c': temperature,
                    }
                )
            except Exception:
                continue
        return results

    def _collect_process_metrics(self) -> Dict[str, Any]:
        try:
            cpu_percent = self.process.cpu_percent(interval=None)
            memory_info = self.process.memory_info()
            return {
                'process_pid': self.process_pid,
                'process_rss_mb': memory_info.rss / 1024**2,
                'process_cpu_percent': cpu_percent,
            }
        except Exception:
            return {
                'process_pid': self.process_pid,
                'process_rss_mb': None,
                'process_cpu_percent': None,
            }

    def _collect_system_metrics(self) -> Dict[str, Any]:
        virtual = psutil.virtual_memory()
        return {
            'system_ram_used_mb': virtual.used / 1024**2,
            'system_ram_total_mb': virtual.total / 1024**2,
        }

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            timestamp = datetime.now(timezone.utc).isoformat()
            gpu_metrics = self._collect_gpu_metrics() or [
                {
                    'gpu_index': None,
                    'gpu_name': None,
                    'gpu_memory_used_mb': None,
                    'gpu_memory_total_mb': None,
                    'gpu_memory_free_mb': None,
                    'gpu_utilization_percent': None,
                    'gpu_power_watts': None,
                    'gpu_temperature_c': None,
                }
            ]
            process_metrics = self._collect_process_metrics()
            system_metrics = self._collect_system_metrics()
            for gpu in gpu_metrics:
                record = {
                    'timestamp': timestamp,
                    **gpu,
                    **process_metrics,
                    **system_metrics,
                }
                self.samples.append(record)
            time.sleep(self.sample_interval)

    def save_timeseries(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not self.samples:
            return
        keys = list(self.samples[0].keys())
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for row in self.samples:
                writer.writerow(row)

    def export_summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {
                'gpu_memory_peak_mb': None,
                'gpu_memory_mean_mb': None,
                'gpu_memory_before_benchmark_mb': None,
                'gpu_memory_after_benchmark_mb': None,
                'gpu_utilization_mean_percent': None,
                'gpu_utilization_peak_percent': None,
                'process_rss_peak_mb': None,
                'system_ram_peak_mb': None,
            }
        gpu_used = [float(_safe_float(s['gpu_memory_used_mb'])) for s in self.samples if _safe_float(s['gpu_memory_used_mb']) is not None and math.isfinite(_safe_float(s['gpu_memory_used_mb']))]
        gpu_util = [float(_safe_float(s['gpu_utilization_percent'])) for s in self.samples if _safe_float(s['gpu_utilization_percent']) is not None and math.isfinite(_safe_float(s['gpu_utilization_percent']))]
        process_rss = [float(_safe_float(s['process_rss_mb'])) for s in self.samples if _safe_float(s['process_rss_mb']) is not None and math.isfinite(_safe_float(s['process_rss_mb']))]
        system_ram = [float(_safe_float(s['system_ram_used_mb'])) for s in self.samples if _safe_float(s['system_ram_used_mb']) is not None and math.isfinite(_safe_float(s['system_ram_used_mb']))]
        return {
            'gpu_memory_peak_mb': max(gpu_used) if gpu_used else None,
            'gpu_memory_mean_mb': sum(gpu_used) / len(gpu_used) if gpu_used else None,
            'gpu_memory_before_benchmark_mb': gpu_used[0] if gpu_used else None,
            'gpu_memory_after_benchmark_mb': gpu_used[-1] if gpu_used else None,
            'gpu_utilization_mean_percent': sum(gpu_util) / len(gpu_util) if gpu_util else None,
            'gpu_utilization_peak_percent': max(gpu_util) if gpu_util else None,
            'process_rss_peak_mb': max(process_rss) if process_rss else None,
            'system_ram_peak_mb': max(system_ram) if system_ram else None,
        }

    def save_summary(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.export_summary(), f, indent=2)
