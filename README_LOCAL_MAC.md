# Qwen3-TTS Local Mac Inference

This document explains how to run Qwen3-TTS inference **directly on a MacBook (Apple Silicon)** using a PyTorch/HuggingFace runtime — without any vLLM or HTTP server.

---

## Important differences from the RunPod / vLLM mode

| Feature | RunPod vLLM mode | Local Mac mode |
|---|---|---|
| Runtime | vLLM-Omni HTTP server | Direct PyTorch (`qwen_tts`) |
| Device | CUDA GPU | MPS (Apple Silicon) or CPU |
| Quantization | BitsAndBytes (optional) | None (BF16/FP16/FP32) |
| Flash Attention 2 | Supported | Not used (SDPA instead) |
| Benchmark scripts | `benchmark_manifest.py` | `benchmark_local_mac.py` |
| Config | `configs/qwen3_tts_base.yaml` | `configs/qwen3_tts_local_mac.yaml` |

**The existing RunPod benchmark code is not affected.** The local Mac path is a separate addition.

---

## Prerequisites

- macOS 12.3 (Monterey) or later for MPS support
- Apple Silicon Mac (M1/M2/M3/M4) recommended
- Python 3.10 or later
- ~4 GB free disk space for model cache

---

## 1. Create a virtual environment

```bash
# conda (recommended)
conda create -n qwen3_tts_mac python=3.11 -y
conda activate qwen3_tts_mac

# or venv
python3.11 -m venv ~/Workspace/envs/qwen3_tts_mac
source ~/Workspace/envs/qwen3_tts_mac/bin/activate
```

---

## 2. Install dependencies

```bash
bash scripts/setup_mac_local.sh
```

This installs:
- `torch`, `torchvision`, `torchaudio` (Apple Silicon MPS)
- `transformers`, `accelerate`, `huggingface_hub`
- `qwen-tts`
- `numpy`, `pandas`, `soundfile`, `librosa`, `scipy`, `pyyaml`, `tqdm`, `psutil`, `rich`

It does **not** install `vllm`, `vllm-omni`, `bitsandbytes`, `flash-attn`, or `flashinfer`.

---

## 3. Verify MPS

```bash
python - <<'PY'
import torch

def main():
    print("torch:", torch.__version__)
    print("mps built:", torch.backends.mps.is_built())
    print("mps available:", torch.backends.mps.is_available())

if __name__ == "__main__":
    main()
PY
```

---

## 4. Place manifest and reference audio

| What | Default path |
|---|---|
| Manifest CSV | `~/Workspace/datasets/qwen3_tts_vllm_benchmark/reference_manifest.csv` |
| Reference audio root | `~/Workspace/datasets/qwen3_tts_vllm_benchmark/` |

The manifest must have these columns:
`subset`, `pair_id`, `ref_audio_path`, `ref_audio_relpath`, `ref_duration_sec`,
`ref_text`, `target_audio_path`, `target_audio_relpath`, `target_duration_sec`,
`target_text`, `sim_audio_path`, `sim_audio_relpath`, `wer_ref_text`

---

## 5. Smoke test (1 sample)

```bash
bash scripts/run_local_mac_smoke.sh
```

Or manually:

```bash
python -m src.benchmark_local_mac \
  --config configs/qwen3_tts_local_mac.yaml \
  --output-dir ~/Workspace/results/qwen3_tts_vllm_benchmark/local_mac_smoke \
  --limit 1 \
  --batch-size 1 \
  --warmup-requests 0 \
  --save-audio
```

---

## 6. Full manifest inference

```bash
bash scripts/run_local_mac_manifest.sh
```

Or manually:

```bash
python -m src.benchmark_local_mac \
  --config configs/qwen3_tts_local_mac.yaml \
  --output-dir ~/Workspace/results/qwen3_tts_vllm_benchmark/local_mac_full \
  --batch-size 1 \
  --warmup-requests 1 \
  --save-audio
```

---

## 7. All CLI arguments

```
--config            Path to YAML config (required)
--output-dir        Output directory for this run (required)
--manifest          Override manifest CSV path
--audio-root        Override audio root directory
--limit             Limit number of samples
--batch-size        Samples per generation call (default: 1)
--warmup-requests   Number of warmup samples (default: 1)
--save-audio        Save generated WAV files (default: on)
--no-save-audio     Skip saving audio
--device            cuda / mps / cpu / auto (default: auto)
--dtype             float32 / float16 / bfloat16 / auto (default: auto)
--max-new-tokens    Token limit (default: 512)
--temperature       Sampling temperature (default: 1.0)
--top-p             Top-p sampling (default: 0.95)
--top-k             Top-k sampling (default: 50)
--do-sample         Enable sampling (default: on)
--no-do-sample      Disable sampling (greedy decoding)
--seed              Random seed
```

---

## 8. Output files

All files are saved under `--output-dir`:

```
<output-dir>/
├── manifest_validated.csv          all samples with validity flags
├── requests.csv                    per-sample results
├── requests.jsonl                  per-sample results (JSONL)
├── summary.csv                     aggregate metrics
├── summary.json                    aggregate metrics (JSON)
├── metadata.json                   run metadata and config snapshot
├── local_mac_run_settings.json     Mac-specific run settings
├── memory_summary.json             RAM usage summary
├── ram_timeseries.csv              per-second RAM samples
├── <run_id>_run_settings.json
├── <run_id>_run_summary.csv
├── <run_id>_inference_manifest.csv
└── audio/
    └── <subset>/
        └── <pair_id>_local_mac.wav
```

Model cache is stored at:
```
~/Workspace/models/qwen3_tts_vllm_benchmark/hf_cache/
```

---

## 9. Device and dtype selection

| Device | Default dtype | Notes |
|---|---|---|
| `cuda` | `bfloat16` | Standard for CUDA |
| `mps` | `float16` | Falls back to `float32` if load fails |
| `cpu` | `float32` | Safe and portable |

Override with `--device` and `--dtype`.

---

## 10. Common errors and fixes

### `qwen_tts` import fails
```
ImportError: No module named 'qwen_tts'
```
**Fix:** `pip install qwen-tts`

### MPS out of memory
```
RuntimeError: MPS backend out of memory
```
**Fix:**
- Reduce `--max-new-tokens` (e.g. `--max-new-tokens 256`)
- Set `--batch-size 1` (already the default)
- Close other GPU-using applications
- Try `--device cpu` to use system RAM instead

### dtype mismatch on MPS
```
RuntimeError: expected scalar type Float but found Half
```
**Fix:** Override to float32: `--dtype float32`

### Reference audio path not found
```
reference audio file missing
```
**Fix:**
- Verify `manifest.audio_root` in `configs/qwen3_tts_local_mac.yaml`
- Check that `ref_audio_relpath` columns in your CSV resolve correctly
- Override: `--audio-root /path/to/audio/root`

### `speech_tokenizer` config missing
```
OSError: ... speech_tokenizer/config.json not found
```
**Fix:** The setup script auto-copies `config.json` to `speech_tokenizer/`. If the model snapshot is partial, delete the cache directory and re-run to trigger a clean re-download:
```bash
rm -rf ~/Workspace/models/qwen3_tts_vllm_benchmark/hf_cache
python -m src.benchmark_local_mac --config configs/qwen3_tts_local_mac.yaml ...
```

### `librosa` keyword argument deprecation warning
This is a harmless warning from `librosa.get_duration`. Audio duration will still be computed correctly via `soundfile`.

---

## 11. Comparing results with RunPod

The output schema (`requests.csv`, `summary.csv`, etc.) is designed to match the RunPod vLLM benchmark format. Key metric columns are compatible:

- `end_to_end_latency_sec`
- `audio_duration_sec`
- `rtf`
- `latency_mean_sec` / `latency_p50_sec` / `latency_p95_sec`

GPU-specific columns (`gpu_mem_after_load_mb`, `peak_gpu_mem_mb`) will be `null` in the Mac output; Mac-specific columns (`process_ram_peak_mb`, `system_ram_used_peak_mb`, `mps_available`) will be `null` in the RunPod output.
