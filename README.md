# Qwen3-TTS vLLM Benchmark

A benchmark suite for Qwen3-TTS inference on RunPod using vLLM or vLLM-Omni runtime.

This repository currently targets `Qwen/Qwen3-TTS-12Hz-1.7B-Base` with a vLLM-Omni OpenAI-compatible server. The benchmark client sends requests to `/v1/audio/speech`, records latency/RTF/GPU metrics, and writes Colab-compatible run artifacts for comparison with the previous PyTorch/Hugging Face runtime notebook.

## Repository structure

- `configs/` - YAML example configurations
- `scripts/` - install / launch / stop / health check scripts
- `src/` - benchmark implementation and utilities
- `results/` - benchmark output

## Runtime difference from the Colab notebook

The previous Colab notebook used `qwen_tts.Qwen3TTSModel` directly with PyTorch/Hugging Face runtime. This repo now uses a persistent vLLM-Omni server:

```text
benchmark_manifest.py
  -> http://127.0.0.1:8091/v1/audio/speech
  -> vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni
  -> Qwen3-TTS inference
```

Key differences:

- Runtime: Colab used direct PyTorch/HF calls; this repo uses `vllm-omni serve`.
- Attention: Colab set `attn_implementation="flash_attention_2"`; vLLM logs show `FLASH_ATTN` and `Using FlashAttention version 2`.
- dtype: both use BF16 for the current L4/Ampere-style setup.
- Quantization: Colab had low-bit plans (`bnb_int8`, `bnb_nf4`, `hqq_3bit/2bit`). The default vLLM setup is BF16. A BitsAndBytes 4bit vLLM config is included for stage 0, the Qwen3-TTS talker LLM plus MTP body, with stage 1 left in the default dtype for audio decoding.
- Batch behavior: Colab used HF batch generation; this benchmark sends individual HTTP requests and controls parallelism with `--concurrency`.
- Logging: the benchmark now writes both original benchmark files and Colab-compatible files.

## Install dependencies

```bash
bash scripts/install_runpod.sh
```

## Launch vLLM server

```bash
bash scripts/launch_server.sh configs/qwen3_tts_base.yaml
```

Follow the latest server log until the API is ready:

```bash
tail -f "$(ls -t logs/server_*.log | head -n1)"
```

## Health check

```bash
bash scripts/health_check.sh http://127.0.0.1:8091
```

Expected checks:

- `/v1/models` returns `owned_by: "vllm"`
- `/health` returns `200 OK`
- `/metrics` returns Prometheus metrics

## vLLM 4bit BitsAndBytes config

The repository includes a BitsAndBytes 4bit configuration for the vLLM-Omni runtime:

- `configs/qwen3_tts_bnb4.yaml` - benchmark/server config that records `quant_method=bitsandbytes`, `llm_bit=4`, `mtp_bit=4`, and `mtp_quant_method=int4_weightonly`
- `configs/qwen3_tts_stage0_bnb4.yaml` - vLLM-Omni stage config with BitsAndBytes applied to stage 0, the Qwen3-TTS talker LLM
- MTP/code predictor body linear layers are converted to a custom `int4_weightonly` linear path. This avoids the BitsAndBytes `Linear4bit` MTP path, which started but timed out during generation in this environment.
- Stage 1, `Qwen3TTSCode2Wav`, is intentionally left unquantized because it is the audio decoder path

This environment needs a compatibility patch because upstream vLLM-Omni does not currently expose `packed_modules_mapping` on `Qwen3TTSTalkerForConditionalGeneration`, vLLM's BitsAndBytes loader can otherwise match nested `code_predictor.model.layers` weights too broadly, and the Qwen3-TTS MTP/code predictor is implemented with plain PyTorch `nn.Linear` layers rather than vLLM quantized linear layers. Apply the patch after installing dependencies or after reinstalling `vllm` / `vllm-omni`:

```bash
bash scripts/patch_vllm_qwen3_tts_bnb4.sh
```

Then launch the 4bit server:

```bash
bash scripts/launch_server.sh configs/qwen3_tts_bnb4.yaml
```

Confirm the server is using BitsAndBytes:

```bash
tail -f "$(ls -t logs/server_*.log | head -n1)"
```

Expected log markers:

```text
load_format=bitsandbytes
quantization=bitsandbytes
Loading weights with BitsAndBytes quantization
code_predictor: int4_weightonly enabled for 36 MTP Linear modules
Using FlashAttention version 2
Application startup complete
```

Current behavior observed for LLM BitsAndBytes 4bit + MTP `int4_weightonly`:

- Server startup succeeds.
- `/health` and `/v1/models` succeed.
- Logs confirm LLM BitsAndBytes and MTP `int4_weightonly`.
- A direct short-text request succeeded with HTTP 200 in about 9.3s.
- A one-sample manifest smoke run succeeded with HTTP 200 in about 26.1s, `rtf_mean=1.61`.

Use this smoke command to reproduce the current LLM+MTP 4bit behavior:

```bash
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_bnb4.yaml \
  --output-dir results/bnb4_llm_mtp_int4_smoke \
  --warmup-requests 0 \
  --concurrency 1 \
  --batch-size 1 \
  --limit 1 \
  --save-audio \
  --response-format wav
```

## Run benchmark on 300 reference manifest samples

The checked-in reference manifest has 300 samples. If `--limit` is omitted, all valid manifest rows are used.

```bash
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base.yaml \
  --output-dir results/reference_all_c1 \
  --warmup-requests 5 \
  --concurrency 1 \
  --save-audio \
  --response-format wav
```

## Parallel inference

`--concurrency` controls how many inference requests can run at the same time.

`--batch-size` controls how many samples the benchmark schedules into one async group. In this client it is not a single GPU tensor batch. To make concurrency effective across all 300 samples, use `--batch-size 300`.

Examples:

```bash
# Run all 300 samples, max 2 concurrent requests.
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base.yaml \
  --output-dir results/reference_all_parallel_c2 \
  --warmup-requests 5 \
  --concurrency 2 \
  --batch-size 300 \
  --save-audio \
  --response-format wav
```

```bash
# Run all 300 samples, max 8 concurrent requests.
# Current server config uses max_num_seqs: 8, so this is a practical first high-throughput test.
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base.yaml \
  --output-dir results/reference_all_parallel_c8 \
  --warmup-requests 5 \
  --concurrency 8 \
  --batch-size 300 \
  --save-audio \
  --response-format wav
```

Behavior:

```text
--concurrency 2 --batch-size 300
  -> schedules 300 samples
  -> keeps up to 2 requests in flight
  -> starts the next sample as soon as one finishes
```

If `--batch-size 1`, each group contains only one sample, so `--concurrency 2` or higher has little practical effect.

## Smoke test

```bash
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base.yaml \
  --output-dir results/smoke_test \
  --warmup-requests 2 \
  --concurrency 1 \
  --limit 5 \
  --save-audio
```

## Progress and logs

The benchmark prints a `tqdm` progress bar for warmup and each concurrency/batch group. To append benchmark progress into the latest server log:

```bash
SERVER_LOG="$(ls -t logs/server_*.log | head -n1)"

python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base.yaml \
  --output-dir results/reference_all_parallel_c8 \
  --warmup-requests 5 \
  --concurrency 8 \
  --batch-size 300 \
  --save-audio \
  --response-format wav \
  2>&1 | tee -a "$SERVER_LOG"
```

Then watch it from another shell:

```bash
tail -f "$SERVER_LOG"
```

Note: the vLLM server itself does not know the full manifest progress. The benchmark client knows the 300-sample progress and can append it into the same log file with `tee`.

## Output files

The benchmark writes the original benchmark artifacts:

- `manifest_validated.csv` - validated manifest rows and resolved audio paths
- `requests.csv` / `requests.jsonl` - per-request latency, status, audio path, RTF, byte metrics
- `summary.csv` / `summary.json` - aggregate metrics per concurrency value
- `gpu_memory_timeseries.csv` - sampled GPU/process/system memory and utilization
- `memory_summary.json` - peak/mean memory and utilization summary
- `prometheus/*.prom` - vLLM Prometheus metrics before/after each concurrency run
- `metadata.json` - config, environment, CLI, output paths

It also writes Colab-compatible artifacts inspired by `Qwen3_TTS_Colab_2Session_lowBit (1).ipynb`:

- `<run_id>_prompt_list.jsonl` - prompt/ref list for reproducibility
- `<run_id>_run_settings.json` - run settings, hardware/software info, vLLM runtime config
- `<run_id>_inference_manifest.partial.csv` - partial manifest, updated during the run
- `<run_id>_inference_manifest.csv` - final Colab-style inference manifest
- `<run_id>_run_summary.csv` - Colab-style per-run summary
- `all_runs_summary.csv` - combined Colab-style summary rows
- `logs/<run_id>_..._error.txt` - warmup or batch traceback files if a batch-level exception occurs

The Colab-compatible manifest includes fields such as:

```text
run_name, model_id, model_tag, quant_method, bit_width,
compute_dtype, attn_implementation, runtime_env, runtime_engine,
configured_batch_size, batch_index, generation_language,
batch_mode, status, gen_sec, audio_sec, gen_rtf,
prompt_list_path, run_settings_path
```

### Streaming steady-state metrics

The per-request files (`requests.csv`, `requests.jsonl`, and the Colab-compatible inference manifests) keep the existing `rtf` and `audio_throughput_sec_per_sec` columns for backward compatibility, and also add explicit end-to-end aliases:

- `end_to_end_rtf`: RTF from request start until the full audio response is complete.
- `end_to_end_audio_throughput_sec_per_sec`: generated audio seconds per wall-clock second over the full end-to-end request.

Streaming-oriented steady-state metrics are calculated only when the request succeeds and `time_to_first_audio_chunk_sec`, `end_to_end_latency_sec`, and a positive `audio_duration_sec` are available:

- `post_first_audio_chunk_latency_sec`: time from first audio chunk until response completion.
- `steady_state_streaming_rtf`: streaming RTF after the first audio chunk, excluding initial latency.
- `steady_state_audio_throughput_sec_per_sec`: generated audio seconds per second after the first audio chunk. Values greater than `1.0` mean audio generation is faster than real time.
- `steady_state_metric_status`: `ok`, `missing_time_to_first_audio_chunk`, `invalid_audio_duration`, `invalid_latency_range`, or `request_failed`.

The OpenAI-compatible `/v1/audio/speech` endpoint usually returns audio bytes rather than a token usage JSON payload. In that case token/s fields remain empty and `token_metric_status` is recorded as `unavailable_from_endpoint_or_metrics`. Token throughput is calculated only when token counts are actually present in endpoint metadata or server metrics.

For the default BF16 vLLM run, quantization fields are recorded as `none`/`bf16`. For `configs/qwen3_tts_bnb4.yaml`, run settings and summary fields record `llm_quant_method=bitsandbytes`, `mtp_quant_method=int4_weightonly`, `llm_bit=4`, and `mtp_bit=4`. HF direct-load fields such as `load_time_sec` and `model_memory_footprint_mb` are not directly available from the benchmark client, so they are left empty while vLLM/GPU metrics are recorded separately.

## Summarize results

```bash
python -m src.summarize_results --run-dir results/reference_all_c1
```

## Four-Way Runtime Comparison

To compare the four target conditions with the same manifest and output schema:

1. Torch/Hugging Face direct runtime, BF16 baseline
2. vLLM-Omni runtime, BF16 baseline
3. Torch/Hugging Face direct runtime, LLM 4bit + MTP 4bit
4. vLLM-Omni runtime, BF16 base dtype with LLM 4bit + MTP 4bit

Install the direct torch dependency if it is not already available:

```bash
pip install -r requirements.txt
```

Then run the full comparison:

```bash
LIMIT=300 WARMUP=5 CONCURRENCY=1 TORCH_BATCH_SIZE=1 VLLM_BATCH_SIZE=1 \
  bash scripts/run_4way_benchmark.sh
```

The script writes one result directory per condition:

```text
results/fourway_<timestamp>/
  01_torch_bf16/
  02_vllm_bf16/
  03_torch_llm4_mtp4/
  04_vllm_llm4_mtp4/
  comparison/comparison.csv
  comparison/comparison.json
```

For quick smoke tests, lower `LIMIT` and `WARMUP`:

```bash
LIMIT=1 WARMUP=0 CONCURRENCY=1 bash scripts/run_4way_benchmark.sh
```

The comparison table includes latency, RTF, audio throughput, steady-state streaming metrics when available, success rate, GPU memory peak/mean, GPU utilization, load time, and model memory footprint. Torch direct runtime does not expose a real first audio chunk timestamp, so its steady-state streaming metrics remain unavailable with `steady_state_metric_status=missing_time_to_first_audio_chunk`.

For torch direct runtime, `--attn-implementation auto` is the default: it uses `flash_attention_2` only when the `flash_attn` package is installed, otherwise it falls back to `sdpa`. If you want to force the official FlashAttention2 path, install `flash-attn` first and pass `--attn-implementation flash_attention_2` to `python -m src.benchmark_torch`.

## Stop server

```bash
bash scripts/stop_server.sh
```
