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
- Quantization: Colab had low-bit plans (`bnb_int8`, `bnb_nf4`, `hqq_3bit/2bit`). The default vLLM setup is BF16. BitsAndBytes 4bit vLLM configs are included for stage 0, the Qwen3-TTS talker LLM plus MTP body, with stage 1 left in the default dtype for audio decoding.
- Batch behavior: Colab used HF batch generation; this benchmark sends individual HTTP requests and controls parallelism with `--concurrency`.
- Logging: the benchmark now writes both original benchmark files and Colab-compatible files.

## Install dependencies

```bash
bash scripts/install_runpod.sh
```

## Launch vLLM server

```bash
bash scripts/launch_server.sh configs/qwen3_tts_base_optimized.yaml
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

The repository includes final optimized BitsAndBytes 4bit configurations for the vLLM-Omni runtime:

- `configs/qwen3_tts_bnb4_optimized.yaml` - final benchmark/server config that records `quant_method=bitsandbytes`, `llm_bit=4`, `mtp_bit=4`, and `mtp_quant_method=bitsandbytes`
- `configs/qwen3_tts_stage0_bnb4_optimized.yaml` - final vLLM-Omni stage config with BitsAndBytes applied to stage 0, the Qwen3-TTS talker LLM plus MTP/code predictor linear modules
- Stage 1, `Qwen3TTSCode2Wav`, is intentionally left unquantized because it is the audio decoder path

This environment needs a compatibility patch because upstream vLLM-Omni does not currently expose `packed_modules_mapping` on `Qwen3TTSTalkerForConditionalGeneration`, vLLM's BitsAndBytes loader can otherwise match nested `code_predictor.model.layers` weights too broadly, and the Qwen3-TTS MTP/code predictor is implemented with plain PyTorch `nn.Linear` layers rather than vLLM quantized linear layers. Apply the patch after installing dependencies or after reinstalling `vllm` / `vllm-omni`:

```bash
bash scripts/patch_vllm_qwen3_tts_bnb4.sh
```

Then launch the 4bit server:

```bash
bash scripts/launch_server.sh configs/qwen3_tts_bnb4_optimized.yaml
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
code_predictor: Linear4bit/quant inventory entries for MTP modules
Using FlashAttention version 2
Application startup complete
```

Current behavior observed for LLM BitsAndBytes 4bit + MTP BitsAndBytes:

- Server startup succeeds.
- `/health` and `/v1/models` succeed.
- Logs confirm LLM BitsAndBytes and MTP quant inventory dumps.
- A direct short-text request succeeded with HTTP 200 in about 9.3s.
- A one-sample manifest smoke run succeeded with HTTP 200 in about 26.1s, `rtf_mean=1.61`.

Use this smoke command to reproduce the current LLM+MTP 4bit behavior:

```bash
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_bnb4_optimized.yaml \
  --output-dir results/bnb4_llm_mtp_bnb4_smoke \
  --warmup-requests 0 \
  --concurrency 1 \
  --batch-size 1 \
  --limit 1 \
  --save-audio \
  --response-format wav
```

## vLLM throughput optimization

The default upstream Qwen3-TTS vLLM-Omni stage config has a clear throughput limiter:

```text
stage 0 talker:  max_num_seqs: 10
stage 1 code2wav: max_num_seqs: 1
runtime defaults: max_inflight: 1
```

That means the LLM/talker stage can accept multiple requests, but the final audio decoder stage is effectively serialized. GPU utilization can therefore bounce below 100% even when `--concurrency 8` is used.

The optimized configs keep the stable vLLM-Omni scheduling limits and reduce client-side benchmark overhead:

- `configs/qwen3_tts_base_optimized.yaml`
- `configs/qwen3_tts_bnb4_optimized.yaml`
- `configs/qwen3_tts_stage0_optimized.yaml`
- `configs/qwen3_tts_stage0_bnb4_optimized.yaml`

Key changes:

- server-level `max_num_seqs: 8`
- stage 0 `max_num_seqs: 10`
- stage 0 `max_num_batched_tokens: 768`
- stage 1 `max_num_seqs: 1`
- runtime `max_inflight: 1`
- connector polling sleep `0.01s -> 0.005s`
- benchmark `write_partial_csv_every_n: 0` to avoid rewriting the full partial manifest every few requests
- benchmark `save_audio: false` in optimized configs to avoid disk I/O during speed runs

Use the optimized BF16 server like this:

```bash
bash scripts/stop_server.sh || true
bash scripts/launch_server.sh configs/qwen3_tts_base_optimized.yaml
until curl -fsS http://127.0.0.1:8091/v1/models >/dev/null; do sleep 5; done

python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base_optimized.yaml \
  --output-dir results/vllm_bf16_optimized_c8 \
  --warmup-requests 1 \
  --concurrency 8 \
  --batch-size 1 \
  --limit 300 \
  --timeout-sec 300 \
  --no-save-audio \
  --write-partial-csv-every-n 0 \
  --response-format wav
```

In this environment, raising stage 1 `code2wav` to `max_num_seqs: 2` caused repeated `Dropping output for unknown req` orchestrator warnings and worse latency. The stable optimized profile therefore keeps stage 1 serialized and focuses on avoiding avoidable client-side I/O during speed runs.

## Run benchmark on 300 reference manifest samples

The checked-in reference manifest has 300 samples. If `--limit` is omitted, all valid manifest rows are used.

```bash
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base_optimized.yaml \
  --output-dir results/reference_all_c1 \
  --warmup-requests 5 \
  --concurrency 1 \
  --save-audio \
  --response-format wav
```

## Parallel inference

`--concurrency` controls how many inference requests can run at the same time.

`--batch-size` is a logical grouping value used for `batch_id` and Colab-compatible reporting. In the vLLM HTTP client it is not a single GPU tensor batch, and it no longer limits whether `--concurrency` is effective.

Examples:

```bash
# Run all 300 samples, max 2 concurrent requests.
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base_optimized.yaml \
  --output-dir results/reference_all_parallel_c2 \
  --warmup-requests 5 \
  --concurrency 2 \
  --batch-size 1 \
  --save-audio \
  --response-format wav
```

```bash
# Run all 300 samples, max 8 concurrent requests.
# Current optimized server config uses max_num_seqs: 8, so this is a practical first high-throughput test.
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base_optimized.yaml \
  --output-dir results/reference_all_parallel_c8 \
  --warmup-requests 5 \
  --concurrency 8 \
  --batch-size 1 \
  --save-audio \
  --response-format wav
```

Behavior:

```text
--concurrency 2 --batch-size 1
  -> schedules 300 samples
  -> keeps up to 2 requests in flight
  -> starts the next sample as soon as one finishes
```

The benchmark uses a worker queue internally, so `--concurrency 8` means up to 8 HTTP inference requests are in flight regardless of `--batch-size`.

For throughput runs, prefer the optimized configs and start with `--concurrency 8`. Use `aggregate_rtf` and `aggregate_audio_throughput_sec_per_sec` to judge system throughput under concurrency; per-request `rtf_mean` includes queueing and can look worse as concurrency increases.

## Smoke test

```bash
python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base_optimized.yaml \
  --output-dir results/smoke_test \
  --warmup-requests 2 \
  --concurrency 1 \
  --limit 5 \
  --save-audio
```

## Progress and logs

The benchmark prints a `tqdm` progress bar for warmup and each concurrency run. To append benchmark progress into the latest server log:

```bash
SERVER_LOG="$(ls -t logs/server_*.log | head -n1)"

python -u -m src.benchmark_manifest \
  --config configs/qwen3_tts_base_optimized.yaml \
  --output-dir results/reference_all_parallel_c8 \
  --warmup-requests 5 \
  --concurrency 8 \
  --batch-size 1 \
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

For the default BF16 vLLM run, quantization fields are recorded as `none`/`bf16`. For `configs/qwen3_tts_bnb4_optimized.yaml`, run settings and summary fields record `llm_quant_method=bitsandbytes`, `mtp_quant_method=bitsandbytes`, `llm_bit=4`, and `mtp_bit=4`. HF direct-load fields such as `load_time_sec` and `model_memory_footprint_mb` are not directly available from the benchmark client, so they are left empty while vLLM/GPU metrics are recorded separately.

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
LIMIT=300 WARMUP=5 CONCURRENCY=8 TORCH_BATCH_SIZE=4 VLLM_BATCH_SIZE=1 \
  bash scripts/run_4way_benchmark.sh
```

`TORCH_BATCH_SIZE=4` keeps the direct PyTorch path busier on GPU by batching multiple samples in one `generate_voice_clone` call. Use `TORCH_BATCH_SIZE=1` only when you need strictly one-sample-at-a-time latency semantics.

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

For torch direct runtime, `configs/qwen3_tts_base_optimized.yaml` forces `attn_implementation: flash_attention_2` and `require_flash_attention: true`. If `flash_attn` is missing, `python -m src.benchmark_torch` stops with a clear install command instead of silently falling back to `sdpa`.

The torch config can also install FlashAttention before running:

```yaml
torch_runtime:
  attn_implementation: flash_attention_2
  require_flash_attention: true
  flash_attention:
    install: true
    package: flash-attn
    no_build_isolation: true
    no_cache_dir: true
    max_jobs: 4
    cuda_arch_list: null
    show_progress: true
    progress_interval_sec: 30
```

Manual install with progress logging:

```bash
bash scripts/install_flash_attention.sh configs/qwen3_tts_base_optimized.yaml
```

The installer streams pip/ninja/nvcc output to stdout and writes a log like `logs/flash_attention_install_<timestamp>.log`. When `show_progress=true`, it also prints periodic process and disk status while the CUDA extension is compiling.

The 4-way benchmark calls this installer before each torch run by default. To skip it after `flash_attn` is already installed:

```bash
INSTALL_FLASH_ATTN=0 LIMIT=300 WARMUP=5 bash scripts/run_4way_benchmark.sh
```

## Stop server

```bash
bash scripts/stop_server.sh
```
