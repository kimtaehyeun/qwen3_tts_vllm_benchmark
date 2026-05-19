import argparse
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize benchmark results')
    parser.add_argument('--run-dir', required=True)
    return parser.parse_args()


def _load_results(run_dir: str) -> Dict[str, Any]:
    requests_path = os.path.join(run_dir, 'requests.csv')
    summary_path = os.path.join(run_dir, 'summary.csv')
    memory_path = os.path.join(run_dir, 'memory_summary.json')
    if not os.path.exists(requests_path):
        raise FileNotFoundError(requests_path)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)
    if not os.path.exists(memory_path):
        raise FileNotFoundError(memory_path)
    return {
        'requests': pd.read_csv(requests_path),
        'summary': pd.read_csv(summary_path),
        'memory': json.loads(open(memory_path, 'r', encoding='utf-8').read()),
    }


def _summarize_by_group(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    if group_key not in df.columns:
        raise ValueError(f'{group_key} not found in requests data')
    df_success = df[df['success'] == True].copy()
    grouped = df_success.groupby(group_key)
    rows: List[Dict[str, Any]] = []
    for key, group in grouped:
        row = {
            group_key: key,
            'num_requests': len(group),
            'success_rate': group['success'].mean(),
            'latency_mean_sec': group['end_to_end_latency_sec'].mean(),
            'latency_p95_sec': group['end_to_end_latency_sec'].quantile(0.95),
            'rtf_mean': group['rtf'].mean(),
            'rtf_p95': group['rtf'].quantile(0.95),
            'audio_duration_mean_sec': group['audio_duration_sec'].mean(),
        }
        if 'steady_state_streaming_rtf' in group.columns:
            row['steady_state_streaming_rtf_mean'] = group['steady_state_streaming_rtf'].mean()
            row['steady_state_streaming_rtf_p95'] = group['steady_state_streaming_rtf'].quantile(0.95)
        if 'steady_state_audio_throughput_sec_per_sec' in group.columns:
            row['steady_state_audio_throughput_mean_sec_per_sec'] = group['steady_state_audio_throughput_sec_per_sec'].mean()
            row['steady_state_audio_throughput_p95_sec_per_sec'] = group['steady_state_audio_throughput_sec_per_sec'].quantile(0.95)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(by=group_key)


def _plot_metric(summary: pd.DataFrame, x_key: str, y_key: str, output_path: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(summary[x_key], summary[y_key], marker='o')
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    data = _load_results(args.run_dir)
    requests_df = data['requests']
    summary_df = data['summary']
    memory_summary = data['memory']
    os.makedirs(os.path.join(args.run_dir, 'plots'), exist_ok=True)

    subset_summary = _summarize_by_group(requests_df, 'subset')
    concurrency_summary = _summarize_by_group(requests_df, 'concurrency')
    subset_summary.to_csv(os.path.join(args.run_dir, 'summary_by_subset.csv'), index=False)
    concurrency_summary.to_csv(os.path.join(args.run_dir, 'summary_by_concurrency.csv'), index=False)

    console = Console()
    console.print('[bold]Benchmark summary by subset[/bold]')
    table = Table(show_header=True, header_style='bold magenta')
    for col in subset_summary.columns:
        table.add_column(str(col))
    for _, row in subset_summary.iterrows():
        table.add_row(*[str(row[col]) for col in subset_summary.columns])
    console.print(table)

    console.print('\n[bold]Benchmark summary by concurrency[/bold]')
    table = Table(show_header=True, header_style='bold green')
    for col in concurrency_summary.columns:
        table.add_column(str(col))
    for _, row in concurrency_summary.iterrows():
        table.add_row(*[str(row[col]) for col in concurrency_summary.columns])
    console.print(table)

    if 'concurrency' in summary_df.columns:
        x = summary_df['concurrency']
        _plot_metric(summary_df, 'concurrency', 'latency_mean_sec', os.path.join(args.run_dir, 'plots', 'latency_mean_vs_concurrency.png'), 'Latency Mean vs Concurrency')
        _plot_metric(summary_df, 'concurrency', 'latency_p95_sec', os.path.join(args.run_dir, 'plots', 'latency_p95_vs_concurrency.png'), 'Latency P95 vs Concurrency')
        if 'rtf_mean' in summary_df.columns:
            _plot_metric(summary_df, 'concurrency', 'rtf_mean', os.path.join(args.run_dir, 'plots', 'rtf_mean_vs_concurrency.png'), 'RTF Mean vs Concurrency')
        if 'rtf_p95' in summary_df.columns:
            _plot_metric(summary_df, 'concurrency', 'rtf_p95', os.path.join(args.run_dir, 'plots', 'rtf_p95_vs_concurrency.png'), 'RTF P95 vs Concurrency')
        if 'steady_state_streaming_rtf_mean' in summary_df.columns and summary_df['steady_state_streaming_rtf_mean'].notna().any():
            _plot_metric(summary_df, 'concurrency', 'steady_state_streaming_rtf_mean', os.path.join(args.run_dir, 'plots', 'steady_state_streaming_rtf_mean_vs_concurrency.png'), 'Steady-State Streaming RTF Mean vs Concurrency')
        if 'steady_state_audio_throughput_mean_sec_per_sec' in summary_df.columns and summary_df['steady_state_audio_throughput_mean_sec_per_sec'].notna().any():
            _plot_metric(summary_df, 'concurrency', 'steady_state_audio_throughput_mean_sec_per_sec', os.path.join(args.run_dir, 'plots', 'steady_state_audio_throughput_mean_vs_concurrency.png'), 'Steady-State Audio Throughput Mean vs Concurrency')
        _plot_metric(summary_df, 'concurrency', 'requests_per_sec', os.path.join(args.run_dir, 'plots', 'requests_per_sec_vs_concurrency.png'), 'Requests/s vs Concurrency')
        if 'tokens_per_sec_mean' in summary_df.columns and summary_df['tokens_per_sec_mean'].notna().any():
            _plot_metric(summary_df, 'concurrency', 'tokens_per_sec_mean', os.path.join(args.run_dir, 'plots', 'tokens_per_sec_vs_concurrency.png'), 'Tokens/s Mean vs Concurrency')
    if memory_summary:
        memory_plot_path = os.path.join(args.run_dir, 'plots', 'gpu_memory_peak_vs_concurrency.png')
        if os.path.exists(memory_plot_path) or 'gpu_memory_peak_mb' in summary_df.columns:
            try:
                _plot_metric(summary_df, 'concurrency', 'gpu_memory_peak_mb', memory_plot_path, 'GPU Memory Peak vs Concurrency')
            except Exception:
                pass

    console.print(f'Plots saved to {os.path.join(args.run_dir, "plots")}')


if __name__ == '__main__':
    main()
