import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd


REQUIRED_COLUMNS = [
    'subset',
    'pair_id',
    'ref_audio_path',
    'ref_audio_relpath',
    'ref_duration_sec',
    'ref_text',
    'target_audio_path',
    'target_audio_relpath',
    'target_duration_sec',
    'target_text',
    'sim_audio_path',
    'sim_audio_relpath',
    'wer_ref_text',
]


@dataclass
class BenchmarkSample:
    row_index: int
    subset: str
    pair_id: str
    ref_audio_path: Optional[str]
    ref_audio_relpath: Optional[str]
    resolved_ref_audio_path: Optional[str]
    ref_duration_sec: Optional[float]
    ref_text: Optional[str]
    target_audio_path: Optional[str]
    target_audio_relpath: Optional[str]
    target_duration_sec: Optional[float]
    target_text: Optional[str]
    sim_audio_path: Optional[str]
    sim_audio_relpath: Optional[str]
    wer_ref_text: Optional[str]
    valid: bool = True
    error_message: Optional[str] = None


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    cleaned = str(path).strip()
    return cleaned if cleaned else None


def _resolve_candidate(path: str, audio_root: Optional[str], prefix_from: Optional[str], prefix_to: Optional[str]) -> str:
    candidate = path
    if prefix_from and prefix_to and candidate.startswith(prefix_from):
        candidate = candidate.replace(prefix_from, prefix_to, 1)
    if audio_root and not os.path.isabs(candidate):
        candidate = os.path.join(audio_root, candidate)
    return os.path.normpath(candidate)


def resolve_ref_audio_path(ref_audio_relpath: Optional[str], ref_audio_path: Optional[str], audio_root: Optional[str], path_prefix_from: Optional[str], path_prefix_to: Optional[str]) -> Optional[str]:
    if ref_audio_relpath:
        candidate = os.path.join(audio_root or '', str(ref_audio_relpath))
        candidate = _resolve_candidate(candidate, None, path_prefix_from, path_prefix_to)
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    if ref_audio_path:
        candidate = _resolve_candidate(str(ref_audio_path), audio_root, path_prefix_from, path_prefix_to)
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    return None


def load_manifest(
    manifest_path: str,
    audio_root: Optional[str] = None,
    path_prefix_from: Optional[str] = None,
    path_prefix_to: Optional[str] = None,
    subset_filter: Optional[List[str]] = None,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[BenchmarkSample]:
    df = pd.read_csv(manifest_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f'Manifest missing required columns: {missing}')

    if subset_filter:
        df = df[df['subset'].isin(subset_filter)]

    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if limit is not None:
        df = df.head(limit)

    samples: List[BenchmarkSample] = []
    for index, row in df.iterrows():
        ref_audio_path = _normalize_path(row.get('ref_audio_path'))
        ref_audio_relpath = _normalize_path(row.get('ref_audio_relpath'))
        target_text = _normalize_path(row.get('target_text'))
        ref_text = _normalize_path(row.get('ref_text'))

        resolved_ref_audio_path = resolve_ref_audio_path(
            ref_audio_relpath,
            ref_audio_path,
            audio_root,
            path_prefix_from,
            path_prefix_to,
        )

        sample = BenchmarkSample(
            row_index=int(index + 1),
            subset=_normalize_path(row.get('subset')) or 'unknown',
            pair_id=_normalize_path(row.get('pair_id')) or f'row_{index + 1}',
            ref_audio_path=ref_audio_path,
            ref_audio_relpath=ref_audio_relpath,
            resolved_ref_audio_path=resolved_ref_audio_path,
            ref_duration_sec=float(row.get('ref_duration_sec')) if row.get('ref_duration_sec') not in (None, '') else None,
            ref_text=ref_text,
            target_audio_path=_normalize_path(row.get('target_audio_path')),
            target_audio_relpath=_normalize_path(row.get('target_audio_relpath')),
            target_duration_sec=float(row.get('target_duration_sec')) if row.get('target_duration_sec') not in (None, '') else None,
            target_text=target_text,
            sim_audio_path=_normalize_path(row.get('sim_audio_path')),
            sim_audio_relpath=_normalize_path(row.get('sim_audio_relpath')),
            wer_ref_text=_normalize_path(row.get('wer_ref_text')),
        )

        if not sample.ref_text:
            sample.valid = False
            sample.error_message = 'missing ref_text'
        elif not sample.target_text:
            sample.valid = False
            sample.error_message = 'missing target_text'
        elif not sample.resolved_ref_audio_path or not os.path.exists(sample.resolved_ref_audio_path):
            sample.valid = False
            sample.error_message = 'reference audio file missing'

        samples.append(sample)

    total = len(samples)
    valid = sum(1 for sample in samples if sample.valid)
    invalid = total - valid
    subset_counts = df['subset'].value_counts().to_dict()
    print(f'Loaded manifest {manifest_path}, rows={total}, valid={valid}, invalid={invalid}')
    print(f'Subset counts: {subset_counts}')
    return samples


def save_validated_manifest(samples: List[BenchmarkSample], output_path: str) -> None:
    rows: List[Dict[str, Any]] = [asdict(sample) for sample in samples]
    pd.DataFrame(rows).to_csv(output_path, index=False)
