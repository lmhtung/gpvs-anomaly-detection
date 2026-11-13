"""Read-only raw CSV loading with original indices and timestamp segments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ['Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', 'If', 'Vabc', 'Vf']


@dataclass(frozen=True)
class Recording:
    """Retained raw rows; ``sample_index`` is the zero-based row of the source CSV."""
    name: str
    values: pd.DataFrame
    time: np.ndarray
    elapsed: np.ndarray        # seconds since the first CSV timestamp (not since warmup)
    sample_index: np.ndarray
    segment_id: np.ndarray     # a timestamp gap starts a new segment
    dt: float                  # median sampling interval of the retained rows


def read_recording(path, warmup_seconds, gap_factor, channels):
    """Drop rows with elapsed < warmup_seconds and split at gaps > gap_factor × dt."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Recording not found: {path}')
    columns = ['Time', *channels]
    raw = pd.read_csv(path, float_precision='round_trip', usecols=lambda column: column in columns)
    missing = set(columns) - set(raw.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns {sorted(missing)}')
    data = raw[columns].to_numpy(dtype=float)
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError(f'{path.name}: expected at least two finite numeric rows')
    time = data[:, 0]
    if np.any(np.diff(time) <= 0):
        raise ValueError(f'{path.name}: Time must be strictly increasing')
    elapsed = time - time[0]
    kept = np.flatnonzero(elapsed >= warmup_seconds)
    if len(kept) < 2:
        raise ValueError(f'{path.name}: fewer than two rows after the {warmup_seconds:g} s warmup')
    delta = np.diff(time[kept])
    dt = float(np.median(delta))
    segment_id = np.r_[0, np.cumsum(delta > gap_factor * dt)].astype(np.int64)
    return Recording(name=path.stem, values=pd.DataFrame(data[kept, 1:], columns=channels),
                     time=time[kept].copy(), elapsed=elapsed[kept].copy(),
                     sample_index=kept.astype(np.int64), segment_id=segment_id, dt=dt)


def subset(recording, rows):
    """Contiguous part of a recording (slice or boolean mask), keeping original coordinates."""
    positions = np.arange(len(recording.time))[rows]
    segment = recording.segment_id[positions]
    return Recording(name=recording.name, values=recording.values.iloc[positions].reset_index(drop=True),
                     time=recording.time[positions], elapsed=recording.elapsed[positions],
                     sample_index=recording.sample_index[positions],
                     segment_id=(segment - segment[0]).astype(np.int64), dt=recording.dt)
