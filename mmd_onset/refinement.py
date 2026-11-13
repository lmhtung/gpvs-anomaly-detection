"""Raw-sample refinement of fast events on the grid-cycle residual.

The two-sided Energy-MMD profile starts rising up to ``h`` seconds before a
step change, because its right window already holds post-change data, so a
grid-level cut is early. A grid-periodic waveform has the natural pre-fault
model "this cycle repeats the previous one": the residual
``d[k] = x[k] - x[k-P]`` (P = one grid period) is noise before the change and
jumps when the waveform first deviates. Inside a bracket derived only from the
detector, a least-squares two-segment mean shift on the residual energy gives
the onset sample.

* Scales come from same-mode F0 (robust MAD of ``d`` per channel).
* Bracket: one grid period before the first confirmed Energy row, to one
  period after the prominence-filtered event peak. No reference times.
* If the best cut lies within half a period of the bracket end, the event may
  have been cut short and the coarse grid point is kept.
* Only Energy-confirmed (fast) events are refined. Gradual long-context drifts
  hardly change ``d`` and keep their grid coordinate.

The cut is a least-squares estimate, not a confidence interval.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

REFINED_BRANCHES = frozenset({'short_energy'})
CURVE_COLUMNS = ['sample_index', 'Time', 'elapsed', 'gain', 'n_left', 'n_right', 'eligible', 'selected']


def cycle_residual(values, period):
    """``x[k] - x[k-period]``; rows without one period of history are NaN."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) > period:
        out[period:] = values[period:] - values[:-period]
    return out


def residual_energy(values, scales, period, clip):
    """log1p of the channel mean of the clipped squared standardized residual.

    Clipping bounds one saturated channel; log1p makes the energy close to
    additive noise, so a least-squares mean shift is a reasonable model.
    """
    d = cycle_residual(values, period) / scales
    return np.log1p(np.mean(np.minimum(d * d, clip), axis=1))


def mean_shift_curve(y):
    """Two-segment least squares for every cut k = 1..n-1.

    gain[k] = SSE(one segment) - SSE(split at k) = k(n-k)/n · (mean_right - mean_left)²;
    ``signed`` keeps the direction so that only increases can be selected.
    """
    n = len(y)
    c = np.cumsum(y)
    k = np.arange(1, n)
    left = c[:-1] / k
    right = (c[-1] - c[:-1]) / (n - k)
    gain = k * (n - k) / n * (right - left) ** 2
    return k, gain, np.sign(right - left) * gain


def _segments(recording):
    edges = np.flatnonzero(np.diff(recording.segment_id)) + 1
    return list(zip(np.r_[0, edges].tolist(), np.r_[edges, len(recording.segment_id)].tolist()))


def _position(recording, sample_index):
    position = int(np.searchsorted(recording.sample_index, sample_index))
    if position >= len(recording.sample_index) or recording.sample_index[position] != sample_index:
        raise ValueError('Index is outside the retained raw rows')
    return position


@dataclass
class CycleResidualRefiner:
    mode: str
    config: dict
    period: int
    scales: np.ndarray
    reference_sd: float

    @classmethod
    def fit(cls, reference, config):
        opts = config['refinement']
        period = int(round(1.0 / (opts['grid_frequency_hz'] * reference.dt)))
        values = reference.values[config['channels']].to_numpy(float)
        segments = [(a, b) for a, b in _segments(reference) if b - a > period]
        residual = np.vstack([cycle_residual(values[a:b], period) for a, b in segments])
        residual = residual[np.isfinite(residual).all(axis=1)]
        if len(residual) < 10 * period:
            raise ValueError('F0 is too short for cycle-residual calibration')
        center = np.median(residual, axis=0)
        scales = 1.4826 * np.median(np.abs(residual - center), axis=0)
        for j in np.flatnonzero(scales <= 0):  # quantized channel: smallest positive step
            steps = np.abs(np.diff(values[:, j]))
            steps = steps[steps > 0]
            scales[j] = float(steps.min()) if len(steps) else 1.0
        energy = np.concatenate([residual_energy(values[a:b], scales, period, opts['residual_clip'])[period:]
                                 for a, b in segments])
        sd = 1.4826 * float(np.median(np.abs(energy - np.median(energy))))
        return cls(reference.name[-1], deepcopy(config), period, scales, max(sd, 1e-12))

    def snapshot(self):
        return {'reference': f'F0{self.mode}', 'grid_period_samples': self.period,
                'residual_scale': dict(zip(self.config['channels'], self.scales.tolist())),
                'reference_energy_robust_sd': self.reference_sd}

    def refine(self, recording, coarse, table):
        """Return (result dict, curve DataFrame) for one coarse onset row."""
        if coarse.branch not in REFINED_BRANCHES:
            return {'status': 'kept_coarse_gradual_event'}, pd.DataFrame(columns=CURVE_COLUMNS)
        # Use the prominence-filtered event peak. An earlier ripple can otherwise
        # put the end of the search bracket before the main transition.
        first, peak = int(coarse.group_index), int(coarse.peak_group_index)
        start_raw = _position(recording, int(table.original_split.iloc[first]))
        peak_raw = _position(recording, int(table.original_split.iloc[peak]))
        same = np.flatnonzero(recording.segment_id == recording.segment_id[start_raw])
        lo = max(int(same[0]) + self.period, start_raw - self.period)
        hi = min(int(same[-1]) + 1, peak_raw + self.period + 1)
        opts = self.config['refinement']
        min_side = max(1, int(round(opts['min_side_seconds'] / recording.dt)))
        if hi - lo < 2 * min_side + 1:
            return {'status': 'kept_coarse_short_bracket'}, pd.DataFrame(columns=CURVE_COLUMNS)
        values = recording.values[self.config['channels']].to_numpy(float)
        energy = residual_energy(values[lo - self.period:hi], self.scales, self.period,
                                 opts['residual_clip'])[self.period:]
        cut, gain, signed = mean_shift_curve(energy)
        eligible = (cut >= min_side) & (len(energy) - cut >= min_side)
        score = np.where(eligible, signed, -np.inf)
        best = int(np.argmax(score))
        if not score[best] > 0:
            return {'status': 'kept_coarse_no_increase'}, pd.DataFrame(columns=CURVE_COLUMNS)
        position = lo + int(cut[best])
        if hi - position <= self.period // 2:
            # The shift sits at the right edge: the bracket may have cut the event
            # short, so the refined sample is not trusted and the grid point is kept.
            return {'status': 'kept_coarse_unreliable'}, pd.DataFrame(columns=CURVE_COLUMNS)
        rows = lo + cut
        curve = pd.DataFrame({'sample_index': recording.sample_index[rows], 'Time': recording.time[rows],
                              'elapsed': recording.elapsed[rows], 'gain': gain, 'n_left': cut,
                              'n_right': len(energy) - cut, 'eligible': eligible.astype(np.uint8),
                              'selected': (np.arange(len(cut)) == best).astype(np.uint8)},
                             columns=CURVE_COLUMNS)
        return {'status': 'refined', 'sample_index': int(recording.sample_index[position]),
                'Time': float(recording.time[position]), 'elapsed': float(recording.elapsed[position]),
                'bracket_start_sample_index': int(recording.sample_index[lo]),
                'bracket_end_sample_index': int(recording.sample_index[hi - 1]) + 1,
                'shift_z': float(np.sqrt(gain[best]) / self.reference_sd)}, curve
