"""Window descriptors, local two-sample MMD profiles and F0 calibration.

Pipeline per recording (see README):

1. 20 ms descriptor windows, 10 ms hop; each candidate split uses only
   windows lying fully before / fully after it (disjoint raw support).
2. Long-context Gaussian-mixture MMD² and short-context Energy MMD per
   channel, divided by the same-mode F0 maximum of each context, fused over
   contexts by a geometric mean and over channels by a maximum.
3. Thresholds come from F0 only: ``multiplier × max(1, in-sample peak,
   held-out peak)``. No target statistic is used for normalization.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Recording, subset
from .selection import options, short_config, select_local

# One row per candidate split: retained-row position, original CSV row and clock.
FRAME_COLUMNS = ['split', 'original_split', 'split_time', 'split_elapsed',
                 'segment_id', 'window_samples', 'hop_samples']
# Level/frequency channels use (mean, std); phase currents/voltages use the AC group.
LEVEL_CHANNELS = frozenset(('Ipv', 'Vpv', 'Vdc', 'Iabc', 'If', 'Vabc', 'Vf'))


def feature_group(channel):
    return 'dc' if channel in LEVEL_CHANNELS else 'ac'


def floor_setting(spec, channel):
    """Scale-floor value: per-channel override, otherwise the default."""
    value = float(spec.get('channels', {}).get(channel, spec['default']))
    if not np.isfinite(value):
        raise ValueError(f'{channel}: floor must be finite')
    return value


# --------------------------------------------------------------------------- windows

def _sample_counts(recording, window_seconds, hop_seconds):
    if not 0 < hop_seconds <= window_seconds:
        raise ValueError('Require 0 < hop_seconds <= window_seconds')
    count, hop = (int(round(value / recording.dt)) for value in (window_seconds, hop_seconds))
    if count < 2 or hop < 1:
        raise ValueError(f'{recording.name}: window needs >=2 samples and hop >=1 sample')
    return count, hop


def build_groups(recording, window_seconds, hop_seconds):
    """Candidate splits: complete 4-window groups on each segment's hop grid.

    The split is the start of the third window. Short segments and incomplete
    tails are never padded or bridged across timestamp gaps. ``split`` is a
    retained-row offset; ``original_split`` the zero-based source CSV row.
    """
    count, hop = _sample_counts(recording, window_seconds, hop_seconds)
    rows = []
    for segment in np.unique(recording.segment_id):
        positions = np.flatnonzero(recording.segment_id == segment)
        first, stop = int(positions[0]), int(positions[-1]) + 1
        for start in range(first, stop - 4 * count + 1, hop):
            split = start + 2 * count
            rows.append({
                'split': split, 'original_split': int(recording.sample_index[split]),
                'split_time': float(recording.time[split]),
                'split_elapsed': float(recording.elapsed[split]),
                'segment_id': int(segment), 'window_samples': count, 'hop_samples': hop,
            })
    return pd.DataFrame(rows, columns=FRAME_COLUMNS)


def extract(recording, config):
    """Return (frames, windows, values, feature names) for one recording."""
    frames = build_groups(recording, config['window_seconds'], config['hop_seconds'])
    if frames.empty:
        raise ValueError(f'{recording.name}: no complete four-window groups')
    width, hop = int(frames.iloc[0].window_samples), int(frames.iloc[0].hop_samples)
    if width % hop:
        raise ValueError('window_samples must be a multiple of hop_samples')
    for contexts in (config['contexts_seconds'], options(config)['short_contexts_seconds']):
        samples = [int(round(value / recording.dt)) for value in contexts]
        if len(set(samples)) != len(samples) or min(samples) < width + hop:
            raise ValueError('Contexts must be distinct and contain >=2 windows per side')
    starts, segments = [], []
    for segment in np.unique(recording.segment_id):
        positions = np.flatnonzero(recording.segment_id == segment)
        local = np.arange(positions[0], positions[-1] + 2 - width, hop, dtype=int)
        starts.extend(local)
        segments.extend([segment] * len(local))
    starts, segments = np.asarray(starts, int), np.asarray(segments, int)
    indices = starts[:, None] + np.arange(width)
    values, names = {}, {}
    for channel in config['channels']:
        raw = recording.values[channel].to_numpy(float)[indices]
        selected = config['features'][feature_group(channel)]
        descriptors = {'mean': lambda: raw.mean(axis=1),
                       'std': lambda: raw.std(axis=1, ddof=0),
                       'rms': lambda: np.sqrt(np.mean(raw * raw, axis=1))}
        values[channel] = np.column_stack([descriptors[name]() for name in selected])
        names[channel] = list(selected)
        if not np.isfinite(values[channel]).all():
            raise ValueError(f'{recording.name}/{channel}: non-finite window descriptors')
    windows = pd.DataFrame({'start': starts, 'segment_id': segments})
    return frames, windows, values, names


# --------------------------------------------------------------------------- kernels

def squared_distances(x):
    result = np.zeros((len(x), len(x)), dtype=float)
    for column in x.T:
        difference = column[:, None] - column[None, :]
        result += difference * difference
    if not np.isfinite(result).all():
        raise ValueError('Non-finite kernel distances')
    return result


def bandwidth(x, config):
    """Median pairwise distance of up to `max_reference_points` evenly spaced F0 rows."""
    count = min(len(x), config['kernel']['max_reference_points'])
    subset = x[np.linspace(0, len(x) - 1, count, dtype=int)]
    values = squared_distances(subset)[np.triu_indices(count, 1)]
    base = float(np.sqrt(np.median(values))) if len(values) else 0.
    return max(base, config['kernel']['bandwidth_floor'])


def gaussian_kernel(x, base, factors):
    distances = squared_distances(x)
    kernel = np.zeros_like(distances)
    for factor in factors:
        kernel += np.exp(-distances / (2. * (base * factor) ** 2)) / len(factors)
    return kernel


def energy_kernel(x):
    """Negative Euclidean distance: the Energy-MMD V-statistic kernel."""
    return -np.sqrt(squared_distances(x))


def _rectangle(integral, a, b, c, d):
    return integral[b, d] - integral[a, d] - integral[b, c] + integral[a, c]


def profiles(recording, frames, windows, values, transforms, config, metric='gaussian'):
    """MMD² per channel/context at every split, using a 2-D integral image.

    Each side only contains windows lying fully inside ``[split-h, split)`` or
    ``[split, split+h)``; a context needs at least two windows per side.
    """
    contexts = config['contexts_seconds']
    output = {channel: np.zeros((len(frames), len(contexts))) for channel in transforms}
    valid = np.zeros((len(frames), len(contexts)), dtype=bool)
    width = int(frames.iloc[0].window_samples)
    for segment in np.unique(frames.segment_id):
        rows = np.flatnonzero(frames.segment_id.to_numpy() == segment)
        selected = np.flatnonzero(windows.segment_id.to_numpy() == segment)
        starts = windows.start.to_numpy(int)[selected]
        splits = frames.split.to_numpy(int)[rows]
        ranges = []
        for j, seconds in enumerate(contexts):
            samples = int(round(seconds / recording.dt))
            a = np.searchsorted(starts, splits - samples, side='left')
            b = np.searchsorted(starts, splits - width, side='right')
            c = np.searchsorted(starts, splits, side='left')
            d = np.searchsorted(starts, splits + samples - width, side='right')
            good = ((splits - samples >= starts[0]) & (splits + samples <= starts[-1] + width)
                    & (b - a >= 2) & (d - c >= 2))
            valid[rows, j] = good
            ranges.append((good, a, b, c, d))
        for channel, model in transforms.items():
            x = (values[channel][selected] - model['center']) / model['scale']
            kernel = (energy_kernel(x) if metric == 'energy' else
                      gaussian_kernel(x, model['bandwidth'], config['kernel']['bandwidth_factors']))
            tolerance = 1e-8 * max(1., float(np.max(np.abs(kernel))))
            integral = np.pad(kernel.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
            for j, (good, a, b, c, d) in enumerate(ranges):
                a, b, c, d = (indices[good] for indices in (a, b, c, d))
                nl, nr = b - a, d - c
                score = (_rectangle(integral, a, b, a, b) / nl ** 2
                         + _rectangle(integral, c, d, c, d) / nr ** 2
                         - 2. * _rectangle(integral, a, b, c, d) / (nl * nr))
                if not np.isfinite(score).all() or np.any(score < -tolerance):
                    raise ValueError(f'{recording.name}/{channel}: invalid MMD evidence')
                output[channel][rows[good], j] = np.maximum(score, 0.)
    return output, valid


def aggregate(recording, frames, raw, valid, reference_max, config, prefix=''):
    """Geometric mean over admissible contexts per channel, then max over channels."""
    times = frames.split_elapsed.to_numpy(float)
    eligible = ((valid.sum(axis=1) >= config['min_scales'])
                & (times >= config['onset']['search_start_seconds'])
                & (times <= recording.elapsed[-1] - config['onset']['search_end_margin_seconds']))
    columns, channel_scores = {}, []
    for channel in raw:
        ratios = raw[channel] / np.maximum(reference_max[channel], config['mmd_floor'])
        score = np.zeros(len(frames))
        for row in np.flatnonzero(eligible):
            score[row] = np.exp(np.mean(np.log(np.maximum(ratios[row, valid[row]], np.finfo(float).tiny))))
        columns[f'{prefix}score_{channel}'] = score
        channel_scores.append(score)
        for j, context in enumerate(config['contexts_seconds']):
            columns[f'{prefix}{channel}__mmd__{context:g}s'] = raw[channel][:, j]
            columns[f'{prefix}{channel}__ratio__{context:g}s'] = ratios[:, j]
    columns[f'{prefix}score'] = np.max(np.column_stack(channel_scores), axis=1)
    for j, context in enumerate(config['contexts_seconds']):
        columns[f'{prefix}valid__{context:g}s'] = valid[:, j].astype(np.uint8)
    columns[f'{prefix}valid_scales'] = valid.sum(axis=1)
    columns[f'{prefix}valid'] = eligible.astype(np.uint8)
    return columns


def _reference_max(raw, valid, config, name):
    """Per-channel F0 maximum of every context (each context must exist on F0)."""
    result = {}
    for channel in raw:
        if not valid.any(axis=0).all():
            raise ValueError(f'{name}: F0 lacks support for a context; reduce contexts')
        result[channel] = np.array([raw[channel][valid[:, j], j].max() for j in range(valid.shape[1])])
    return result


def _peak(columns, prefix=''):
    eligible = columns[f'{prefix}valid'].astype(bool)
    return float(columns[f'{prefix}score'][eligible].max()) if eligible.any() else 0.


# --------------------------------------------------------------------------- model

@dataclass
class Model:
    """Same-mode F0 calibration; scoring never refits anything on the target."""
    mode: str
    config: dict
    transforms: dict
    reference_max: dict
    short_reference_max: dict
    threshold: float
    short_threshold: float
    reference_peak: float
    short_reference_peak: float
    reference: Recording
    reference_extract: tuple
    reference_columns: dict
    holdout_peaks: dict
    holdout_models: list
    joint_model: object | None = None

    def local_scores(self, recording, extracted=None):
        """Long Gaussian ('') and short Energy ('short_') evidence."""
        frames, windows, values, _ = extracted or extract(recording, self.config)
        raw, valid = profiles(recording, frames, windows, values, self.transforms, self.config)
        columns = aggregate(recording, frames, raw, valid, self.reference_max, self.config)
        columns['threshold'] = np.full(len(frames), self.threshold)
        columns['ratio'] = columns['score'] / self.threshold
        cfg = short_config(self.config)
        raw, valid = profiles(recording, frames, windows, values, self.transforms, cfg, metric='energy')
        columns.update(aggregate(recording, frames, raw, valid, self.short_reference_max, cfg, 'short_'))
        columns['short_threshold'] = np.full(len(frames), self.short_threshold)
        columns['short_ratio'] = columns['short_score'] / self.short_threshold
        return columns

    def ensure_joint(self):
        """Fit the joint branch on F0 once, only when a recording needs it."""
        if self.joint_model is None:
            from . import joint
            holdout = 0.
            for model, test in self.holdout_models:
                part = joint.fit(model.reference, *model.reference_extract[:3], model.transforms, model.config)
                frames, windows, values, _ = extract(test, self.config)
                columns = part.score(test, frames, windows, values)
                holdout = max(holdout, _peak(columns, 'joint_'))
            self.holdout_peaks['joint'] = holdout
            self.joint_model = joint.fit(self.reference, *self.reference_extract[:3], self.transforms,
                                         self.config, holdout_peak=holdout)
            self.reference_columns.update(self.joint_model.reference_columns)
            self.reference_columns['joint_evaluated'] = np.ones(len(self.reference_extract[0]), np.uint8)
        return self.joint_model

    def score(self, recording):
        """All evidence for one recording; joint runs only if the local gate needs it."""
        from . import joint
        if recording.name[-1] != self.mode:
            raise ValueError('Reference and target control modes must match')
        extracted = extract(recording, self.config)
        frames, windows, values, _ = extracted
        columns = self.local_scores(recording, extracted)
        columns['joint_evaluated'] = np.zeros(len(frames), dtype=np.uint8)
        table = pd.concat([frames, pd.DataFrame(columns, index=frames.index)], axis=1)
        if joint.needs_score(table, self.config, select_local(table, recording, self.config)):
            columns.update(self.ensure_joint().score(recording, frames, windows, values))
            columns['joint_evaluated'][:] = 1
        return frames, columns


# --------------------------------------------------------------------------- calibration

def split_recording(recording, fraction=.5):
    """Two contiguous parts of the retained rows, keeping original coordinates."""
    cut = int(np.searchsorted(recording.elapsed, recording.elapsed[0]
                              + fraction * (recording.elapsed[-1] - recording.elapsed[0])))
    return [subset(recording, slice(0, cut)), subset(recording, slice(cut, None))]


def _holdout_peaks(reference, config):
    """Largest score of one F0 half under a model fitted on the other half.

    Context scores normalized by their own in-sample F0 maxima are bounded by
    one on that reference. Scoring the other half measures how much healthy
    variation can exceed those maxima.
    A reference too short to halve keeps the in-sample threshold; the reason
    is recorded in the snapshot.
    """
    local = {**deepcopy(config), 'calibration': {'holdout_halves': False}}
    first, second = split_recording(reference)
    peaks, models = {'long': 0., 'short': 0.}, []
    for train, test in ((first, second), (second, first)):
        try:
            model = fit(train, local)
        except ValueError as exc:
            return {'long': 0., 'short': 0., 'unavailable': str(exc)}, []
        columns = model.local_scores(test)
        peaks['long'] = max(peaks['long'], _peak(columns))
        peaks['short'] = max(peaks['short'], _peak(columns, 'short_'))
        models.append((model, test))
    return peaks, models


def fit(reference, config):
    """Calibrate descriptor scales, kernels, F0 maxima and thresholds on F0."""
    if reference.name not in {'F0L', 'F0M'}:
        raise ValueError('The reference must be F0L or F0M')
    config = deepcopy(config)
    extracted = extract(reference, config)
    frames, windows, values, names = extracted
    holdout, holdout_models = ({'long': 0., 'short': 0.}, [])
    if config['calibration']['holdout_halves']:
        holdout, holdout_models = _holdout_peaks(reference, config)
    normal = config['normalization']
    transforms = {}
    for channel in config['channels']:
        x = values[channel]
        center = np.median(x, axis=0)
        amplitude = float(np.median(np.abs(reference.values[channel].to_numpy(float))))
        floor = max(floor_setting(normal['absolute_floor'], channel),
                    floor_setting(normal['relative_floor'], channel) * amplitude)
        scale = np.maximum(normal['mad_multiplier'] * np.median(np.abs(x - center), axis=0), floor)
        transforms[channel] = {'feature_names': names[channel], 'center': center, 'scale': scale,
                               'bandwidth': bandwidth((x - center) / scale, config)}
    multiplier, minimum = config['threshold_multiplier'], config['min_threshold']

    raw, valid = profiles(reference, frames, windows, values, transforms, config)
    reference_max = _reference_max(raw, valid, config, reference.name)
    columns = aggregate(reference, frames, raw, valid, reference_max, config)
    peak = _peak(columns)
    threshold = multiplier * max(peak, minimum, holdout['long'])
    columns['threshold'] = np.full(len(frames), threshold)
    columns['ratio'] = columns['score'] / threshold

    cfg = short_config(config)
    raw, valid = profiles(reference, frames, windows, values, transforms, cfg, metric='energy')
    short_max = _reference_max(raw, valid, cfg, reference.name)
    columns.update(aggregate(reference, frames, raw, valid, short_max, cfg, 'short_'))
    short_peak = _peak(columns, 'short_')
    short_threshold = multiplier * max(short_peak, minimum, holdout['short'])
    columns['short_threshold'] = np.full(len(frames), short_threshold)
    columns['short_ratio'] = columns['short_score'] / short_threshold
    columns['joint_evaluated'] = np.zeros(len(frames), dtype=np.uint8)
    if not columns['valid'].any() or not columns['short_valid'].any():
        raise ValueError(f'{reference.name}: no admissible F0 split')
    return Model(reference.name[-1], config, transforms, reference_max, short_max, threshold,
                 short_threshold, peak, short_peak, reference, extracted, columns, holdout, holdout_models)
