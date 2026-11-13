"""A2 features: E1 representation + left padding + 16 history-free channels -> [157, 32] sequences.

Pipeline (all causal: a value at endpoint e uses samples before e only)
1. Descriptors on 200-sample windows, hop 100, from 0.5 s after the recording start (warm-up cut):
   level channels (Ipv Vpv Vdc Iabc If Vabc Vf) -> (mean, std); phases (ia..vc) -> (RMS, mean).
2. Source F0 bank (from 3 s on): median/MAD scaling, kernel bandwidths, maxima of every score.
3. Context h compares descriptor windows in [e-2h, e-h) and [e-h, e):
   Gaussian MMD (h = .2 .5 1 2 s) 13x4 = 52, Energy distance (h = .04 .08 .16 s) 13x3 = 39, 7 validity masks.
4. Lag channels: Gaussian MMD of [e-h, e) vs [e-lag-h, e-lag), (h, lag) = (1,2) (1,4) (1,6) s: 39 scores + 3 masks.
   The older window never reaches before start_seconds.
5. log1p(MMD / source-F0 max), z-score on source train (std floor, clip), invalid -> 0.
6. 32 consecutive endpoints (10 ms apart) end at each evaluation window's right edge. Steps before the first
   descriptor (or across a timestamp gap) are zero and a step-valid channel marks them, so a decision exists from
   the first descriptor after the warm-up cut.
7. 16 dimensionless per-descriptor channels that need no history: log1p(std/|mean|) of the 7 level channels,
   mean/RMS of the 6 phases, RMS unbalance of the 3 currents and of the 3 voltages, log(P_AC / P_DC) with
   P_AC = mean(Iabc) mean(Vabc), P_DC = mean(Ipv) mean(Vpv). No absolute DC level (a recording-identity shortcut).
Channel order: [91 scores | 39 lag | 16 absolute | 7 masks | 3 lag masks | 1 step-valid].
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VARIABLES = ['Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', 'If', 'Vabc', 'Vf']
LEVEL_CHANNELS = {'Ipv', 'Vpv', 'Vdc', 'Iabc', 'If', 'Vabc', 'Vf'}
N_LAG = 3
N_ABS = 16
LEVEL = [0, 1, 2, 9, 10, 11, 12]          # descriptor = (mean, std)
PHASE_I, PHASE_V = [3, 4, 5], [6, 7, 8]   # descriptor = (rms, mean)


@dataclass
class Recording:
    name: str
    values: np.ndarray
    times: np.ndarray
    segments: np.ndarray


def read_recording(path, cfg):
    path = Path(path)
    frame = pd.read_csv(path)
    times = frame.Time.to_numpy(dtype=np.float64)
    values = frame[VARIABLES].to_numpy(dtype=np.float64)
    dt = np.diff(times)
    period = 1 / cfg['sample_rate_hz']
    if (dt <= 0).any() or abs(np.median(dt) / period - 1) > .01 or not np.isfinite(values).all():
        raise ValueError(f'{path}: require finite data and increasing 10 kHz timestamps')
    segments = np.r_[0, np.cumsum(dt > 2 * period)].astype(np.int64)
    return Recording(path.stem, values, times, segments)


def crop(rec, start, length=None):
    """Recording from `start` (original time) on (optionally `length` seconds), times reset to 0."""
    keep = rec.times >= start
    if length is not None:
        keep &= rec.times < start + length
    times, segments = rec.times[keep], rec.segments[keep]
    return Recording(rec.name, rec.values[keep], times - times[0], segments - segments[0])


# ----------------------------------------------------------------------------- kernels (from mmd_onset/detector.py)
def squared_distances(x):
    result = np.zeros((len(x), len(x)), dtype=float)
    for column in x.T:
        difference = column[:, None] - column[None, :]
        result += difference * difference
    return result


def bandwidth(x, max_points, floor):
    """Median pairwise distance of up to `max_points` evenly spaced rows."""
    count = min(len(x), max_points)
    subset = x[np.linspace(0, len(x) - 1, count, dtype=int)]
    values = squared_distances(subset)[np.triu_indices(count, 1)]
    base = float(np.sqrt(np.median(values))) if len(values) else 0.
    return max(base, floor)


def gaussian_kernel(x, base, factors):
    distances = squared_distances(x)
    kernel = np.zeros_like(distances)
    for factor in factors:
        kernel += np.exp(-distances / (2. * (base * factor) ** 2)) / len(factors)
    return kernel


def energy_kernel(x):
    return -np.sqrt(squared_distances(x))


# ----------------------------------------------------------------------------- descriptors and MMD
def descriptors(rec, cfg, minimum_time=None):
    """ends [n] (sample index after the window), values [n,13,2], segment [n]."""
    opts = cfg['features']
    width, hop = opts['descriptor_samples'], opts['hop_samples']
    starts = np.arange(0, len(rec.times) - width + 1, hop, dtype=np.int64)
    keep = rec.segments[starts] == rec.segments[starts + width - 1]
    if minimum_time is None:
        keep &= rec.times[starts] - rec.times[0] >= cfg['warmup_seconds']
    else:
        keep &= rec.times[starts] >= minimum_time
    starts = starts[keep]
    if not len(starts):
        raise ValueError(f'{rec.name}: no complete descriptor windows')
    raw = rec.values[starts[:, None] + np.arange(width)]  # [n, width, 13]
    out = np.empty((len(starts), 13, 2))
    for j, channel in enumerate(VARIABLES):
        x = raw[:, :, j]
        if channel in LEVEL_CHANNELS:
            out[:, j] = np.column_stack([x.mean(1), x.std(1)])
        else:
            out[:, j] = np.column_stack([np.sqrt(np.mean(x * x, 1)), x.mean(1)])
    return starts + width, out, rec.segments[starts]


def _bounds(ends, width, h):
    starts = ends - width
    a = np.searchsorted(starts, ends - 2 * h, side='left')
    b = np.searchsorted(starts, ends - h - width, side='right')
    c = np.searchsorted(starts, ends - h, side='left')
    d = np.searchsorted(starts, ends - width, side='right')
    return a, b, c, d, (ends - 2 * h >= starts[0]) & (b - a >= 2) & (d - c >= 2)


def _lag_bounds(ends, width, h, lag):
    starts = ends - width
    a = np.searchsorted(starts, ends - lag - h, side='left')
    b = np.searchsorted(starts, ends - lag - width, side='right')
    c = np.searchsorted(starts, ends - h, side='left')
    d = np.searchsorted(starts, ends - width, side='right')
    return a, b, c, d, (ends - lag - h >= starts[0]) & (b - a >= 2) & (d - c >= 2)


def _kernel_scores(kernel, bounds):
    """Exact V-statistic MMD^2 between descriptor index ranges [a,b) and [c,d)."""
    a, b, c, d, valid = bounds
    out = np.zeros(len(valid))
    a, b, c, d = [x[valid] for x in (a, b, c, d)]
    integral = np.pad(kernel.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def rect(a, b, c, d):
        return integral[b, d] - integral[a, d] - integral[b, c] + integral[a, c]
    nl, nr = b - a, d - c
    score = rect(a, b, a, b) / nl ** 2 + rect(c, d, c, d) / nr ** 2 - 2 * rect(a, b, c, d) / (nl * nr)
    out[valid] = np.maximum(score, 0)
    return out


def raw_profiles(ends, desc, seg, bank, cfg):
    """52 Gaussian + 39 Energy MMD values and 4+3 shared validity masks."""
    opts = cfg['features']
    contexts = opts['gaussian_contexts_seconds'] + opts['energy_contexts_seconds']
    raw = np.zeros((len(ends), 91))
    masks = np.zeros((len(ends), 7), dtype=bool)
    center, scale, bands = (np.asarray(bank[k]) for k in ('center', 'scale', 'bandwidth'))
    for segment in np.unique(seg):
        sel = np.flatnonzero(seg == segment)
        bounds = [_bounds(ends[sel], opts['descriptor_samples'], int(round(h * cfg['sample_rate_hz'])))
                  for h in contexts]
        masks[sel] = np.column_stack([b[-1] for b in bounds])
        for j in range(13):
            x = (desc[sel, j] - center[j]) / scale[j]
            gaussian, energy = gaussian_kernel(x, bands[j], opts['bandwidth_factors']), energy_kernel(x)
            for k in range(4):
                raw[sel, j * 4 + k] = _kernel_scores(gaussian, bounds[k])
            for k in range(3):
                raw[sel, 52 + j * 3 + k] = _kernel_scores(energy, bounds[4 + k])
    return raw, masks


def lag_raw(rec, bank, cfg, start_seconds):
    """Lagged MMD [n, 39] (column j*3+k) and validity [n, 3]; the older window starts >= start_seconds."""
    ends, desc, seg = descriptors(rec, cfg)
    fs, width = cfg['sample_rate_hz'], cfg['features']['descriptor_samples']
    first_allowed = int(np.searchsorted(rec.times, start_seconds))
    center, scale, bands = (np.asarray(bank[k]) for k in ('center', 'scale', 'bandwidth'))
    raw = np.zeros((len(ends), 13 * N_LAG))
    masks = np.zeros((len(ends), N_LAG), dtype=bool)
    for segment in np.unique(seg):
        sel = np.flatnonzero(seg == segment)
        e = ends[sel]
        bounds = []
        for h, lag in cfg['lags']:
            h, lag = int(round(h * fs)), int(round(lag * fs))
            a, b, c, d, valid = _lag_bounds(e, width, h, lag)
            bounds.append((a, b, c, d, valid & (e - lag - h >= first_allowed)))
        masks[sel] = np.column_stack([bd[-1] for bd in bounds])
        for j in range(13):
            kernel = gaussian_kernel((desc[sel, j] - center[j]) / scale[j], bands[j],
                                     cfg['features']['bandwidth_factors'])
            for k, bd in enumerate(bounds):
                raw[sel, j * N_LAG + k] = _kernel_scores(kernel, bd)
    return ends, raw, masks, seg


def expanded_masks(masks):
    """[..., 7] -> [..., 91]: context validity shared by all 13 variables."""
    return np.concatenate([np.tile(masks[..., :4], (1, 13)), np.tile(masks[..., 4:], (1, 13))], axis=-1)


# ----------------------------------------------------------------------------- source-only calibration
def fit_bank(reference, cfg):
    """Scales, bandwidths and score maxima from the SOURCE F0 recording, t >= startup_seconds."""
    opts, t0 = cfg['features'], cfg['startup_seconds']
    ends, desc, seg = descriptors(reference, cfg, t0)
    center = np.median(desc, axis=0)
    amplitude = np.median(np.abs(reference.values[reference.times >= t0]), axis=0)
    floor = np.maximum(opts['relative_floor'] * amplitude,
                       [opts['voltage_floor'] if c in ('Vpv', 'Vdc', 'Vabc') else opts['absolute_floor']
                        for c in VARIABLES])
    scale = np.maximum(1.4826 * np.median(np.abs(desc - center), axis=0), floor[:, None])
    bands = [bandwidth((desc[:, j] - center[j]) / scale[j], opts['max_reference_points'], opts['bandwidth_floor'])
             for j in range(13)]
    bank = dict(reference=reference.name, center=center.tolist(), scale=scale.tolist(), bandwidth=bands)
    raw, masks = raw_profiles(ends, desc, seg, bank, cfg)
    valid = expanded_masks(masks)
    bank['maxima'] = np.maximum([raw[valid[:, j], j].max() for j in range(91)], opts['mmd_floor']).tolist()
    _, lraw, lmask, _ = lag_raw(reference, bank, cfg, t0)  # calibrate the lag channels after start-up too
    lvalid = np.tile(lmask, (1, 13))
    bank['lag_maxima'] = np.maximum([lraw[lvalid[:, c], c].max() for c in range(lraw.shape[1])],
                                    opts['mmd_floor']).tolist()
    return bank


# ----------------------------------------------------------------------------- windows and sequences
def window_metadata(rec, cfg, onset=None):
    """Evaluation windows (200 samples, stride 200) with centre >= start_seconds.
    label = 1 from the onset on (None = all normal); near_onset marks |centre - onset| < guard."""
    width = cfg['evaluation_window_samples']
    starts = np.arange(0, len(rec.times) - width + 1, cfg['evaluation_stride_samples'], dtype=np.int64)
    centers, right = starts + width // 2, starts + width - 1
    keep = (rec.times[centers] >= cfg['start_seconds']) & (rec.segments[starts] == rec.segments[right])
    t = rec.times[centers[keep]]
    label = np.zeros(len(t), dtype=np.int64) if onset is None else (t >= onset).astype(np.int64)
    near = np.zeros(len(t), dtype=bool) if onset is None else np.abs(t - onset) < cfg['onset_guard_seconds']
    return pd.DataFrame(dict(recording=rec.name, right_sample_index=right[keep], center_time=t,
                             label=label, near_onset=near))


def absolute_raw(desc, bank):
    """[n, 16] dimensionless values per descriptor; constants come from the source F0 bank only."""
    center = np.asarray(bank['center'])
    eps = 0.01 * np.abs(center[:, 0]) + 1e-6
    out = [np.log1p(desc[:, j, 1] / (np.abs(desc[:, j, 0]) + eps[j])) for j in LEVEL]
    out += [desc[:, j, 1] / (desc[:, j, 0] + eps[j]) for j in PHASE_I + PHASE_V]
    for group in (PHASE_I, PHASE_V):
        rms = desc[:, group, 0]
        out.append((rms.max(1) - rms.min(1)) / (rms.mean(1) + eps[group].mean()))
    p_ac, p_dc = np.abs(desc[:, 9, 0] * desc[:, 11, 0]), np.abs(desc[:, 0, 0] * desc[:, 1, 0])
    e = 1e-3 * (abs(center[9, 0] * center[11, 0]) + abs(center[0, 0] * center[1, 0])) + 1e-9
    out.append(np.log((p_ac + e) / (p_dc + e)))
    return np.column_stack(out)


def _padded(values, ends, segments, endpoints, length):
    """values [n, c] -> [N, c, length] ending at each endpoint; missing / other-segment steps are zero."""
    idx = np.searchsorted(ends, endpoints)
    safe = np.minimum(idx, len(ends) - 1)
    exact = (idx < len(ends)) & (ends[safe] == endpoints)
    grid = safe[:, None] - np.arange(length - 1, -1, -1)
    ok = (grid >= 0) & exact[:, None]
    ok &= segments[np.maximum(grid, 0)] == segments[safe][:, None]
    x = np.where(ok[:, :, None], values[np.maximum(grid, 0)], 0).transpose(0, 2, 1)
    return x.astype(np.float32), ok, exact


def raw_sequences(rec, bank, cfg, onset=None):
    """Unscaled base [N,98,T], lag [N,42,T], absolute [N,16,T], step [N,1,T] and window metadata."""
    length = cfg['features']['sequence_length']
    ends, desc, seg = descriptors(rec, cfg)
    raw, masks = raw_profiles(ends, desc, seg, bank, cfg)
    base = np.concatenate([np.log1p(raw / np.asarray(bank['maxima'])), masks], 1)
    lag_ends, lraw, lmask, _ = lag_raw(rec, bank, cfg, cfg['start_seconds'])
    lag = np.concatenate([np.log1p(lraw / np.asarray(bank['lag_maxima'])), lmask], 1)
    assert (lag_ends == ends).all()
    meta = window_metadata(rec, cfg, onset)
    endpoints = meta.right_sample_index.to_numpy() + 1
    b, ok, exact = _padded(base, ends, seg, endpoints, length)
    meta, b, ok, endpoints = meta.loc[exact].reset_index(drop=True), b[exact], ok[exact], endpoints[exact]
    lg, _, _ = _padded(lag, ends, seg, endpoints, length)
    absolute, _, _ = _padded(absolute_raw(desc, bank), ends, seg, endpoints, length)
    return b, lg, absolute, ok[:, None, :].astype(np.float32), meta


def _masked_moments(x, valid):
    count = valid.sum((0, 2))
    mean = (x * valid).sum((0, 2), dtype=np.float64) / count
    var = (((x.astype(np.float64) - mean[None, :, None]) ** 2) * valid).sum((0, 2)) / count
    return mean, np.sqrt(var)


def _base_valid(base):
    return expanded_masks(base[:, 91:].transpose(0, 2, 1)).transpose(0, 2, 1) > 0


def _lag_valid(lag):
    return np.tile(lag[:, 13 * N_LAG:] > 0, (1, 13, 1))


def fit_scaler(base, lag, absolute, step):
    """Source-train mean/std of every score, valid entries only."""
    bm, bs = _masked_moments(base[:, :91], _base_valid(base))
    lm, ls = _masked_moments(lag[:, :13 * N_LAG], _lag_valid(lag))
    am, as_ = _masked_moments(absolute, np.broadcast_to(step > 0, absolute.shape))
    return dict(base_mean=bm.tolist(), base_std=bs.tolist(), lag_mean=lm.tolist(), lag_std=ls.tolist(),
                abs_mean=am.tolist(), abs_std=as_.tolist())


def _z(x, mean, std, valid, cfg):
    std = np.maximum(std, cfg['features']['std_floor'])
    z = np.clip((x - np.asarray(mean)[None, :, None]) / std[None, :, None], -cfg['features']['clip'], cfg['features']['clip'])
    return np.where(valid, z, 0)


def assemble(base, lag, absolute, step, scaler, cfg):
    """Scaled [N,157,T]: [91 scores | 39 lag | 16 absolute | 7 masks | 3 lag masks | 1 step-valid]."""
    b = _z(base[:, :91], scaler['base_mean'], np.asarray(scaler['base_std']), _base_valid(base), cfg)
    lg = _z(lag[:, :13 * N_LAG], scaler['lag_mean'], np.asarray(scaler['lag_std']), _lag_valid(lag), cfg)
    a = _z(absolute, scaler['abs_mean'], np.asarray(scaler['abs_std']), np.broadcast_to(step > 0, absolute.shape), cfg)
    return np.concatenate([b, lg, a, base[:, 91:], lag[:, 13 * N_LAG:], step], 1).astype(np.float32)
