"""Joint Gaussian MMD over all 26 descriptors: confirmation for weak local evidence.

One Gaussian-mixture kernel sees the concatenated F0-standardized descriptors,
so a change in cross-channel dependence can register even when every marginal
stays unchanged. Transforms, bandwidth and thresholds come from same-mode F0.
It is evaluated only when the local branches lack a strong event.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


def options(config):
    return config['joint_confirmation']


def matrix(values, transforms, channels):
    return np.column_stack([(values[c] - transforms[c]['center']) / transforms[c]['scale'] for c in channels])


def _context_config(config):
    opts = options(config)
    return {**config, 'contexts_seconds': list(opts['contexts_seconds']), 'min_scales': opts['min_scales']}


def _columns(recording, frames, windows, x, band, reference_max, config):
    """Raw joint MMD², per-context ratios and the geometric-mean score."""
    from .detector import aggregate, profiles
    cfg = _context_config(config)
    transform = {'joint': {'center': np.zeros(x.shape[1]), 'scale': np.ones(x.shape[1]), 'bandwidth': band}}
    raw, valid = profiles(recording, frames, windows, {'joint': x}, transform, cfg)
    if reference_max is None:  # fitting on F0
        if not valid.any(axis=0).all():
            raise ValueError('F0 lacks support for a joint context; reduce joint contexts')
        reference_max = np.array([raw['joint'][valid[:, j], j].max() for j in range(valid.shape[1])])
    columns = aggregate(recording, frames, raw, valid, {'joint': reference_max}, cfg, 'joint_')
    # The single "channel" is the joint vector: keep one score column.
    for key in [k for k in columns if k.startswith('joint_score_') or k.startswith('joint_joint__')]:
        value = columns.pop(key)
        if '__mmd__' in key or '__ratio__' in key:
            columns['joint_' + key.split('joint_joint__')[1]] = value
    return columns, reference_max


@dataclass
class Model:
    config: dict
    transforms: dict
    bandwidth: float
    reference_max: np.ndarray
    threshold: float
    reference_peak: float
    reference_columns: dict

    def score(self, recording, frames, windows, values):
        x = matrix(values, self.transforms, self.config['channels'])
        columns, _ = _columns(recording, frames, windows, x, self.bandwidth, self.reference_max, self.config)
        columns['joint_threshold'] = np.full(len(frames), self.threshold)
        columns['joint_ratio'] = columns['joint_score'] / self.threshold
        return columns


def fit(recording, frames, windows, values, transforms, config, holdout_peak=0.):
    """Calibrate on F0; threshold = multiplier × max(1, in-sample, held-out peak)."""
    from .detector import bandwidth
    x = matrix(values, transforms, config['channels'])
    band = bandwidth(x, config)
    columns, reference_max = _columns(recording, frames, windows, x, band, None, config)
    eligible = columns['joint_valid'].astype(bool)
    if not eligible.any():
        raise ValueError('F0 has no split with enough joint contexts')
    peak = float(columns['joint_score'][eligible].max())
    threshold = config['threshold_multiplier'] * max(peak, config['min_threshold'], holdout_peak)
    columns['joint_threshold'] = np.full(len(frames), threshold)
    columns['joint_ratio'] = columns['joint_score'] / threshold
    return Model(deepcopy(config), deepcopy(transforms), band, reference_max, threshold, peak, columns)


def needs_score(table, config, local):
    """True unless local evidence already yields a strong event.

    Shared by scoring and selection so that skipping the branch can never
    change the decision.
    """
    if local is not None and local[0] == 'short_energy':
        return False
    valid = table.valid.to_numpy(bool)
    return not (valid.any() and table.loc[valid, 'ratio'].max() >= options(config)['max_local_ratio'])


def select(table, config, local):
    """Keep a strong local event; otherwise require joint confirmation or return None."""
    if not needs_score(table, config, local):
        return local
    if 'joint_score' not in table:
        raise ValueError('Joint evidence is required by the selection gate but was not computed')
    from .selection import maximum
    opts = options(config)
    confirmation = np.zeros(len(table), dtype=int)
    for context in opts['contexts_seconds']:
        tag = f'{context:g}s'
        confirmation += (table[f'joint_valid__{tag}'].to_numpy(bool)
                         & (table[f'joint_ratio__{tag}'].to_numpy() > opts['context_ratio']))
    view = table.assign(joint_valid=table.joint_valid.to_numpy(bool)
                        & (confirmation >= opts['min_confirming_contexts']))
    # Early edge (>= 95 % of the peak, <= 50 ms) as for the long Gaussian branch:
    # joint also uses long two-sided contexts (0.5-2 s) and is reached only when
    # local evidence is weak. The peak may occur after a gradual change begins.
    chosen = maximum(view, 'joint_', config, early_edge=True)
    return None if chosen is None else ('joint_rbf', *chosen, 0)
