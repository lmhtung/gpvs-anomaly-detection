"""Strict validation of config.yaml and resolution of relative paths."""
from __future__ import annotations

import math
import re
from pathlib import Path

from .data import CHANNELS

METHOD = 'mmd_onset'
METHOD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = METHOD_DIR.parent

SCHEMA = {
    'method': None, 'data_dir': None, 'output_dir': None, 'run_name': None, 'recordings': None,
    'warmup_seconds': None, 'gap_factor': None, 'window_seconds': None, 'hop_seconds': None,
    'channels': None, 'features': {'dc', 'ac'},
    'normalization': {'absolute_floor', 'relative_floor', 'mad_multiplier'},
    'calibration': {'holdout_halves'},
    'kernel': {'bandwidth_factors', 'bandwidth_floor', 'max_reference_points'},
    'contexts_seconds': None, 'min_scales': None, 'mmd_floor': None,
    'threshold_multiplier': None, 'min_threshold': None,
    'onset': {'search_start_seconds', 'search_end_margin_seconds', 'short_contexts_seconds',
              'short_min_scales', 'strong_peak_ratio', 'min_prominence_ratio', 'prominence_fraction',
              'min_peak_separation_seconds', 'confirmation_ratio', 'early_fraction', 'max_early_seconds'},
    'joint_confirmation': {'contexts_seconds', 'min_scales', 'max_local_ratio',
                           'min_confirming_contexts', 'context_ratio'},
    'refinement': {'enabled', 'grid_frequency_hz', 'residual_clip', 'min_side_seconds'},
    'plot': {'reference_alpha', 'target_alpha', 'linewidth', 'figsize', 'dpi',
             'max_points_per_channel', 'save_score_figures'},
}


def _fields(value, name, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        found = set(value) if isinstance(value, dict) else set()
        raise ValueError(f'{name}: missing {sorted(set(expected) - found)}, unknown {sorted(found - set(expected))}')
    return value


def _number(value, name, lower=None, upper=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name}: expected a finite number')
    if lower is not None and (value <= lower if strict else value < lower):
        raise ValueError(f'{name}: must be {">" if strict else ">="} {lower}')
    if upper is not None and value > upper:
        raise ValueError(f'{name}: must be <= {upper}')
    return float(value)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name}: expected an integer >= {minimum}')
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f'{name}: expected true/false')


def _contexts(value, name, minimum, count=1):
    if not isinstance(value, list) or len(value) < count or len(set(value)) != len(value):
        raise ValueError(f'{name}: expected >= {count} distinct values')
    for item in value:
        _number(item, name, lower=minimum)
    return value


def _floor(spec, name, strict):
    _fields(spec, name, {'default', 'channels'})
    if not isinstance(spec['channels'], dict) or not set(spec['channels']) <= set(CHANNELS):
        raise ValueError(f'{name}.channels: unknown channel')
    for value in [spec['default'], *spec['channels'].values()]:
        _number(value, name, lower=0, strict=strict)


def validate(config):
    _fields(config, 'config', SCHEMA)
    for key, fields in SCHEMA.items():
        if fields is not None:
            _fields(config[key], key, fields)
    if config['method'] != METHOD:
        raise ValueError(f'method must be {METHOD}')
    for key in ('data_dir', 'output_dir'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key}: expected a path')
    if not isinstance(config['run_name'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', config['run_name']):
        raise ValueError('run_name: expected a single safe directory name')
    names = config['recordings']
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not re.fullmatch(r'F[0-7][LM]', n) for n in names)):
        raise ValueError('recordings: expected unique names F0L..F7M')
    channels = config['channels']
    if not isinstance(channels, list) or not channels or len(set(channels)) != len(channels) \
            or not set(channels) <= set(CHANNELS):
        raise ValueError(f'channels: expected unique names from {CHANNELS}')
    _number(config['warmup_seconds'], 'warmup_seconds', lower=0)
    _number(config['gap_factor'], 'gap_factor', lower=1, strict=True)
    width = _number(config['window_seconds'], 'window_seconds', lower=0, strict=True)
    hop = _number(config['hop_seconds'], 'hop_seconds', lower=0, strict=True, upper=width)
    for group, features in config['features'].items():
        if not isinstance(features, list) or not features or not set(features) <= {'mean', 'std', 'rms'}:
            raise ValueError(f'features.{group}: choose from mean, std, rms')
    normal = config['normalization']
    _floor(normal['absolute_floor'], 'absolute_floor', strict=True)
    _floor(normal['relative_floor'], 'relative_floor', strict=False)
    _number(normal['mad_multiplier'], 'mad_multiplier', lower=0, strict=True)
    _boolean(config['calibration']['holdout_halves'], 'calibration.holdout_halves')
    kernel = config['kernel']
    _contexts(kernel['bandwidth_factors'], 'kernel.bandwidth_factors', 0)
    _number(kernel['bandwidth_floor'], 'kernel.bandwidth_floor', lower=0, strict=True)
    _integer(kernel['max_reference_points'], 'kernel.max_reference_points', 2)
    contexts = _contexts(config['contexts_seconds'], 'contexts_seconds', width + hop)
    if _integer(config['min_scales'], 'min_scales') > len(contexts):
        raise ValueError('min_scales exceeds the number of contexts')
    for key in ('mmd_floor', 'min_threshold'):
        _number(config[key], key, lower=0, strict=True)
    _number(config['threshold_multiplier'], 'threshold_multiplier', lower=1)

    onset = config['onset']
    for key in ('search_start_seconds', 'search_end_margin_seconds', 'min_prominence_ratio',
                'min_peak_separation_seconds', 'max_early_seconds'):
        _number(onset[key], f'onset.{key}', lower=0)
    for key in ('prominence_fraction', 'early_fraction'):
        _number(onset[key], f'onset.{key}', lower=0, upper=1, strict=True)
    for key in ('strong_peak_ratio', 'confirmation_ratio'):
        _number(onset[key], f'onset.{key}', lower=1)
    short = _contexts(onset['short_contexts_seconds'], 'onset.short_contexts_seconds', width + hop, count=2)
    if not 2 <= _integer(onset['short_min_scales'], 'onset.short_min_scales') <= len(short):
        raise ValueError('onset.short_min_scales must lie in [2, number of short contexts]')

    joint = config['joint_confirmation']
    joint_contexts = _contexts(joint['contexts_seconds'], 'joint_confirmation.contexts_seconds', width + hop)
    for key in ('min_scales', 'min_confirming_contexts'):
        if _integer(joint[key], f'joint_confirmation.{key}') > len(joint_contexts):
            raise ValueError(f'joint_confirmation.{key} exceeds the number of contexts')
    for key in ('max_local_ratio', 'context_ratio'):
        _number(joint[key], f'joint_confirmation.{key}', lower=1)

    refinement = config['refinement']
    _boolean(refinement['enabled'], 'refinement.enabled')
    for key in ('grid_frequency_hz', 'residual_clip', 'min_side_seconds'):
        _number(refinement[key], f'refinement.{key}', lower=0, strict=True)

    plot = config['plot']
    for key in ('reference_alpha', 'target_alpha'):
        _number(plot[key], f'plot.{key}', lower=0, upper=1, strict=True)
    _number(plot['linewidth'], 'plot.linewidth', lower=0, strict=True)
    if not isinstance(plot['figsize'], list) or len(plot['figsize']) != 2:
        raise ValueError('plot.figsize: expected [width, height]')
    for value in plot['figsize']:
        _number(value, 'plot.figsize', lower=0, strict=True)
    _integer(plot['dpi'], 'plot.dpi')
    _integer(plot['max_points_per_channel'], 'plot.max_points_per_channel', 0)
    _boolean(plot['save_score_figures'], 'plot.save_score_figures')
    return config


def load_config(path):
    import yaml
    path = Path(path)
    try:
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ValueError(f'{path}: invalid YAML') from exc
    validate(config)
    for key in ('data_dir', 'output_dir'):
        value = Path(config[key]).expanduser()
        config[key] = str((value if value.is_absolute() else PROJECT_ROOT / value).resolve())
    return config


def save_config(path, config):
    import yaml
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
