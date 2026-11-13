"""CSV -> onset workflow: versioned runs, saved evidence and figure re-rendering."""
from copy import deepcopy
from pathlib import Path
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from .configuration import METHOD, load_config, save_config, validate
from .data import read_recording
from .detector import feature_group, fit
from .refinement import CURVE_COLUMNS, CycleResidualRefiner
from .selection import select
from .plotting import plot_comparison, plot_refinement, plot_score

ONSET_COLUMNS = [
    'file', 'mode', 'sample_index', 'Time', 'elapsed', 'branch', 'score', 'threshold', 'ratio',
    'dominant_channel', 'dominant_feature', 'supporting_channels', 'segment_id',
    'group_index', 'peak_group_index', 'peak_elapsed', 'peak_ratio', 'strong_candidates',
    'coarse_sample_index', 'coarse_Time', 'coarse_elapsed', 'refinement_status',
    'bracket_start_sample_index', 'bracket_end_sample_index', 'shift_z',
]
PREFIX = {'long_gaussian': '', 'short_energy': 'short_', 'joint_rbf': 'joint_'}


def read_source(config, name):
    return read_recording(Path(config['data_dir']) / f'{name}.csv', config['warmup_seconds'],
                          config['gap_factor'], config['channels'])


def run_path(config):
    target = (Path(config['output_dir']) / METHOD / config['run_name']).resolve()
    data = Path(config['data_dir']).resolve()
    if target == data or target.is_relative_to(data) or data.is_relative_to(target):
        raise ValueError('Run output must not overlap the source CSV folder')
    return target


def _attribution(row, branch, config):
    """Dominant channel and the channels whose own score exceeds the threshold."""
    if branch == 'joint_rbf':
        return 'joint', 'joint Gaussian MMD (all channels)', ''
    prefix = PREFIX[branch]
    channel = max(config['channels'], key=lambda c: row[f'{prefix}score_{c}'])
    support = ';'.join(c for c in config['channels'] if row[f'{prefix}score_{c}'] > row[f'{prefix}threshold'])
    return channel, f'{channel} ({"/".join(config["features"][feature_group(channel)])})', support


def select_onset(recording, table, config, refiner=None):
    """One confirmed onset row (possibly refined to a raw sample), or an empty frame."""
    chosen = select(table, recording, config)
    if chosen is None:
        return pd.DataFrame(columns=ONSET_COLUMNS), pd.DataFrame(columns=CURVE_COLUMNS)
    branch, index, peak, count = chosen
    prefix, row = PREFIX[branch], table.iloc[index]
    channel, feature, support = _attribution(row, branch, config)
    onset = {'file': recording.name, 'mode': recording.name[-1],
             'sample_index': int(row.original_split), 'Time': float(row.split_time),
             'elapsed': float(row.split_elapsed), 'branch': branch,
             'score': float(row[f'{prefix}score']), 'threshold': float(row[f'{prefix}threshold']),
             'ratio': float(row[f'{prefix}ratio']), 'dominant_channel': channel,
             'dominant_feature': feature, 'supporting_channels': support,
             'segment_id': int(row.segment_id), 'group_index': index, 'peak_group_index': peak,
             'peak_elapsed': float(table.split_elapsed.iloc[peak]),
             'peak_ratio': float(table[f'{prefix}ratio'].iloc[peak]), 'strong_candidates': count,
             'coarse_sample_index': int(row.original_split), 'coarse_Time': float(row.split_time),
             'coarse_elapsed': float(row.split_elapsed), 'refinement_status': 'disabled',
             'bracket_start_sample_index': -1, 'bracket_end_sample_index': -1, 'shift_z': np.nan}
    curve = pd.DataFrame(columns=CURVE_COLUMNS)
    if refiner is not None:
        result, curve = refiner.refine(recording, pd.Series(onset), table)
        onset['refinement_status'] = result.pop('status')
        onset.update(result)
    return pd.DataFrame([onset], columns=ONSET_COLUMNS), curve


def validate_saved_onset(recording, table, onset, curve):
    """Check an onset row against the raw CSV coordinates and its saved evidence.

    Called before a run is written, when figures are re-rendered and before
    evaluation, so every exported coordinate is traceable to one source row.
    """
    name = recording.name
    if name.startswith('F0') and len(onset):
        raise ValueError(f'{name}: the normal reference must not have an onset')
    if len(onset) > 1:
        raise ValueError(f'{name}: at most one onset per recording')
    if onset.empty:
        return
    row = onset.iloc[0]

    def raw(index, label):
        """Position of a CSV row; its Time/elapsed must match the saved values."""
        position = int(np.searchsorted(recording.sample_index, int(index)))
        if position >= len(recording.sample_index) or recording.sample_index[position] != int(index):
            raise ValueError(f'{name}: {label} sample {int(index)} is not a retained CSV row')
        return position

    coarse = raw(row.coarse_sample_index, 'coarse')
    primary = raw(row.sample_index, 'onset')
    group = table.iloc[int(row.group_index)]
    prefix = PREFIX[row.branch]
    checks = {
        'coarse index = saved grid split': int(group.original_split) == int(row.coarse_sample_index),
        'coarse Time/elapsed = CSV': (recording.time[coarse] == row.coarse_Time
                                      and recording.elapsed[coarse] == row.coarse_elapsed),
        'onset Time/elapsed = CSV': (recording.time[primary] == row.Time
                                     and recording.elapsed[primary] == row.elapsed),
        'onset in the coarse segment': int(recording.segment_id[primary]) == int(row.segment_id)
                                       == int(group.segment_id),
        'score/threshold/ratio = saved evidence': np.allclose(
            [row.score, row.threshold, row.ratio],
            [group[f'{prefix}score'], group[f'{prefix}threshold'], group[f'{prefix}ratio']],
            rtol=1e-12, atol=0) and row.ratio > 1,
    }
    if row.refinement_status == 'refined':
        selected = curve.loc[curve.selected.astype(bool)]
        checks['refined onset inside its bracket'] = (
            int(row.bracket_start_sample_index) <= int(row.sample_index) < int(row.bracket_end_sample_index))
        checks['refined onset = selected curve row'] = (
            len(selected) == 1 and int(selected.sample_index.iloc[0]) == int(row.sample_index))
    else:
        checks['unrefined onset = coarse point'] = int(row.sample_index) == int(row.coarse_sample_index)
    failed = [label for label, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f'{name}: saved onset failed checks: {"; ".join(failed)}')


def _peak(table, prefix):
    if f'{prefix}score' not in table or not table[f'{prefix}valid'].any():
        return np.nan, np.nan
    positions = np.flatnonzero(table[f'{prefix}valid'].to_numpy(bool))
    peak = int(positions[np.argmax(table[f'{prefix}score'].to_numpy()[positions])])
    return float(table.split_elapsed.iloc[peak]), float(table[f'{prefix}ratio'].iloc[peak])


def diagnostic(name, table, onset):
    """One row per recording; sub-threshold evidence is reported here, never as an onset."""
    result = {'file': name, 'role': 'normal_reference' if name.startswith('F0') else 'target',
              'status': 'onset' if len(onset) else 'no_confirmed_onset',
              'branch': onset.iloc[0].branch if len(onset) else '',
              'joint_evaluated': bool(table.joint_evaluated.iloc[0])}
    for prefix, key in (('', 'long'), ('short_', 'short'), ('joint_', 'joint')):
        result[f'{key}_peak_elapsed'], result[f'{key}_peak_ratio'] = _peak(table, prefix)
    return result


def model_snapshot(model, refiner):
    result = {
        'reference': f'F0{model.mode}',
        'thresholds': {'long_gaussian': model.threshold, 'short_energy': model.short_threshold,
                       'joint': model.joint_model.threshold if model.joint_model else None},
        'in_sample_peaks': {'long_gaussian': model.reference_peak, 'short_energy': model.short_reference_peak,
                            'joint': model.joint_model.reference_peak if model.joint_model else None},
        'holdout_halves_peaks': dict(model.holdout_peaks),
        'reference_max_mmd': {c: v.tolist() for c, v in model.reference_max.items()},
        'short_reference_max_mmd': {c: v.tolist() for c, v in model.short_reference_max.items()},
        'channels': {c: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in item.items()}
                     for c, item in model.transforms.items()},
    }
    if model.joint_model is not None:
        result['joint'] = {'bandwidth': model.joint_model.bandwidth,
                           'reference_max_mmd': model.joint_model.reference_max.tolist()}
    if refiner is not None:
        result['refinement'] = refiner.snapshot()
    return result


def _unique_run(config, overwrite):
    if overwrite:
        return config
    result, version = deepcopy(config), 1
    while run_path(result).exists():
        version += 1
        result['run_name'] = f'{config["run_name"]}_v{version}'
    return result


def run_method(config, overwrite=False):
    """Fit F0 per mode, score every recording blind, save evidence and onsets."""
    config = _unique_run(config, overwrite)
    target = run_path(config)
    modes = sorted({name[-1] for name in config['recordings']})
    names = list(dict.fromkeys([*[f'F0{m}' for m in modes], *config['recordings']]))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{config["run_name"]}.', dir=target.parent))
    try:
        for folder in ('scores', 'onsets', 'refinement', 'figures', 'source_snapshot'):
            (stage / folder).mkdir()
        save_config(stage / 'config_used.yaml', config)
        for path in Path(__file__).parent.iterdir():
            if path.is_file() and path.suffix in {'.py', '.yaml'}:
                shutil.copyfile(path, stage / 'source_snapshot' / path.name)
        references, models, refiners = {}, {}, {}
        for mode in modes:
            references[mode] = read_source(config, f'F0{mode}')
            models[mode] = fit(references[mode], config)
            refiners[mode] = (CycleResidualRefiner.fit(references[mode], config)
                              if config['refinement']['enabled'] else None)
            print(f'[{METHOD}] F0{mode}: thresholds long={models[mode].threshold:.4g} '
                  f'short={models[mode].short_threshold:.4g}', flush=True)
        diagnostics = {}
        for name in names:
            mode = name[-1]
            model = models[mode]
            if name.startswith('F0'):
                # F0 takes the same decision path as a target (no file-name shortcut);
                # its in-sample joint evidence is needed by the selection gate.
                recording = references[mode]
                model.ensure_joint()
                frames, columns = model.reference_extract[0], model.reference_columns
            else:
                recording = read_source(config, name)
                frames, columns = model.score(recording)
            table = pd.concat([frames.copy(), pd.DataFrame(columns, index=frames.index)], axis=1)
            if not np.isfinite(table.to_numpy(float)).all():
                raise ValueError(f'{name}: non-finite evidence')
            onset, curve = select_onset(recording, table, config, refiners[mode])
            validate_saved_onset(recording, table, onset, curve)
            diagnostics[name] = diagnostic(name, table, onset)
            table.to_csv(stage / 'scores' / f'{name}.windows.csv', index=False)
            onset.to_csv(stage / 'onsets' / f'{name}.csv', index=False)
            curve.to_csv(stage / 'refinement' / f'{name}.csv', index=False)
            if not name.startswith('F0'):
                _figures(stage, references[mode], recording, table, onset, curve, config)
            detail = ('no confirmed onset' if onset.empty else
                      f'onset={onset.iloc[0].elapsed:.6f}s ({onset.iloc[0].branch}, '
                      f'ratio={onset.iloc[0].ratio:.3f}, {onset.iloc[0].refinement_status})')
            print(f'[{METHOD}] {name}: {detail}', flush=True)
        save_config(stage / 'reference_models.yaml',
                    {m: model_snapshot(models[m], refiners[m]) for m in modes})
        pd.DataFrame([diagnostics[n] for n in names]).to_csv(stage / 'selection_diagnostics.csv', index=False)
        if target.exists():
            shutil.rmtree(target)
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f'Output: {target}', flush=True)
    return target


def _figures(folder, reference, recording, table, onset, curve, config):
    name = recording.name
    plot_comparison(reference, recording, onset, folder / 'figures' / f'{name}_vs_{reference.name}.png', config)
    if config['plot']['save_score_figures']:
        plot_score(recording, table, onset, folder / 'figures' / f'{name}_scores.png', config)
    if not curve.empty:
        plot_refinement(recording, curve, onset, folder / 'figures' / f'{name}_refinement.png', config)


def read_saved(target, recording):
    """Saved (scores, onset, refinement curve) of one recording, validated against its CSV."""
    def read(path):
        return pd.read_csv(path, float_precision='round_trip')
    name = recording.name
    table = read(target / 'scores' / f'{name}.windows.csv')
    onset = read(target / 'onsets' / f'{name}.csv')
    curve = read(target / 'refinement' / f'{name}.csv')
    validate_saved_onset(recording, table, onset, curve)
    return table, onset, curve


def plot_method(config):
    """Re-render figures from saved evidence; nothing is fitted or rescored."""
    target = run_path(config)
    saved = load_config(target / 'config_used.yaml')
    rendering = {**saved, 'plot': deepcopy(config['plot'])}
    references = {}
    for name in config['recordings']:
        if name.startswith('F0'):
            continue
        if name not in saved['recordings']:
            raise ValueError(f'{name}: not part of the saved run')
        mode = name[-1]
        references.setdefault(mode, read_source(saved, f'F0{mode}'))
        recording = read_source(saved, name)
        table, onset, curve = read_saved(target, recording)
        _figures(target, references[mode], recording, table, onset, curve, rendering)
    print(f'Figures: {target / "figures"}', flush=True)
    return target


def main(argv=None, *, plot_only=False):
    parser = argparse.ArgumentParser(description='13-channel MMD onset localization on GPVS-Faults')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--recordings', nargs='+')
    parser.add_argument('--run-name')
    if not plot_only:
        parser.add_argument('--overwrite', action='store_true', help='replace an existing run')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.recordings is not None:
            config['recordings'] = args.recordings
        if args.run_name is not None:
            config['run_name'] = args.run_name
        validate(config)
        return plot_method(config) if plot_only else run_method(config, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1) from None
