"""Invariants of the MMD onset method on small synthetic recordings."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mmd_onset.configuration import validate
from mmd_onset.data import CHANNELS, Recording
from mmd_onset.detector import fit, split_recording
from mmd_onset.refinement import CycleResidualRefiner, mean_shift_curve
from mmd_onset.selection import maximum, separate
from mmd_onset.workflow import select_onset

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config.yaml').read_text(encoding='utf-8'))


def config():
    return deepcopy(CONFIG)


def recording(name='F0L', *, seconds=12., fs=1000., change_time=None, seed=2043):
    """1 kHz synthetic PV/grid record; optional +2 step on the DC channels."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    time = .5 + np.arange(n) / fs
    base = {'Ipv': 2., 'Vpv': 90., 'Vdc': 300., 'Iabc': 2., 'If': 50., 'Vabc': 100., 'Vf': 50.}
    values = {}
    for channel in CHANNELS:
        if channel in base:
            signal = base[channel] + .015 * rng.normal(size=n)
        else:
            phase = {'a': 0., 'b': -2 * np.pi / 3, 'c': 2 * np.pi / 3}[channel[-1]]
            amplitude = 2. if channel[0] == 'i' else 100.
            signal = amplitude * np.sin(2 * np.pi * 50 * time + phase) + .015 * rng.normal(size=n)
        if change_time is not None and channel in {'Ipv', 'Vpv', 'Vdc', 'Iabc'}:
            signal = signal + 2. * (time >= change_time)
        values[channel] = signal
    return Recording(name, pd.DataFrame(values), time, time.copy(),
                     np.arange(int(.5 * fs), int(.5 * fs) + n), np.zeros(n, dtype=int), 1. / fs)


def test_config_is_valid_and_sets_expected_defaults():
    cfg = validate(config())
    assert cfg['features']['ac'] == ['rms', 'mean']
    assert cfg['calibration']['holdout_halves'] is True
    assert cfg['refinement']['enabled'] is True
    broken = config()
    broken['graph_fallback'] = {}
    with pytest.raises(ValueError, match='unknown'):
        validate(broken)


@pytest.fixture(scope='module')
def model():
    return fit(recording('F0L'), config())


def table_of(model, target):
    frames, columns = model.score(target)
    return pd.concat([frames, pd.DataFrame(columns, index=frames.index)], axis=1)


def test_step_is_detected_and_refined_to_the_sample(model):
    target = recording('F1L', change_time=7.0, seed=7)
    refiner = CycleResidualRefiner.fit(model.reference, config())
    onset, curve = select_onset(target, table_of(model, target), config(), refiner)
    assert len(onset) == 1 and onset.iloc[0].branch == 'short_energy'
    change = int(target.sample_index[np.searchsorted(target.time, 7.0)])
    assert onset.iloc[0].refinement_status == 'refined'
    assert abs(int(onset.iloc[0].sample_index) - change) <= 1
    assert onset.iloc[0].coarse_sample_index < onset.iloc[0].sample_index  # coarse grid is early
    assert curve.selected.sum() == 1


def grid_event(target, start, peak):
    """Coarse Energy event on a 10 ms grid: (onset row, score table)."""
    times = np.round(np.arange(6.90, 7.10, .01), 3)
    rows = [int(np.searchsorted(target.time, t)) for t in times]
    table = pd.DataFrame({'original_split': target.sample_index[rows], 'segment_id': 0})
    onset = pd.Series({'branch': 'short_energy', 'group_index': int(np.argmin(abs(times - start))),
                       'peak_group_index': int(np.argmin(abs(times - peak)))})
    return onset, table


def test_ripple_before_the_transition_does_not_end_the_bracket(model):
    # Energy rises from 6.97 s, has a small non-prominent
    # ripple at 6.98 s and peaks (prominence-filtered) at 7.01 s; the raw step
    # is at 7.00 s. The bracket must reach the event peak, not the ripple.
    target = recording('F1L', change_time=7.0, seed=7)
    refiner = CycleResidualRefiner.fit(model.reference, config())
    onset, table = grid_event(target, start=6.97, peak=7.01)
    result, _ = refiner.refine(target, onset, table)
    change = int(target.sample_index[np.searchsorted(target.time, 7.0)])
    assert result['status'] == 'refined' and abs(result['sample_index'] - change) <= 1


def test_cut_at_the_bracket_edge_keeps_the_coarse_point(model):
    # If the bracket ended at the ripple (old behaviour), the step would sit at its
    # right edge: the refined sample is then not trusted.
    target = recording('F1L', change_time=7.0, seed=7)
    refiner = CycleResidualRefiner.fit(model.reference, config())
    onset, table = grid_event(target, start=6.97, peak=6.98)
    result, curve = refiner.refine(target, onset, table)
    assert result['status'] == 'kept_coarse_unreliable' and curve.empty


def test_joint_selection_uses_the_early_edge():
    # Joint MMD with long two-sided contexts moves back from its
    # peak while the score stays >= 95 % of it, for at most 50 ms.
    from mmd_onset.joint import select as joint_select
    score = np.array([.5, 1.5, 2.9, 2.95, 3.0, 1.0])
    table = pd.DataFrame({'valid': 1, 'ratio': .5, 'segment_id': 0,
                          'split_elapsed': 1. + .01 * np.arange(6),
                          'joint_valid': 1, 'joint_score': score, 'joint_ratio': score})
    for context in CONFIG['joint_confirmation']['contexts_seconds']:
        table[f'joint_valid__{context:g}s'] = 1
        table[f'joint_ratio__{context:g}s'] = score
    assert joint_select(table, config(), None) == ('joint_rbf', 2, 4, 0)


def test_saved_onset_validation_catches_corrupted_coordinates(model):
    from mmd_onset.workflow import validate_saved_onset
    target = recording('F1L', change_time=7.0, seed=7)
    refiner = CycleResidualRefiner.fit(model.reference, config())
    table = table_of(model, target)
    onset, curve = select_onset(target, table, config(), refiner)
    validate_saved_onset(target, table, onset, curve)  # untouched output passes
    for column, change in (('sample_index', 1), ('Time', 1e-4), ('coarse_sample_index', 10), ('ratio', .1)):
        broken = onset.copy()
        broken.loc[0, column] = broken.loc[0, column] + change
        with pytest.raises(ValueError, match='failed checks|not a retained'):
            validate_saved_onset(target, table, broken, curve)
    with pytest.raises(ValueError, match='normal reference'):
        validate_saved_onset(recording('F0L'), table, onset, curve)


def test_healthy_recording_gives_no_onset(model):
    target = recording('F1L', seed=11)
    onset, _ = select_onset(target, table_of(model, target), config())
    assert onset.empty


def test_f0_in_sample_ratios_stay_below_threshold(model):
    columns = model.reference_columns
    assert columns['ratio'].max() <= 1 / CONFIG['threshold_multiplier'] + 1e-12
    assert model.threshold >= CONFIG['threshold_multiplier']


def test_holdout_halves_preserve_original_coordinates():
    source = recording('F0L', seconds=3.)
    first, second = split_recording(source)
    assert np.array_equal(np.r_[first.sample_index, second.sample_index], source.sample_index)
    assert first.elapsed[-1] < second.elapsed[0]


def test_mean_shift_gain_matches_brute_force_sse():
    y = np.random.default_rng(5).normal(size=40)
    y[23:] += 3.
    cut, gain, signed = mean_shift_curve(y)
    for k, g in zip(cut, gain):
        left, right = y[:k], y[k:]
        sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        assert np.isclose(((y - y.mean()) ** 2).sum() - sse, g)
    assert cut[np.argmax(signed)] == 23


def test_all_zero_profile_selects_nothing():
    table = pd.DataFrame({'valid': [1] * 4, 'score': [0.] * 4, 'ratio': [2.] * 4,
                          'segment_id': [0] * 4, 'split_elapsed': [1., 1.01, 1.02, 1.03]})
    assert maximum(table, '', config()) is None


def test_peak_separation_keeps_the_higher_of_close_survivors():
    peaks, _ = separate(np.array([10, 13, 30]), np.array([9., 8., 7.]), np.array([5., 4., 3.]), 5)
    assert list(peaks) == [10, 30]
