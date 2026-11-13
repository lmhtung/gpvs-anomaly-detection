"""Figures from saved evidence only (no detector is run here)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LAYOUT = (('Ipv', 'va'), ('Vpv', 'vb'), ('Vdc', 'vc'), ('ia', 'Vabc'),
          ('ib', 'If'), ('ic', 'Vf'), ('Iabc', None))
UNITS = {c: 'A' for c in ('Ipv', 'ia', 'ib', 'ic', 'Iabc')}
UNITS.update({c: 'V' for c in ('Vpv', 'Vdc', 'va', 'vb', 'vc', 'Vabc')})
UNITS.update(If='Hz', Vf='Hz')
PRIMARY, COARSE = '#d62728', '#c58b2b'


def _line(axis, x, y, segments, max_points, valid=None, **style):
    """Uniformly thinned line, broken at timestamp gaps and invalid rows."""
    x, y, segments = np.asarray(x), np.asarray(y), np.asarray(segments)
    positions = np.flatnonzero(np.ones(len(x), bool) if valid is None else np.asarray(valid, bool))
    if not len(positions):
        return
    breaks = (np.diff(positions) != 1) | (segments[positions[1:]] != segments[positions[:-1]])
    xs, ys = [], []
    for block in np.split(positions, np.flatnonzero(breaks) + 1):
        budget = len(block) if not max_points else max(2, round(max_points * len(block) / len(positions)))
        chosen = block[np.linspace(0, len(block) - 1, min(len(block), budget), dtype=int)]
        xs.extend([*x[chosen], np.nan])
        ys.extend([*y[chosen], np.nan])
    axis.plot(xs, ys, **style)


def _markers(axis, onset):
    for row in onset.itertuples(index=False):
        axis.axvline(float(row.elapsed), color=PRIMARY, linestyle='--', linewidth=.9)
        if abs(float(row.coarse_elapsed) - float(row.elapsed)) > 1e-12:
            axis.axvline(float(row.coarse_elapsed), color=COARSE, linestyle='-.', linewidth=.8)


def _note(onset):
    if onset.empty:
        return 'No confirmed onset\n(sub-threshold evidence: selection_diagnostics.csv)'
    row = onset.iloc[0]
    return (f'Onset: {row.elapsed:.6f} s (sample {int(row.sample_index)})\n'
            f'Grid (coarse): {row.coarse_elapsed:.6f} s\nBranch: {row.branch}\n'
            f'Score / F0 threshold: {row.ratio:.3f}\nRefinement: {row.refinement_status}\n'
            'Red: onset; amber: coarse grid point')


def _save(figure, output, config):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(output, dpi=config['plot']['dpi'])


def plot_comparison(reference, recording, onset, output, config):
    """Same-mode F0 and target waveforms of all channels with the onset marked."""
    cfg, channels = config['plot'], config['channels']
    layout = LAYOUT if set(channels) == set(UNITS) else tuple(
        zip((channels + [None])[::2], (channels + [None, None])[1::2]))
    figure, axes = plt.subplots(len(layout), 2, figsize=cfg['figsize'], sharex=True, squeeze=False)
    try:
        for row, pair in enumerate(layout):
            for col, channel in enumerate(pair):
                axis = axes[row, col]
                if channel is None:
                    axis.axis('off')
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    axis.legend(handles, labels, loc='upper left', fontsize=8, frameon=False)
                    axis.text(.03, .6, _note(onset), transform=axis.transAxes, va='top', fontsize=8)
                    continue
                for data, color, alpha, label in (
                        (reference, '#1f77b4', cfg['reference_alpha'], f'{reference.name}: normal reference'),
                        (recording, '#2ca02c', cfg['target_alpha'], f'{recording.name}: target')):
                    _line(axis, data.elapsed, data.values[channel], data.segment_id,
                          cfg['max_points_per_channel'], color=color, alpha=alpha,
                          linewidth=cfg['linewidth'], label=label)
                _markers(axis, onset)
                axis.set_ylabel(f'{channel} ({UNITS.get(channel, "")})', fontsize=9)
                axis.ticklabel_format(axis='both', style='plain', useOffset=False)
                axis.tick_params(labelsize=8)
                axis.grid(alpha=.15)
        for col in range(2):
            active = [row for row, pair in enumerate(layout) if pair[col] is not None]
            axes[active[-1], col].set_xlabel('Elapsed from recording start (s)')
            axes[active[-1], col].tick_params(labelbottom=True)
        figure.suptitle(f'{recording.name} vs {reference.name} | {config["run_name"]}')
        _save(figure, output, config)
    finally:
        plt.close(figure)


def plot_score(recording, table, onset, output, config):
    """Score of every evaluated branch against its F0 threshold."""
    branches = [(p, label) for p, label in (('', 'Long-context Gaussian MMD'),
                                            ('short_', 'Short-context Energy MMD'),
                                            ('joint_', 'Joint Gaussian MMD')) if f'{p}score' in table]
    figure, axes = plt.subplots(len(branches), 1, figsize=(12, 3.5 * len(branches)), sharex=True, squeeze=False)
    cfg = config['plot']
    try:
        for axis, (prefix, title) in zip(axes[:, 0], branches):
            signals = ([('joint', f'{prefix}score')] if prefix == 'joint_' else
                       [(c, f'{prefix}score_{c}') for c in config['channels']])
            for label, column in signals:
                _line(axis, table.split_elapsed, table[column], table.segment_id,
                      cfg['max_points_per_channel'], table[f'{prefix}valid'], label=label, linewidth=.8)
            axis.axhline(float(table[f'{prefix}threshold'].iloc[0]), color='black', linestyle='--',
                         label='F0 threshold')
            _markers(axis, onset)
            axis.set(title=title, ylabel='Score')
            axis.legend(ncol=5, fontsize=7)
            axis.grid(alpha=.2)
        axes[-1, 0].set_xlabel('Elapsed from recording start (s)')
        figure.suptitle(f'{recording.name} | {config["run_name"]}\n{_note(onset).splitlines()[0]}', fontsize=10)
        _save(figure, output, config)
    finally:
        plt.close(figure)


def plot_refinement(recording, curve, onset, output, config):
    """Saved two-segment gain of the cycle-residual energy inside the bracket."""
    row = onset.iloc[0]
    part = curve.loc[curve.eligible.astype(bool)].sort_values('sample_index')
    figure, axis = plt.subplots(figsize=(12, 4.5))
    try:
        axis.plot(part.elapsed, part.gain, lw=.9, label='Mean-shift gain (SSE reduction)')
        axis.axvline(float(row.coarse_elapsed), color=COARSE, ls='-.', lw=1, label='Coarse grid point')
        axis.axvline(float(row.elapsed), color=PRIMARY, ls='--', lw=1, label='Refined onset')
        axis.set(xlabel='Elapsed from recording start (s)', ylabel='Gain')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        figure.suptitle(f'{recording.name} | cycle-residual refinement | shift z = {row.shift_z:.1f}', fontsize=10)
        _save(figure, output, config)
    finally:
        plt.close(figure)
