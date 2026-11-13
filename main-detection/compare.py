"""Compare E1, A2 and Repo-Delta on the shared scenarios.

Event level (primary): a detection is the first run of K = 3 consecutive fault predictions that starts at or after
max(onset, S); delay = that window's centre - max(onset, S). A false alarm is such a run that ends before the onset
(any run in a healthy piece). Window level (secondary): windows within 0.5 s of the onset are dropped.
Windows a model cannot decide yet count as normal.
Run: python compare.py  ->  outputs/comparison/
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = {'E1': HERE / 'outputs/E1/scenarios', 'A2': HERE / 'outputs/A2/scenarios',
          'Repo-Delta': HERE / 'outputs/Hoque/binary_delta/scenarios'}
K = 3
EARLY = ['onset_after_0.5', 'onset_after_1', 'onset_after_1.5', 'onset_after_2']


def load():
    frames = []
    for model, root in MODELS.items():
        paths = sorted(root.glob('*/seed_*.csv.gz'))
        if not paths:
            raise FileNotFoundError(f'{model}: no scenario predictions in {root}. See main-detection/README.md.')
        for path in paths:
            f = pd.read_csv(path, usecols=['piece', 'scenario', 'recording', 'right_sample_index', 'time_since_start',
                                           'onset_since_start', 'label', 'near_onset', 'decided', 'p_fault'])
            frames.append(f.assign(model=model, direction=path.parent.name, seed=int(path.stem.split('_')[1].split('.')[0])))
    w = pd.concat(frames, ignore_index=True)
    w['pred'] = (w.decided & (w.p_fault >= .5)).astype(int)
    runs = w[['model', 'direction', 'seed']].drop_duplicates()
    coverage = runs.groupby(['direction', 'seed']).model.nunique()
    if not coverage.eq(len(MODELS)).all():
        raise ValueError('Scenario predictions must cover the same directions and seeds for all models.')
    # Every model must be scored on the same windows with the same labels.
    keys = ['direction', 'piece', 'right_sample_index']
    ref = None
    for (model, seed), g in w.groupby(['model', 'seed']):
        cohort = g[keys + ['label']].sort_values(keys).reset_index(drop=True)
        if ref is None:
            ref = cohort
        pd.testing.assert_frame_equal(cohort, ref, check_dtype=False)
    return w.sort_values(['model', 'direction', 'seed', 'piece', 'right_sample_index']).reset_index(drop=True)


def first_run(pred, ok):
    """Index of the first window starting K consecutive predictions among `ok` windows, or None."""
    run = 0
    for i in range(len(pred)):
        run = run + 1 if (pred[i] and ok[i]) else 0
        if run == K:
            return i - K + 1
    return None


def events(w):
    rows = []
    for (model, direction, seed, piece), g in w.groupby(['model', 'direction', 'seed', 'piece'], sort=False):
        t, pred = g.time_since_start.to_numpy(), g.pred.to_numpy()
        o = g.onset_since_start.iloc[0]
        ref = 0.0 if np.isnan(o) else max(o, 0.0)
        healthy = np.isnan(o)
        before = np.ones(len(t), bool) if healthy else t < ref
        fa = first_run(pred, before)
        false_alarm = fa is not None  # runs are restricted to windows before the onset
        row = dict(model=model, direction=direction, seed=seed, piece=piece, scenario=g.scenario.iloc[0],
                   recording=g.recording.iloc[0], false_alarm=bool(false_alarm), has_before=bool(before.any()))
        if not healthy:
            d = first_run(pred, t >= ref)
            row.update(detected=d is not None, delay=t[d] - ref if d is not None else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def window_metrics(w):
    v = w[~w.near_onset]
    rows = []
    for (model, direction, seed, scenario), g in v.groupby(['model', 'direction', 'seed', 'scenario']):
        y, p = g.label.to_numpy(), g.pred.to_numpy()
        tp, fp, fn = ((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum()
        rows.append(dict(model=model, direction=direction, seed=seed, scenario=scenario,
                         f1=2 * tp / max(2 * tp + fp + fn, 1) if y.any() else np.nan,
                         recall=p[y == 1].mean() if y.any() else np.nan,
                         false_alarm=p[y == 0].mean() if (y == 0).any() else np.nan))
    return pd.DataFrame(rows)


def main():
    w = load()
    ev = events(w)
    win = window_metrics(w)
    out = HERE / 'outputs/comparison'
    out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out / 'events.csv', index=False)
    win.to_csv(out / 'windows_per_seed.csv', index=False)

    faults = ev[ev.scenario != 'healthy_start']
    det = faults.groupby(['direction', 'scenario', 'model']).agg(
        detected=('detected', 'mean'), median_delay_s=('delay', 'median'), mean_delay_s=('delay', 'mean'))
    fa = ev[ev.has_before].groupby(['direction', 'scenario', 'model']).false_alarm.mean().rename('false_alarm_events')
    event_summary = det.join(fa, how='outer').round(3)
    event_summary.to_csv(out / 'event_summary.csv')
    window_summary = win.groupby(['direction', 'scenario', 'model'])[['f1', 'recall', 'false_alarm']].mean().round(3)
    window_summary.to_csv(out / 'window_summary.csv')

    hyp = []
    for direction in sorted(ev.direction.unique()):
        early = faults[(faults.direction == direction) & faults.scenario.isin(EARLY)]
        s = early.groupby('model').agg(miss=('detected', lambda x: 1 - x.mean()), median_delay=('delay', 'median'))
        a, r = s.loc['A2'], s.loc['Repo-Delta']
        hyp.append(dict(direction=direction, hypothesis='H1 early onset (X<=2 s)',
                        A2=f'miss {a.miss:.3f}, median delay {a.median_delay:.3f} s',
                        Repo=f'miss {r.miss:.3f}, median delay {r.median_delay:.3f} s',
                        E1=f'miss {s.loc["E1"].miss:.3f}, median delay {s.loc["E1"].median_delay:.3f} s',
                        supported=bool(a.median_delay < r.median_delay and a.miss <= r.miss)))
        fs = win[(win.direction == direction) & (win.scenario == 'faulty_start')].groupby('model').recall.mean()
        hyp.append(dict(direction=direction, hypothesis='H2 faulty start (window recall)',
                        A2=f'{fs["A2"]:.3f}', Repo=f'{fs["Repo-Delta"]:.3f}', E1=f'{fs["E1"]:.3f}',
                        supported=bool(fs['Repo-Delta'] <= .5 and fs['A2'] > fs['Repo-Delta'])))
    hyp = pd.DataFrame(hyp)
    hyp.to_csv(out / 'hypotheses.csv', index=False)
    pd.set_option('display.width', 250)
    print('cohort check: all models scored on identical windows and labels\n')
    print(event_summary.to_string(), '\n')
    print(window_summary.to_string(), '\n')
    print(hyp.to_string(index=False))


if __name__ == '__main__':
    main()
