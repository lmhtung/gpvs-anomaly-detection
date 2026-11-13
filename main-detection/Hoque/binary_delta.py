"""Repo-Delta, binary: the repo's E3b delta CNN with a normal/fault output, under our deployment convention.

The system starts at S = 0.5 s after the crop start (warm-up cut). Like the repo, every window is the raw 13 x 200
waveform minus the per-channel mean over the run's first 3 s, here [S, S+3); a decision exists only for windows with
centre >= S+3 (the repo uses [0, 3) and t >= 3). Same CNN and training as the repo (repo_cnn.py), 2 classes.
Train: the full SOURCE recordings, windows with centre >= S+3, |centre - onset| >= 0.5 s, F4 excluded.

Run: python binary_delta.py
Writes ../outputs/Hoque/binary_delta/<direction>/seed_<k>/{predictions.csv, metrics.json}, summary.csv and
../outputs/Hoque/binary_delta/scenarios/<direction>/seed_<k>.csv.gz (same cohort and columns as E1/A2).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repo_cnn import SIGNALS, WIN, fit, predict_proba

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / 'data' / 'CSV_Files'
ONSETS = HERE.parent / 'gpvs-fault-detection/results/onsets.csv'
SCENARIOS = HERE.parent / 'outputs' / 'scenarios.csv'
OUT = HERE.parent / 'outputs/Hoque/binary_delta'
WARMUP, BASE_LEN, GUARD, COHORT_START = 0.5, 3.0, 0.5, 0.52
RECORDINGS, SEEDS = [0, 1, 2, 3, 5, 6, 7], [0, 1, 2, 3, 4]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def read(name):
    d = pd.read_csv(RAW / f'{name}.csv')
    return d.Time.to_numpy(), d[SIGNALS].to_numpy(np.float32)


def windows(times, values, crop_start=0.0, onset=None):
    """Windows of 200 samples from the crop start (stride 200), centre >= COHORT_START s after it, delta input."""
    keep = times >= crop_start
    t, v = times[keep] - times[keep][0], values[keep]
    starts = np.arange(0, len(t) - WIN + 1, WIN)
    center = t[starts + WIN // 2]
    sel = center >= COHORT_START
    starts, center = starts[sel], center[sel]
    x = v[starts[:, None] + np.arange(WIN)].transpose(0, 2, 1)       # [n, 13, 200]
    base_rows = (center >= WARMUP) & (center < WARMUP + BASE_LEN)    # the run's first 3 s after S
    x = x - x[base_rows].mean(axis=(0, 2), keepdims=True)
    label = np.zeros(len(center), dtype=int) if onset is None else (center >= onset).astype(int)
    near = np.zeros(len(center), dtype=bool) if onset is None else np.abs(center - onset) < GUARD
    meta = pd.DataFrame(dict(right_sample_index=starts + WIN - 1, center_time=center, label=label, near_onset=near,
                             decided=center >= WARMUP + BASE_LEN))
    return x.astype(np.float32), meta


def metrics(y, p, prob):
    from sklearn.metrics import roc_auc_score
    tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    f1, f1n = 2 * tp / max(2 * tp + fp + fn, 1), 2 * tn / max(2 * tn + fp + fn, 1)
    return dict(f1=f1, macro_f1=(f1 + f1n) / 2, recall=tp / max(tp + fn, 1), false_alarm_rate=fp / max(fp + tn, 1),
                auroc=float(roc_auc_score(y, prob)), tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    onset = pd.read_csv(ONSETS).set_index('file').onset_t
    pieces = pd.read_csv(SCENARIOS)
    data = {}
    for mode in 'LM':
        xs, ms = [], []
        for f in RECORDINGS:
            name = f'F{f}{mode}'
            t, v = read(name)
            x, meta = windows(t, v, onset=None if f == 0 else float(onset[name]))
            keep = meta.decided.to_numpy() & ~meta.near_onset.to_numpy()
            xs.append(x[keep]); ms.append(meta.loc[keep].assign(recording=name))
        data[mode] = (np.concatenate(xs), pd.concat(ms, ignore_index=True))
    rows = []
    for source, target in (('L', 'M'), ('M', 'L')):
        direction = f'{source}_to_{target}'
        xs, ms = data[source]
        xt, mt = data[target]
        models = []
        for seed in SEEDS:
            model, mu, sd, classes = fit(xs, ms.label.to_numpy(), seed, DEVICE)
            assert list(classes) == [0, 1]
            prob = predict_proba(model, mu, sd, xt, DEVICE)[:, 1]
            out = OUT / direction / f'seed_{seed}'
            out.mkdir(parents=True, exist_ok=True)
            mt.assign(p_fault=prob, predicted=(prob >= .5).astype(int)).to_csv(out / 'predictions.csv', index=False)
            r = dict(direction=direction, seed=seed, n_train=len(xs), n_test=len(xt),
                     **metrics(mt.label.to_numpy(), (prob >= .5).astype(int), prob))
            (out / 'metrics.json').write_text(json.dumps(r, indent=1))
            rows.append(r)
            models.append((seed, model, mu, sd))
            print(f'{direction} seed={seed} F1={r["f1"]:.4f} recall={r["recall"]:.4f} false_alarm={r["false_alarm_rate"]:.4f}', flush=True)
        # scenarios on the target mode
        frames = {seed: [] for seed, *_ in models}
        cache = {}
        for row in pieces[pieces['mode'] == target].itertuples():
            if row.recording not in cache:
                cache[row.recording] = read(row.recording)
            o = None if np.isnan(row.onset_in_crop) else row.onset_in_crop
            x, meta = windows(*cache[row.recording], crop_start=row.crop_start, onset=o)
            for seed, model, mu, sd in models:
                prob = predict_proba(model, mu, sd, x, DEVICE)[:, 1]
                frames[seed].append(meta.assign(
                    recording=row.recording, piece=row.piece, scenario=row.scenario,
                    p_fault=np.where(meta.decided, prob, np.nan),
                    time_since_start=meta.center_time - WARMUP, onset_since_start=np.nan if o is None else o - WARMUP))
        sc = OUT / 'scenarios' / direction
        sc.mkdir(parents=True, exist_ok=True)
        for seed, parts in frames.items():
            pd.concat(parts, ignore_index=True).to_csv(sc / f'seed_{seed}.csv.gz', index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby('direction')[['f1', 'macro_f1', 'recall', 'false_alarm_rate', 'auroc']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.round(4).to_csv(OUT / 'summary.csv')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
