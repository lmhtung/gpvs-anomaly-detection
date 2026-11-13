"""Reproduction check: the repo's E3b cross-mode CNN (up to 7 classes, raw and delta input), seeds 0-4, as in
gpvs-fault-detection/code/11_cnn.py (baseline = first 3 s of each run from t = 0, windows t >= 3 s, guard 0.5 s,
F4 excluded, CPU). Compared with the published gpvs-fault-detection/results/cnn_results_seeds.csv.

Run: python reproduce_e3b.py  ->  ../outputs/Hoque/reproduction_e3b.csv, reproduction_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from repo_cnn import SIGNALS, WIN, fit, predict_proba

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / 'data' / 'CSV_Files'
REPO = HERE.parent / 'gpvs-fault-detection'
OUT = HERE.parent / 'outputs/Hoque'
BASE_T, GUARD_T, SIX, SEEDS = 3.0, 0.5, [0, 1, 2, 5, 6, 7], [0, 1, 2, 3, 4]

def main():
    onsets = pd.read_csv(REPO / 'results/onsets.csv').set_index('file').onset_t
    published = pd.read_csv(REPO / 'results/cnn_results_seeds.csv')
    files = sorted(RAW.glob('F*.csv'))
    if not files:
        raise FileNotFoundError(f'No GPVS-Faults recordings found in {RAW}. See README.md.')

    waveforms, windows = [], []
    for path in files:
        data = pd.read_csv(path)
        size = len(data) // WIN * WIN
        waveforms.append(data[SIGNALS].to_numpy(np.float32)[:size].reshape(-1, WIN, len(SIGNALS)).transpose(0, 2, 1))
        windows.append(pd.DataFrame({
            'file': path.stem,
            'fault': int(path.stem[1]),
            'mode': 'MPPT' if path.stem[2] == 'M' else 'IPPT',
            'center': data.Time.to_numpy()[:size].reshape(-1, WIN)[:, WIN // 2],
        }))
    raw = np.concatenate(waveforms)
    metadata = pd.concat(windows, ignore_index=True)
    faults = metadata.fault.to_numpy()
    centers = metadata.center.to_numpy()
    onset = metadata.file.map(onsets).to_numpy(dtype=float, copy=True)
    onset[faults == 0] = np.inf
    labels = np.where(centers >= onset, faults, 0)
    near_onset = np.abs(centers - onset) < GUARD_T

    delta = raw.copy()
    for recording in metadata.file.unique():
        mask = (metadata.file == recording).to_numpy()
        delta[mask] -= raw[mask & (centers < BASE_T)].mean(axis=(0, 2), keepdims=True)

    rows = []
    keep = (centers >= BASE_T) & ~near_onset & (faults != 4)
    for name, features in [('raw', raw), ('delta', delta)]:
        for source, target in [('MPPT', 'IPPT'), ('IPPT', 'MPPT')]:
            train = keep & (metadata['mode'] == source).to_numpy()
            test = keep & (metadata['mode'] == target).to_numpy()
            for seed in SEEDS:
                model, mu, sd, classes = fit(features[train], labels[train], seed)
                prediction = classes[predict_proba(model, mu, sd, features[test]).argmax(1)]
                row = dict(scope=f'{source}->{target}', input=name, seed=seed,
                           acc=float(np.mean(labels[test] == prediction)),
                           macro_f1_six=float(f1_score(labels[test], prediction, labels=SIX,
                                                       average='macro', zero_division=0)))
                rows.append(row)
                print(row, flush=True)

    results = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / 'reproduction_e3b.csv', index=False)
    published = published[(published.exp == 'E3b_cross_mode') & published.seed.isin(SEEDS)]
    reproduced_scores = results.groupby(['scope', 'input']).macro_f1_six.mean().rename('reproduced')
    published_scores = published.groupby(['scope', 'input']).macro_f1_six.mean().rename('published')
    summary = reproduced_scores.to_frame().join(published_scores)
    summary['difference'] = summary.reproduced - summary.published
    summary['accepted (|diff| <= 0.05)'] = summary.difference.abs() <= .05
    summary.round(4).to_csv(OUT / 'reproduction_summary.csv')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
