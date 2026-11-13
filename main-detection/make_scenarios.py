"""Generate the shared target-side test pieces in outputs/scenarios.csv.

crop_start is in original file time; the system starts S = crop_start + 0.5 s (warm-up cut);
onset_in_crop is the fault onset in crop time (NaN for F0). A piece needs >= 1 s of data after S.
Run: python make_scenarios.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data' / 'CSV_Files'
ONSETS = HERE / 'gpvs-fault-detection/results/onsets.csv'
OUTPUT = HERE / 'outputs' / 'scenarios.csv'
WARMUP = 0.5
RECORDINGS = [f'F{f}{m}' for m in 'LM' for f in (0, 1, 2, 3, 5, 6, 7)]  # F4 has no onset

def main():
    onset = pd.read_csv(ONSETS).set_index('file').onset_t
    rows = []
    for name in RECORDINGS:
        end = pd.read_csv(DATA / f'{name}.csv', usecols=['Time']).Time.iloc[-1]
        healthy = name.startswith('F0')
        if healthy:
            pieces = [('original', 0.0), *[('healthy_start', s - WARMUP) for s in (2.0, 5.0, 8.0)]]
        else:
            t = float(onset[name])
            pieces = [('original', 0.0), ('faulty_start', t + 1.0 - WARMUP)]
            pieces.extend((f'onset_after_{x:g}', t - x - WARMUP) for x in (0.5, 1.0, 1.5, 2.0, 3.0))
        for scenario, start in pieces:
            if end - (start + WARMUP) < 1.0:
                continue
            rows.append(dict(scenario=scenario, recording=name, mode=name[2], crop_start=round(start, 6),
                             onset_in_crop=np.nan if healthy else round(float(onset[name]) - start, 6)))
    table = pd.DataFrame(rows)
    table.insert(0, 'piece', range(len(table)))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(table.groupby(['mode', 'scenario']).size().unstack(0).to_string())


if __name__ == '__main__':
    main()
