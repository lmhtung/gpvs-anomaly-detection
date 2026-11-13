"""Score every target-side piece in ../outputs/scenarios.csv with every trained seed.

For each piece the cohort is every 200-sample window with centre >= S + 0.02 s (S = 0.5 s after the crop start).
Windows E1 cannot decide yet (fewer than 32 descriptors) are written with decided = False and p_fault = NaN.
Run: python scenarios.py   ->  <output_dir>/scenarios/<direction>/seed_<k>.csv.gz
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import features as F
from model import BinaryCNN1D, predict_proba
from train import load_config, other

COHORT_START = 0.52


def main():
    cfg = load_config()
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    pieces = pd.read_csv(cfg['scenario_file'])
    for source in ('L', 'M'):
        direction = f'{source}_to_{other(source)}'
        paths = sorted((Path(cfg['output_dir']) / direction).glob('seed_*/checkpoint.pt'))
        if not paths:
            continue
        bundles = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
        bank, scaler = bundles[0]['bank'], bundles[0]['scaler']  # source-only and identical across seeds
        models = []
        for b in bundles:
            assert b['bank'] == bank and b['scaler'] == scaler
            m = BinaryCNN1D(b['in_channels'], cfg['model']['hidden_features'], cfg['model']['dropout'])
            m.load_state_dict(b['model_state'])
            models.append((b['seed'], m.to(device)))
        cohort_cfg = dict(deepcopy(cfg), start_seconds=COHORT_START)
        frames = {seed: [] for seed, _ in models}
        recs = {}
        for row in pieces[pieces['mode'] == other(source)].itertuples():
            if row.recording not in recs:
                recs[row.recording] = F.read_recording(Path(cfg['data_dir']) / f'{row.recording}.csv', cfg)
            piece = F.crop(recs[row.recording], row.crop_start)
            onset = None if np.isnan(row.onset_in_crop) else row.onset_in_crop
            cohort = F.window_metadata(piece, cohort_cfg, onset)
            base, lag, meta = F.raw_sequences(piece, bank, cfg)  # label-free
            x = F.assemble(base, lag, scaler, cfg)
            for seed, model in models:
                p = meta[['right_sample_index']].assign(p_fault=predict_proba(model, x, device))
                f = cohort.merge(p, on='right_sample_index', how='left')
                frames[seed].append(f.assign(piece=row.piece, scenario=row.scenario, decided=f.p_fault.notna(),
                                             time_since_start=f.center_time - cfg['warmup_seconds'],
                                             onset_since_start=np.nan if onset is None else onset - cfg['warmup_seconds']))
        out = Path(cfg['output_dir']) / 'scenarios' / direction
        out.mkdir(parents=True, exist_ok=True)
        for seed, parts in frames.items():
            pd.concat(parts, ignore_index=True).to_csv(out / f'seed_{seed}.csv.gz', index=False)
        print(f'{direction}: {len(frames)} seeds, {sum(len(p) for p in parts)} windows per seed', flush=True)


if __name__ == '__main__':
    main()
