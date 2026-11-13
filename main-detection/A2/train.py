"""Train A2 (binary normal/fault) cross-mode: fit on one mode, test on the original recordings of the other.

Run: python train.py [--directions L M] [--seeds 0 1 2 3 4]
Writes <output_dir>/<L_to_M|M_to_L>/seed_<k>/{checkpoint.pt, history.csv, predictions.csv, metrics.json}
and <output_dir>/summary.csv. Target data never touch fitting, scaling or checkpoint choice (fixed last epoch).
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

import features as F
from model import BinaryCNN1D, feature_dropout, predict_proba

HERE = Path(__file__).resolve().parent


def load_config(path=HERE / 'config.yaml'):
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ('data_dir', 'onset_file', 'scenario_file', 'output_dir'):
        cfg[key] = str((Path(path).parent / cfg[key]).resolve())
    return cfg


def other(mode):
    return 'M' if mode == 'L' else 'L'


def build_inputs(pieces, bank, cfg):
    """Unscaled (base, lag, absolute, step) and metadata for (recording, onset) pieces; near-onset windows dropped."""
    parts = [F.raw_sequences(r, bank, cfg, onset) for r, onset in pieces]
    arrays = [np.concatenate([p[k] for p in parts]) for k in range(4)]
    meta = pd.concat([p[4] for p in parts], ignore_index=True)
    keep = ~meta.near_onset.to_numpy()
    return [a[keep] for a in arrays], meta.loc[keep].reset_index(drop=True)


def source_pieces(rec, onset, cfg):
    """The full source recording plus its training crops [s, s + crop_seconds), s = 1, 2, ..."""
    out = [(rec, onset)]
    length, step = cfg['augment']['crop_seconds'], cfg['augment']['crop_step_seconds']
    for start in np.arange(1.0, rec.times[-1] - length + 1e-9, step):
        piece = F.crop(rec, start, length)
        offset = float(rec.times[rec.times >= start][0])
        out.append((piece, None if onset is None else onset - offset))
    return out


def prepare_direction(cfg, source):
    folder = Path(cfg['data_dir'])
    onset_table = pd.read_csv(cfg['onset_file']).set_index('file').onset_t
    names = {m: [f'F{f}{m}' for f in cfg['recordings']] for m in (source, other(source))}
    onsets = {n: float(onset_table[n]) for m in names for n in names[m] if not n.startswith('F0')}
    recs = {n: F.read_recording(folder / f'{n}.csv', cfg) for m in names for n in names[m]}
    bank = F.fit_bank(recs[f'F0{source}'], cfg)  # source F0 only
    src = [p for n in names[source] for p in source_pieces(recs[n], onsets.get(n), cfg)]
    s_arrays, sm = build_inputs(src, bank, cfg)
    scaler = F.fit_scaler(*s_arrays)  # source train (incl. its crops) only
    t_arrays, tm = build_inputs([(recs[n], onsets.get(n)) for n in names[other(source)]], bank, cfg)
    return F.assemble(*s_arrays, scaler, cfg), sm, F.assemble(*t_arrays, scaler, cfg), tm, bank, scaler


def binary_metrics(y, p, prob):
    from sklearn.metrics import roc_auc_score
    tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    f1n = 2 * tn / max(2 * tn + fp + fn, 1)
    return dict(f1=f1, macro_f1=(f1 + f1n) / 2, accuracy=(tp + tn) / len(y), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), false_alarm_rate=fp / max(fp + tn, 1),
                auroc=float(roc_auc_score(y, prob)) if 0 < y.sum() < len(y) else float('nan'),
                tp=tp, fp=fp, fn=fn, tn=tn)


def train_seed(cfg, data, direction, seed, device):
    out = Path(cfg['output_dir']) / direction / f'seed_{seed}'
    if (out / 'checkpoint.pt').exists():
        print(f'{direction} seed={seed}: exists, skipped')
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    xs, ms, xt, mt, bank, scaler = data
    y = (ms.label.to_numpy() > 0).astype(np.int64)
    counts = np.bincount(y, minlength=2)
    weights = torch.tensor(len(y) / (2 * counts), dtype=torch.float32, device=device)
    x, yt = torch.as_tensor(xs, device=device), torch.as_tensor(y, device=device)
    tr = cfg['train']
    model = BinaryCNN1D(x.shape[1], cfg['model']['hidden_features'], cfg['model']['dropout']).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr['lr'], weight_decay=tr['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tr['epochs'], eta_min=tr['lr'] * .05)
    history, t0 = [], time.time()
    for epoch in range(1, tr['epochs'] + 1):
        model.train()
        order, total = torch.randperm(len(x), device=device), 0.
        for lo in range(0, len(x), tr['batch_size']):
            idx = order[lo:lo + tr['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(feature_dropout(x[idx], tr['feature_dropout'])), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tr['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        history.append(dict(epoch=epoch, loss=total / len(x), lr=optimizer.param_groups[0]['lr']))
        scheduler.step()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    torch.save(dict(config=cfg, bank=bank, scaler=scaler, in_channels=int(x.shape[1]), direction=direction,
                    seed=seed, model_state={k: v.cpu() for k, v in model.state_dict().items()}), out / 'checkpoint.pt')
    prob = predict_proba(model, xt, device)
    pred = mt.assign(p_fault=prob, predicted=(prob >= .5).astype(int), true=(mt.label > 0).astype(int))
    pred.to_csv(out / 'predictions.csv', index=False)
    metrics = binary_metrics(pred.true.to_numpy(), pred.predicted.to_numpy(), prob)
    (out / 'metrics.json').write_text(json.dumps(dict(direction=direction, seed=seed, n_train=len(x),
                                                      n_test=len(xt), **metrics), indent=1))
    print(f'{direction} seed={seed} F1={metrics["f1"]:.4f} recall={metrics["recall"]:.4f} '
          f'false_alarm={metrics["false_alarm_rate"]:.4f} ({time.time() - t0:.0f}s)', flush=True)


def summarize(cfg):
    rows = [json.loads(p.read_text()) for p in sorted(Path(cfg['output_dir']).glob('*_to_*/seed_*/metrics.json'))]
    frame = pd.DataFrame(rows)
    summary = frame.groupby('direction')[['f1', 'macro_f1', 'recall', 'false_alarm_rate', 'auroc']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.round(4).to_csv(Path(cfg['output_dir']) / 'summary.csv')
    print(summary.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directions', nargs='+', default=['L', 'M'], choices=['L', 'M'], help='source modes')
    ap.add_argument('--seeds', nargs='+', type=int)
    args = ap.parse_args()
    cfg = load_config()
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    for source in args.directions:
        direction = f'{source}_to_{other(source)}'
        t0 = time.time()
        data = prepare_direction(cfg, source)
        print(f'{direction}: train {data[0].shape} test {data[2].shape} ({time.time() - t0:.0f}s, {device})', flush=True)
        for seed in args.seeds or cfg['seeds']:
            train_seed(cfg, data, direction, seed, device)
    summarize(cfg)


if __name__ == '__main__':
    main()
