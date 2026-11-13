"""Label-free choice of one confirmed onset per recording.

Selection order:
1. the earliest strong short-context Energy event;
2. otherwise the long-context Gaussian peak, if its ratio >= max_local_ratio;
3. otherwise the joint-MMD peak, if confirmed on >= 2 contexts (joint.py);
4. otherwise no onset. Sub-threshold evidence is never exported as an onset.
"""
import numpy as np


def options(config):
    return config['onset']


def short_config(config):
    """Config view whose contexts are the short Energy contexts."""
    opts = options(config)
    return {**config, 'contexts_seconds': list(opts['short_contexts_seconds']),
            'min_scales': opts['short_min_scales']}


def maximum(table, prefix, config, early_edge=True):
    """(selected row, peak row) of the highest above-threshold score, or None.

    Ties on a contiguous plateau take the row nearest its middle. With
    ``early_edge`` the selected row moves back while the score stays >=
    ``early_fraction`` × peak, for at most ``max_early_seconds``: a two-sided
    profile of a gradual drift can peak after the transition starts.
    """
    valid = table[f'{prefix}valid'].to_numpy(dtype=bool, copy=True)
    valid &= table[f'{prefix}ratio'].to_numpy(float) > 1.
    positions = np.flatnonzero(valid)
    if not len(positions):
        return None
    scores = table[f'{prefix}score'].to_numpy(float)
    first = int(np.argmax(scores[positions]))
    if not scores[positions[first]] > 0:
        return None
    segments = table.segment_id.to_numpy()
    end = first + 1
    while (end < len(positions) and positions[end] == positions[end - 1] + 1
           and segments[positions[end]] == segments[positions[first]]
           and scores[positions[end]] == scores[positions[first]]):
        end += 1
    plateau = positions[first:end]
    times = table.split_elapsed.to_numpy(float)
    midpoint = (times[plateau[0]] + times[plateau[-1]]) * .5
    peak = int(plateau[np.argmin(np.abs(times[plateau] - midpoint))])
    selected = peak
    opts = options(config)
    if early_edge:
        while (selected > 0 and valid[selected - 1] and segments[selected - 1] == segments[peak]
               and times[selected - 1] >= times[peak] - opts['max_early_seconds']
               and scores[selected - 1] >= opts['early_fraction'] * scores[peak]):
            selected -= 1
    return selected, peak


def separate(peaks, heights, prominences, distance):
    """Keep higher peaks first; drop peaks closer than `distance` to a kept one.

    Apply separation after prominence filtering so a discarded spike cannot
    suppress a valid neighbour.
    """
    kept = []
    for i in np.argsort(-heights, kind='stable'):
        if all(abs(int(peaks[i]) - int(peaks[j])) >= distance for j in kept):
            kept.append(i)
    kept = np.sort(np.asarray(kept, dtype=int))
    return peaks[kept], prominences[kept]


def strong_energy_events(table, recording, config):
    """Candidates (first confirmed row, peak row) of strong short Energy events.

    A row is confirmed when one channel has ratio >= strong_peak_ratio and its
    two shortest contexts each have ratio >= confirmation_ratio. An event is a
    prominent peak >= strong_peak_ratio; its start is the first confirmed row of
    the contiguous strong run before the peak (at most max(short contexts)).
    """
    from scipy.signal import find_peaks
    opts = options(config)
    times = table.split_elapsed.to_numpy(float)
    segments = table.segment_id.to_numpy(int)
    valid = table.short_valid.to_numpy(bool)
    ratios = table.short_ratio.to_numpy(float)
    contexts = sorted(opts['short_contexts_seconds'])
    confirmed = np.zeros(len(table), dtype=bool)
    for channel in config['channels']:
        support = (table[f'short_score_{channel}'].to_numpy() / table.short_threshold.to_numpy()
                   >= opts['strong_peak_ratio'])
        for context in contexts[:2]:
            support &= table[f'short_{channel}__ratio__{context:g}s'].to_numpy() >= opts['confirmation_ratio']
            support &= table[f'short_valid__{context:g}s'].to_numpy(bool)
        confirmed |= support
    candidates = []
    for segment in np.unique(segments):
        positions = np.flatnonzero(valid & (segments == segment))
        for block in np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1):
            if not len(block):
                continue
            step = float(table.hop_samples.iloc[block[0]]) * recording.dt
            distance = max(1, int(round(opts['min_peak_separation_seconds'] / step)))
            # Zero padding lets a peak touching the edge of valid support count.
            peaks, props = find_peaks(np.r_[0., ratios[block], 0.], height=opts['strong_peak_ratio'],
                                      prominence=opts['min_prominence_ratio'])
            keep = props['prominences'] >= opts['prominence_fraction'] * props['peak_heights']
            peaks, _ = separate(peaks[keep], props['peak_heights'][keep], props['prominences'][keep], distance)
            for offset in peaks - 1:
                if not 0 <= offset < len(block):
                    continue
                peak = int(block[offset])
                start = peak
                while (start > block[0] and ratios[start - 1] >= opts['strong_peak_ratio']
                       and times[start - 1] >= times[peak] - max(contexts)):
                    start -= 1
                possible = np.flatnonzero(confirmed[start:peak + 1] & valid[start:peak + 1]) + start
                if len(possible):
                    candidates.append((int(possible[0]), peak))
    return candidates


def select_local(table, recording, config):
    """(branch, selected row, peak row, strong-event count) from local evidence, or None."""
    candidates = strong_energy_events(table, recording, config)
    if candidates:
        times = table.split_elapsed.to_numpy(float)
        selected, peak = min(candidates, key=lambda item: (times[item[0]], times[item[1]]))
        return 'short_energy', selected, peak, len(candidates)
    chosen = maximum(table, '', config)
    return None if chosen is None else ('long_gaussian', *chosen, 0)


def select(table, recording, config):
    from . import joint
    return joint.select(table, config, select_local(table, recording, config))
