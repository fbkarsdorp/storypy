import numpy as np


def _peak(scope, scores, peak):
    for n in scope:
        if scores[n] >= peak:
            peak = scores[n]
        else:
            break
    return peak


def _depth_scores(scores):
    clip = min(max(len(scores) / 10, 2), 5)
    depths = [0 for i in range(len(scores))]
    for i in range(clip, len(scores) - clip):
        lpeak = _peak(range(i, -1, -1), scores, scores[i])
        rpeak = _peak(range(i), scores, scores[i])
        depths[i] = (rpeak - scores[i]) + (lpeak - scores[i])
        #depths[i] = lpeak + rpeak - 2*scores[i]
    return depths


def _identify_boundaries(depths, policy='HC', adjacent_gaps=4):
    """Identifies boundaries at the peaks of similarity score
    differences."""
    boundaries = [0 for _ in depths]
    if not depths:
        return boundaries
    if policy == 'HC':
        cutoff = sum(depths) / len(depths) - np.std(depths) / 2.0
    else:
        cutoff = sum(depths) / len(depths) - np.std(depths)
    # sort the scores
    scores = sorted(range(len(depths)), key=lambda x: depths[x], reverse=True)
    hp = [score for score in scores if score > cutoff]
    for i in hp:
        boundaries[i] = 1
        for j in hp:
            if i != j and abs(i - j) < adjacent_gaps and boundaries[j] == 1:
                boundaries[i] = 0
    return boundaries


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[x[window_len - 1: 0: -1], x, x[-1: -window_len: -1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')  # mode='same'?
    return y[(window_len / 2 - 1): -(window_len / 2)]
