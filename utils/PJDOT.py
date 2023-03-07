import numpy as np
from collections import Counter

def match_information(gamma, label_s, label_t):
    threshold = gamma.max() / 2
    loc_x, loc_y = np.where(gamma > threshold)
    match_count = 0
    for x, y, in zip(loc_x, loc_y):
        if label_s[x] == label_t[y]: match_count += 1
    freq_s = Counter(np.sort(label_s))
    freq_t = Counter(np.sort(label_t))
    total_count = 0
    for label in freq_s:
        if label in freq_t:
            total_count += min(freq_s[label], freq_t[label])
        else:
            pass
    return match_count / len(loc_x) * 100, total_count / len(label_s) * 100, match_count / total_count * 100

