import numpy as np
from scipy.signal import correlate
def align(c, y, sr, max_ms=100):
    """Shift y so it lines up with reference c (lag by cross-correlation within +-max_ms). Returns (y, lag_samples); lag > 0
    means y was delayed. Engines with internal delay (C8, HVC) need this before STOI / SI-SDR / frame-wise comparisons."""
    n = min(len(c), len(y)); w = int(sr * max_ms / 1000)
    cc = correlate(y[:n], c[:n], 'full', method='fft'); m = len(cc) // 2
    lag = int(np.argmax(cc[m - w:m + w + 1]) - w)
    if lag > 0: y = np.concatenate([y[lag:], np.zeros(lag, np.float32)])
    elif lag < 0: y = np.concatenate([np.zeros(-lag, np.float32), y[:lag]])
    return y.astype(np.float32), lag
