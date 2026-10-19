"""Time-frequency gate for clean speech targets (open, deterministic): cells far below the clip's speech level are pushed down,
so the model is not taught to keep the recording's own noise floor between harmonics. STFT = the model's (48 kHz, 960/480, Vorbis).
  gate(x, depth_db=50, atten_db=40, width_db=6): threshold = (95th pct of per-frame peak dB) - depth_db;
  cells below threshold - width get -atten_db, cells above threshold are untouched, linear-in-dB ramp between.
Self-check: python tf_gate.py  (perfect reconstruction with gate off, and removed-energy ratio on a test clip)."""
import numpy as np
N, H = 960, 480
_n = np.arange(N); W = np.sin(np.pi / 2 * np.sin(np.pi * (_n + 0.5) / N) ** 2).astype(np.float32)


def stft(x):
    xp = np.concatenate([np.zeros(H, np.float32), x, np.zeros(N, np.float32)]); T = (len(xp) - N) // H + 1
    return np.fft.rfft(np.stack([xp[i * H:i * H + N] for i in range(T)]) * W, N)


def istft(X, n):
    fr = np.fft.irfft(X, N) * W; out = np.zeros(H * (len(fr) + 1), np.float64)
    for t in range(len(fr)):
        out[t * H:t * H + N] += fr[t]
    return out[H:H + n].astype(np.float32)


def gate(x, depth_db=45.0, rel_db=35.0, atten_db=40.0, width_db=6.0):
    """v2: threshold = min(clip level - depth_db, frame peak - rel_db); the frame-relative term protects soft passages (a -25 dB segment loses nothing).
    rel_db=None gives the old clip-level gate. Calibrated 2026-10-13: v2 45/35 gives the same floor as the clip-level gate with 0 dB change on soft segments."""
    X = stft(x); L = 20 * np.log10(np.abs(X) + 1e-12)
    peak = L.max(1); level = np.percentile(peak, 95); thr = np.full_like(peak, level - depth_db)
    if rel_db is not None: thr = np.minimum(thr, peak - rel_db)
    g_db = -atten_db * np.clip((thr[:, None] - L) / width_db, 0.0, 1.0)
    return istft(X * 10 ** (g_db / 20), len(x))


if __name__ == '__main__':
    rng = np.random.default_rng(0); x = (rng.standard_normal(48000 * 2) * 0.1).astype(np.float32)
    assert np.abs(istft(stft(x), len(x)) - x).max() < 1e-6, 'reconstruction'
    y = gate(x, atten_db=0.0); assert np.abs(y - x).max() < 1e-6, 'gate off'
    import soundfile as sf, librosa, sys
    p = sys.argv[1] if len(sys.argv) > 1 else r'D:\OpenNCData\dfn3\floortest\00-clean.wav'
    a, sr = sf.read(p, dtype='float32'); a = librosa.resample(a, orig_sr=sr, target_sr=48000)
    for d in (40, 45, 50, 55):
        y = gate(a, depth_db=d); print(f'depth {d} dB: removed energy {10 * np.log10(((a - y) ** 2).sum() / (a ** 2).sum()):.1f} dB rel')
    print('self-check ok')
