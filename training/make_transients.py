"""Transient-heavy noise set: (a) DNS noise clips selected by burstiness (peak frame far above the median frame),
(b) synthetic claps / key clicks / knocks. Writes noise_transient_train.hdf5 and noise_transient_valid.hdf5 in the DFN3 noise layout.
Usage: box_transients.py <noise_src.hdf5> <out_dir> [n_select=4000] [workers=16]"""
import sys, os, h5py, numpy as np
from multiprocessing import Pool
src, out = sys.argv[1:3]; NSEL = int(sys.argv[3]) if len(sys.argv) > 3 else 4000; NW = int(sys.argv[4]) if len(sys.argv) > 4 else 16
SR, FR = 48000, 960


def score(keys):
    """burstiness = peak 20 ms frame dB - median frame dB; also require the loud frames to be rare (<15% of frames within 20 dB of the peak)."""
    res = []
    with h5py.File(src, 'r') as h:
        for k in keys:
            x = h['noise'][k][()][0].astype(np.float32) / 32768; n = len(x) // FR * FR
            if n < SR: continue
            e = 10 * np.log10((x[:n].reshape(-1, FR) ** 2).mean(1) + 1e-12); pk = e.max()
            res.append((k, float(pk - np.median(e)), float((e > pk - 20).mean())))
    return res


def synth(rng, kind, T=10):
    t = np.arange(SR * T) / SR; y = np.zeros(SR * T, np.float32)
    if kind == 'clap':
        for st in np.cumsum(rng.uniform(0.3, 2.0, 40)):
            if st > T - 0.1: break
            d = rng.uniform(0.004, 0.03); i0 = int(st * SR); n = int(d * SR) * 4
            burst = rng.standard_normal(n) * np.exp(-np.arange(n) / (d * SR)); y[i0:i0 + n] += burst[:len(y) - i0] * rng.uniform(0.3, 1)
    elif kind == 'click':
        for st in np.cumsum(rng.uniform(0.06, 0.25, 120)):
            if st > T - 0.05: break
            n = int(rng.uniform(0.001, 0.006) * SR) * 3; i0 = int(st * SR)
            burst = rng.standard_normal(n) * np.exp(-np.arange(n) / (n / 3)); y[i0:i0 + n] += burst[:len(y) - i0] * rng.uniform(0.2, 1)
    else:  # knock: damped low-frequency sinusoid with a noise onset
        for st in np.cumsum(rng.uniform(0.4, 2.5, 30)):
            if st > T - 0.2: break
            f0 = rng.uniform(80, 250); n = int(rng.uniform(0.05, 0.15) * SR); i0 = int(st * SR); tt = np.arange(n) / SR
            burst = (np.sin(2 * np.pi * f0 * tt) + 0.3 * rng.standard_normal(n) * np.exp(-tt * 400)) * np.exp(-tt * 30); y[i0:i0 + n] += burst[:len(y) - i0] * rng.uniform(0.3, 1)
    if kind in ('clap', 'click'):  # random band colouring
        X = np.fft.rfft(y); f = np.fft.rfftfreq(len(y), 1 / SR); lo, hi = rng.uniform(300, 1500), rng.uniform(4000, 16000)
        X *= 1 / (1 + (lo / np.maximum(f, 1)) ** 4) / (1 + (f / hi) ** 4); y = np.fft.irfft(X, len(y)).astype(np.float32)
    return (y / (np.abs(y).max() + 1e-9) * 0.5 * 32767).astype(np.int16)


if __name__ == '__main__':
    with h5py.File(src, 'r') as h:
        keys = list(h['noise'].keys()); fattrs = dict(h.attrs)
    chunks = [keys[i:i + 200] for i in range(0, len(keys), 200)]; sc = []
    with Pool(NW) as pool:
        for i, r in enumerate(pool.imap_unordered(score, chunks)):
            sc += r
            if i % 50 == 0: print('scored', len(sc), '/', len(keys), flush=True)
    cand = sorted([s for s in sc if s[2] < 0.15], key=lambda s: -s[1]); sel = cand[:NSEL]
    print(f'candidates {len(cand)} selected {len(sel)} burstiness range {sel[-1][1]:.1f}..{sel[0][1]:.1f} dB', flush=True)
    rng = np.random.default_rng(0); rng.shuffle(sel); nval = max(50, len(sel) // 20)
    for name, keys_sel, nsyn in [('noise_transient_train.hdf5', sel[nval:], 900), ('noise_transient_valid.hdf5', sel[:nval], 60)]:
        with h5py.File(src, 'r') as hi, h5py.File(os.path.join(out, name), 'w') as ho:
            for k, v in fattrs.items(): ho.attrs[k] = v
            ho.attrs['db_name'] = name; g = ho.create_group('noise')
            for k, _, _ in keys_sel:
                d = g.create_dataset(k, data=hi['noise'][k][()])
                for ak, av in hi['noise'][k].attrs.items(): d.attrs[ak] = av
            for i in range(nsyn):
                kind = ['clap', 'click', 'knock'][i % 3]; y = synth(rng, kind)
                d = g.create_dataset(f'syn_{kind}_{i:04d}', data=y[None]); d.attrs['n_samples'] = np.array([len(y)])
        print(name, len(keys_sel), '+', nsyn, 'synthetic', flush=True)
    print('TRANSIENTS_DONE', flush=True)
