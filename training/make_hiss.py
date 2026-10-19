"""Synthetic electrical hiss / mains hum / whine noise HDF5s (48 kHz int16) in the DFN3 noise layout. Usage: make_hiss.py <out_dir>."""
import sys, os, h5py, numpy as np
out = sys.argv[1]; sr = 48000; T = 10; rng = np.random.default_rng(0)
f = np.fft.rfftfreq(sr * T, 1 / sr); t = np.arange(sr * T) / sr
def clip():
    tilt = rng.uniform(-12, 0)                                   # dB/octave above 1 kHz: white .. brownish
    X = np.fft.rfft(rng.standard_normal(sr * T)) * np.maximum(f, 200) ** (tilt / 6.02)
    x = np.fft.irfft(X)
    if rng.random() < 0.6:                                        # mains hum harmonics
        f0 = rng.choice([50.0, 60.0]); h = sum(rng.uniform(0.2, 1) / k * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28)) for k in range(1, 13))
        x = x / (np.std(x) + 1e-9) + h / (np.std(h) + 1e-9) * 10 ** (rng.uniform(-20, 5) / 20)
    if rng.random() < 0.3:                                        # high-frequency whine (switching supply / USB)
        x = x / (np.std(x) + 1e-9) + np.sin(2 * np.pi * rng.uniform(6000, 16000) * t) * 10 ** (rng.uniform(-30, -10) / 20)
    return (x / (np.std(x) + 1e-9) * 10 ** (-30 / 20) * 32767).clip(-32768, 32767).astype(np.int16)
for name, n in [('noise_hiss_train.hdf5', 400), ('noise_hiss_valid.hdf5', 40)]:
    with h5py.File(f'{out}/{name}', 'w') as h:
        for k, v in dict(codec='pcm', db_id=rng.integers(1 << 30), db_name=name, dtype='int16', max_freq=24000, sr=48000).items(): h.attrs[k] = v
        g = h.create_group('noise')
        for i in range(n):
            d = g.create_dataset(f'hiss_{i:04d}', data=clip()[None]); d.attrs['n_samples'] = np.array([sr * T])
    print(name, n)
