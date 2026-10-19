"""Where does an enhancer deviate from the clean reference? VoiceBank clips with input SNR > 15 dB only (so noise is not the
story). Per 20 ms frame: speech frames = clean frame level > -40 dBFS; per band, mean (output - clean) dB and the share of
speech frames where the output is > 6 dB below clean; plus the output level in clean-silent frames (clean < -60 dBFS) and
the clean reference's own level there. Usage: band_analysis.py <system_dir>..."""
import sys, os, glob, numpy as np, soundfile as sf, librosa
from multiprocessing import Pool
C = os.environ.get('VBD_CLEAN', 'voicebank-demand/clean_testset_wav'); N = os.environ.get('VBD_NOISY', 'voicebank-demand/noisy_testset_wav')
B = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
def find(d, name):
    p = os.path.join(d, name)
    if os.path.exists(p): return p
    g = glob.glob(os.path.join(d, name[:-4] + '_*.wav')); return g[0] if g else None
def spec(x, sr):
    hop = int(sr * 0.02); S = np.abs(librosa.stft(x, n_fft=1024, hop_length=hop, center=False)) ** 2; f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    lvl = np.array([20 * np.log10(np.sqrt(np.mean(x[i * hop:i * hop + 1024] ** 2)) + 1e-9) for i in range(S.shape[1])])   # true dBFS per frame
    return np.array([10 * np.log10(S[(f >= lo) & (f < hi)].mean(0) + 1e-12) for lo, hi in B]), lvl
def one(args):
    name, d = args
    c, sr = sf.read(os.path.join(C, name), dtype='float32'); x, _ = sf.read(os.path.join(N, name), dtype='float32'); n = min(len(c), len(x))
    snr = 10 * np.log10(np.sum(c[:n] ** 2) / (np.sum((x[:n] - c[:n]) ** 2) + 1e-12))
    if snr < 15: return None
    y, sr2 = sf.read(find(d, name), dtype='float32'); y = y if y.ndim == 1 else y.mean(1)
    if sr2 != sr: y = librosa.resample(y, orig_sr=sr2, target_sr=sr)
    from align_util import align; y, _ = align(c, y, sr)
    m = min(len(c), len(y)); c, y = c[:m], y[:m]
    bc, lc = spec(c, sr); by, ly = spec(y, sr); sp = lc > -40; sil = lc < -60
    d = by[:, sp] - bc[:, sp]
    return [d.mean(1), (d < -6).mean(1), ly[sil].mean() if sil.sum() else np.nan, lc[sil].mean() if sil.sum() else np.nan, sp.sum(), sil.sum()]
if __name__ == '__main__':
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(C, '*.wav')))
    with Pool(6) as pool:
        for d in sys.argv[1:]:
            r = [x for x in pool.map(one, [(n, d) for n in names], chunksize=8) if x is not None]
            md = np.mean([x[0] for x in r], 0); dr = np.mean([x[1] for x in r], 0) * 100
            print(f'{os.path.basename(d.rstrip("/")):16s} n={len(r)}  speech-frame level vs clean, dB per band ' + ' '.join(f'{lo // 1000 if lo >= 1000 else lo / 1000:g}-{hi // 1000}k:{v:+.1f}' for (lo, hi), v in zip(B, md)) +
                  '  |  >6 dB loss %: ' + ' '.join(f'{v:.0f}' for v in dr) + f'  |  silent frames: out {np.nanmean([x[2] for x in r]):.1f} dB, clean ref {np.nanmean([x[3] for x in r]):.1f} dB', flush=True)
