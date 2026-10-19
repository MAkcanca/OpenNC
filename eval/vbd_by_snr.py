"""VoiceBank-DEMAND metrics split by input SNR (estimated per clip from clean and noisy). Usage (WSL): vbd_by_snr.py <system_dir>..."""
import sys, os, glob, numpy as np, soundfile as sf, librosa
from multiprocessing import Pool
from pesq import pesq
from pystoi import stoi
from align_util import align
C = os.environ.get('VBD_CLEAN', 'voicebank-demand/clean_testset_wav'); N = os.environ.get('VBD_NOISY', 'voicebank-demand/noisy_testset_wav')
def find(d, name):
    p = os.path.join(d, name)
    if os.path.exists(p): return p
    g = glob.glob(os.path.join(d, name[:-4] + '_*.wav')); return g[0] if g else None
def one(args):
    name, d = args
    c, sr = sf.read(os.path.join(C, name), dtype='float32'); x, _ = sf.read(os.path.join(N, name), dtype='float32')
    n = min(len(c), len(x)); snr = 10 * np.log10(np.sum(c[:n] ** 2) / (np.sum((x[:n] - c[:n]) ** 2) + 1e-12))
    y, sr2 = sf.read(find(d, name), dtype='float32'); y = y if y.ndim == 1 else y.mean(1)
    if sr2 != sr: y = librosa.resample(y, orig_sr=sr2, target_sr=sr)
    y, _ = align(c, y, sr)
    m = min(len(c), len(y)); c, y = c[:m], y[:m]
    return snr, pesq(16000, librosa.resample(c, orig_sr=sr, target_sr=16000), librosa.resample(y, orig_sr=sr, target_sr=16000), 'wb'), stoi(c, y, sr)
if __name__ == '__main__':
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(C, '*.wav')))
    with Pool(12) as pool:
        for d in sys.argv[1:]:
            r = np.array(pool.map(one, [(n, d) for n in names], chunksize=8)); edges = [-99, 5, 10, 15, 99]
            print(os.path.basename(d.rstrip('/')), end=': ')
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (r[:, 0] >= lo) & (r[:, 0] < hi); print(f'SNR {lo}..{hi} dB n={m.sum()} PESQ {r[m, 1].mean():.2f} STOI {r[m, 2].mean():.3f} | ', end='')
            print(flush=True)
