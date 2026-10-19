"""VoiceBank-DEMAND test set metrics (824 clips): PESQ-WB (16 kHz), STOI, SI-SDR, per system directory vs the clean references.
Usage (WSL venv): eval_vbd.py <clean_dir> <noisy_dir> <system_dir>...   -> prints a table and writes vbd_results.json next to this file."""
import sys, os, glob, json, numpy as np, soundfile as sf, librosa
from multiprocessing import Pool
from pesq import pesq
from pystoi import stoi
CLEAN, NOISY = sys.argv[1], sys.argv[2]; SYS = sys.argv[3:]
from align_util import align
def si_sdr(ref, est):
    ref = ref - ref.mean(); est = est - est.mean(); a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12); t = a * ref; e = est - t
    return 10 * np.log10((np.sum(t ** 2) + 1e-12) / (np.sum(e ** 2) + 1e-12))
def find(d, name):   # deepFilter's CLI appends the model name: p232_001_DeepFilterNet3.wav
    p = os.path.join(d, name)
    if os.path.exists(p): return p
    g = glob.glob(os.path.join(d, name[:-4] + '_*.wav')); return g[0] if g else None
def one(args):
    name, d = args
    c, sr = sf.read(os.path.join(CLEAN, name), dtype='float32'); y, sr2 = sf.read(find(d, name), dtype='float32')
    if y.ndim > 1: y = y.mean(1)
    if sr2 != sr: y = librosa.resample(y, orig_sr=sr2, target_sr=sr)
    y, lag = align(c, y, sr)
    n = min(len(c), len(y)); c, y = c[:n], y[:n]
    c16, y16 = librosa.resample(c, orig_sr=sr, target_sr=16000), librosa.resample(y, orig_sr=sr, target_sr=16000)
    return pesq(16000, c16, y16, 'wb'), stoi(c, y, sr, extended=False), si_sdr(c, y), lag * 1000 / sr
if __name__ == '__main__':
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CLEAN, '*.wav')))
    res = {}
    with Pool(12) as pool:
        for d in [NOISY] + SYS:
            tag = os.path.basename(d.rstrip('/\\'))
            have = [n for n in names if find(d, n)]
            if len(have) < len(names): print(f'{tag}: {len(have)}/{len(names)} files present')
            m = np.array(pool.map(one, [(n, d) for n in have], chunksize=8))
            res[tag] = {'n': len(have), 'pesq_wb': float(m[:, 0].mean()), 'stoi': float(m[:, 1].mean()), 'si_sdr': float(m[:, 2].mean()), 'lag_ms_median': float(np.median(m[:, 3]))}
            print(f'{tag:18s} n {len(have):4d}  PESQ-WB {m[:, 0].mean():.3f}  STOI {m[:, 1].mean():.4f}  SI-SDR {m[:, 2].mean():6.2f} dB  (median lag {np.median(m[:, 3]):.1f} ms)', flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vbd_results.json'), 'w'), indent=1)
