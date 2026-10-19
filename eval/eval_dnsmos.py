"""DNSMOS P.835 (SIG/BAK/OVRL) + P.808 over directories of wavs, Microsoft's non-personalized scorer (training/dnsmos.py).
Usage (Windows python): eval_dnsmos.py <tag=dir>...  -> prints a table, writes dnsmos_<setname>.json next to this file.
Set SETNAME env to name the json (default 'set')."""
import sys, os, glob, json, numpy as np, soundfile as sf
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_D = None
def one(p):
    global _D
    if _D is None:
        from dnsmos import DNSMOS; _D = DNSMOS()
    x, sr = sf.read(p, dtype='float32'); x = x if x.ndim == 1 else x.mean(1)
    s = _D.score(x, sr); return [s['SIG'], s['BAK'], s['OVRL'], s['P808_MOS']]
if __name__ == '__main__':
    res = {}; out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'dnsmos_{os.environ.get("SETNAME", "set")}.json')
    with Pool(10) as pool:
        for arg in sys.argv[1:]:
            tag, d = arg.split('=', 1); files = sorted(glob.glob(os.path.join(d, '*.wav')))
            m = np.array(pool.map(one, files, chunksize=4))
            res[tag] = {'n': len(files), 'SIG': float(m[:, 0].mean()), 'BAK': float(m[:, 1].mean()), 'OVRL': float(m[:, 2].mean()), 'P808': float(m[:, 3].mean())}
            print(f'{tag:18s} n {len(files):4d}  SIG {m[:, 0].mean():.3f}  BAK {m[:, 1].mean():.3f}  OVRL {m[:, 2].mean():.3f}  P808 {m[:, 3].mean():.3f}', flush=True)
    json.dump(res, open(out, 'w'), indent=1); print('wrote', out)
