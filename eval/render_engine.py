"""Render a directory of wavs through the shipped streaming engine (what users download), in parallel.
Usage: render_engine.py <model_dir> <in_dir> <out_dir> [procs=6]   (output at the input's sample rate, same file names)"""
import sys, os, glob, numpy as np, soundfile as sf, librosa
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'inference'))
from engine import StreamingDenoiser, SR
MODEL, IN, OUT = sys.argv[1:4]; NP = int(sys.argv[4]) if len(sys.argv) > 4 else 6
def one(p):
    o = os.path.join(OUT, os.path.basename(p))
    if os.path.exists(o): return 0
    x, sr = sf.read(p, dtype='float32'); x = x if x.ndim == 1 else x.mean(1)
    y = StreamingDenoiser.process_file(librosa.resample(x, orig_sr=sr, target_sr=SR) if sr != SR else x, MODEL, mask_pow=1.0)
    if sr != SR: y = librosa.resample(y, orig_sr=SR, target_sr=sr)[:len(x)]
    sf.write(o, y, sr, subtype='PCM_16'); return 1
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); files = sorted(glob.glob(os.path.join(IN, '*.wav')))
    with Pool(NP) as pool: n = sum(pool.imap_unordered(one, files, chunksize=4))
    print(f'rendered {n} new / {len(files)} files -> {OUT}')
