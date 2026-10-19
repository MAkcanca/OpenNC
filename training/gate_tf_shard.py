"""Apply the time-frequency target gate (tf_gate.gate, depth 50 dB) to every clip of a speech HDF5 shard.
Usage: box_tfgate.py <src.hdf5> <dst.hdf5> [depth_db] [workers] [rel_db|none]   (int16 [1,N] datasets in group 'speech', attrs copied)"""
import sys, os, h5py, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from tf_gate import gate
src, dst = sys.argv[1:3]; DEPTH = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0; NW = int(sys.argv[4]) if len(sys.argv) > 4 else 6
REL = None if len(sys.argv) > 5 and sys.argv[5] == 'none' else (float(sys.argv[5]) if len(sys.argv) > 5 else 35.0)


def work(keys):
    out = []
    with h5py.File(src, 'r') as h:
        for k in keys:
            r = h['speech'][k][()]; x = r[0].astype(np.float32) / 32768
            y = gate(x, depth_db=DEPTH, rel_db=REL) if len(x) >= 1920 else x
            out.append((k, np.clip(y * 32767, -32768, 32767).round().astype(np.int16)[None], dict(h['speech'][k].attrs)))
    return out


if __name__ == '__main__':
    with h5py.File(src, 'r') as h:
        keys = list(h['speech'].keys()); fattrs = dict(h.attrs)
    chunks = [keys[i:i + 64] for i in range(0, len(keys), 64)]; n = 0
    with h5py.File(dst, 'w') as ho, Pool(NW) as pool:
        for k, v in fattrs.items(): ho.attrs[k] = v
        g = ho.create_group('speech')
        for res in pool.imap_unordered(work, chunks):
            for k, y, attrs in res:
                d = g.create_dataset(k, data=y)
                for ak, av in attrs.items(): d.attrs[ak] = av
            n += len(res)
            if n % 6400 < 64: print(n, len(keys), flush=True)
    print(f'TFGATE_DONE {dst} clips {n} depth {DEPTH} rel {REL}', flush=True)
