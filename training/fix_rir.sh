#!/usr/bin/env bash
# RIR wavs have an empty channel layout that torchaudio's ffmpeg backend rejects; rewrite as plain PCM_16 via soundfile, rebuild rir sets.
set -e
cd /workspace/data; mkdir -p rir48
PY=/workspace/opennc/venv/bin/python
$PY - <<'PY'
import glob, os, soundfile as sf, hashlib
from multiprocessing import Pool
src = glob.glob('/workspace/data/datasets_fullband/impulse_responses/**/*.wav', recursive=True)
def conv(p):
    o = '/workspace/data/rir48/' + hashlib.sha1(p.encode()).hexdigest()[:16] + '.wav'
    if not os.path.exists(o):
        x, sr = sf.read(p, dtype='int16')
        if x.ndim > 1: x = x[:, 0]
        sf.write(o, x, sr, subtype='PCM_16')
    return o
with Pool(48) as pool: outs = pool.map(conv, src, chunksize=64)
import random; random.seed(20261011); random.shuffle(outs)
open('/workspace/data/lists/rir_valid.txt', 'w').write('\n'.join(outs[:200]) + '\n')
open('/workspace/data/lists/rir_train.txt', 'w').write('\n'.join(outs[200:]) + '\n')
print('rir converted', len(outs))
PY
PREP=$($PY -c "import df,os; print(os.path.join(os.path.dirname(df.__file__),'scripts','prepare_data.py'))")
rm -f h5/rir_valid.hdf5 h5/rir_train.hdf5
$PY "$PREP" --sr 48000 --num_workers 8 rir lists/rir_valid.txt h5/rir_valid.hdf5 > h5/rir_valid.log 2>&1 && echo "built rir_valid" || echo "FAILED rir_valid"
$PY "$PREP" --sr 48000 --num_workers 16 rir lists/rir_train.txt h5/rir_train.hdf5 > h5/rir_train.log 2>&1 && echo "built rir_train" || echo "FAILED rir_train"
echo RIR_DONE
