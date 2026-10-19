#!/usr/bin/env bash
# Build DeepFilterNet HDF5 sets on the box from the unpacked DNS4 shards + MUSAN music. All sources end up 48 kHz int16.
set -e
cd /workspace/data
PY=/workspace/opennc/venv/bin/python
PREP=$($PY -c "import df,os; print(os.path.join(os.path.dirname(df.__file__),'scripts','prepare_data.py'))")
mkdir -p musan48 lists h5
# MUSAN music is 16 kHz: convert to 48 kHz int16 wav so prepare_data stores int16 (torchaudio returns float32 for resampled input).
find musan/music -name '*.wav' | xargs -P 48 -I{} sh -c 'f="{}"; o="musan48/$(basename "$f")"; [ -f "$o" ] || ffmpeg -loglevel error -y -i "$f" -ar 48000 -ac 1 -c:a pcm_s16le "$o"'
$PY - <<'PY'
import glob, random
random.seed(20261011)
D='/workspace/data/datasets_fullband'
sp = glob.glob(f'{D}/clean_fullband/read_speech/*.wav') + glob.glob(f'{D}/clean_fullband/vctk_wav48_silence_trimmed/**/*.wav', recursive=True) + glob.glob(f'{D}/clean_fullband/emotional_speech/**/*.wav', recursive=True)
nz = glob.glob(f'{D}/noise_fullband/*.wav') + glob.glob('/workspace/data/musan48/*.wav')
rr = glob.glob(f'{D}/impulse_responses/**/*.wav', recursive=True)
for name, files, nv in (('speech', sp, 1000), ('noise', nz, 400), ('rir', rr, 200)):
    random.shuffle(files)
    open(f'/workspace/data/lists/{name}_valid.txt','w').write('\n'.join(files[:nv])+'\n')
    open(f'/workspace/data/lists/{name}_train.txt','w').write('\n'.join(files[nv:])+'\n')
    print(name, len(files)-nv, nv)
PY
build() { [ -f h5/$3.hdf5 ] && { echo skip $3; return; }; $PY "$PREP" --sr 48000 --num_workers 32 "${@:4}" "$1" "lists/$2.txt" "h5/$3.hdf5" > "h5/$3.log" 2>&1 && echo "built $3" || echo "FAILED $3"; }
build speech speech_valid speech_valid &
build noise noise_valid noise_valid --mono &
build rir rir_valid rir_valid &
wait
build speech speech_train speech_train &
build noise noise_train noise_train --mono &
build rir rir_train rir_train &
wait
cat > h5/dataset.cfg <<'CFG'
{"train":[["speech_train.hdf5",1.0],["noise_train.hdf5",1.0],["rir_train.hdf5",1.0]],
 "valid":[["speech_valid.hdf5",1.0],["noise_valid.hdf5",1.0],["rir_valid.hdf5",1.0]],
 "test":[["speech_valid.hdf5",1.0],["noise_valid.hdf5",1.0],["rir_valid.hdf5",1.0]]}
CFG
$PY - <<'PY'
# ponytail: the one check that matters — every group int16, no float32 mixed in (see dfn3-finetune-setup pitfall)
import h5py, numpy as np, glob
bad = 0
for f in sorted(glob.glob('/workspace/data/h5/*.hdf5')):
    with h5py.File(f) as h:
        g = h[list(h.keys())[0]]; ks = list(g.keys()); d = g[ks[0]][...]
        print(f.split('/')[-1], len(ks), d.dtype, d.shape, int(np.abs(d).max()))
        bad += d.dtype != np.int16
assert bad == 0, 'non-int16 group found'
PY
ls -la h5; df -h / | tail -1; echo PREPARE_DONE
