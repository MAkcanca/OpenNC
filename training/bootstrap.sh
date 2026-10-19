#!/usr/bin/env bash
# Full box pipeline (fresh 4x4090 box): deps -> venv -> DNS4/MUSAN download -> HDF5 prepare -> VAD+energy gate (g5) -> TF gate (g6) + transient noise -> stage-2 fine-tune.
# Stage 1 (from scratch) uses the g5 sets + hiss: see TRAINING.md section 4; run it before the final step here (or change the last step).
# Usage: setsid nohup bash /root/scripts/box3_bootstrap.sh > /root/bootstrap.log 2>&1 < /dev/null &    Markers: "== <step> <time>", BOOTSTRAP_DONE / FAILED <step>
set -o pipefail; L=/root; S=/root/scripts; PY=/workspace/opennc/venv/bin/python; H=/workspace/data/h5; D=/workspace/opennc/venv/lib/python3.11/site-packages/df
step() { echo "== $1 $(date -u +%H:%M)"; }
fail() { echo "FAILED $1"; exit 1; }
step apt; (apt-get update -qq && apt-get install -y -qq aria2 lbzip2 ffmpeg) > $L/apt.log 2>&1; command -v aria2c >/dev/null && command -v lbzip2 >/dev/null && command -v ffmpeg >/dev/null || fail apt
command -v uv >/dev/null || pip install -q uv
step setup; for t in 1 2 3 4; do bash $S/setup.sh > $L/box_setup.log 2>&1 && grep -q SETUP_DONE $L/box_setup.log && break; echo "setup retry $t"; sleep 20; done; grep -q SETUP_DONE $L/box_setup.log || fail setup
uv pip install -q --python $PY silero-vad >> $L/box_setup.log 2>&1 || fail silero
step download; bash $S/download.sh > $L/download.log 2>&1; echo "ok $(grep -c '^OK' $L/download.log) fail $(grep -c '^FAIL' $L/download.log)"; grep -q "^FAIL" $L/download.log && fail download
step prepare; bash $S/prepare.sh > $L/prepare.log 2>&1; grep -q PREPARE_DONE $L/prepare.log || fail prepare
step fix_rir; bash $S/fix_rir.sh > $L/fix_rir.log 2>&1; grep -q RIR_DONE $L/fix_rir.log || fail fix_rir
step cleanup; rm -rf /workspace/data/datasets_fullband /workspace/data/musan /workspace/data/musan48 /workspace/data/rir48 /workspace/data/tar; df -h /workspace | tail -1 | awk '{print "free:", $4}'
step gate_g5; cd $H; N=$($PY -c "import h5py; print(len(h5py.File('speech_train.hdf5')['speech']))"); SH=$(( (N + 7) / 8 )); echo "clips $N"
for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$((i % 4)) $PY $S/gate_vad_energy.py speech_train.hdf5 speech_train_g5_$i.hdf5 250 250 0.01 $((i*SH)):$(( (i+1)*SH )) -50 60 > $L/gate_g5_$i.log 2>&1 & done
CUDA_VISIBLE_DEVICES=0 $PY $S/gate_vad_energy.py speech_valid.hdf5 speech_valid_g5.hdf5 250 250 0.01 0: -50 60 > $L/gate_g5_valid.log 2>&1 &
wait; [ "$(grep -h GATE_DONE $L/gate_g5_*.log | wc -l)" = 9 ] || fail gate_g5
step hiss; $PY $S/make_hiss.py $H > $L/hiss.log 2>&1 || fail hiss
step tfgate; for i in 0 1 2 3 4 5 6 7; do $PY $S/gate_tf_shard.py speech_train_g5_$i.hdf5 speech_train_g6_$i.hdf5 50 7 > $L/tfgate_$i.log 2>&1 & done
$PY $S/gate_tf_shard.py speech_valid_g5.hdf5 speech_valid_g6.hdf5 50 4 > $L/tfgate_valid.log 2>&1 &
$PY $S/make_transients.py noise_train.hdf5 $H 4000 8 > $L/transients.log 2>&1 &
wait; [ "$(grep -h TFGATE_DONE $L/tfgate_*.log | wc -l)" = 9 ] || fail tfgate; grep -q TRANSIENTS_DONE $L/transients.log || fail transients; tail -3 $L/transients.log
rm -f speech_train_g5_*.hdf5; df -h /workspace | tail -1 | awk '{print "free:", $4}'
step cfg; $PY - <<'EOF'
import json, glob, os
H = '/workspace/data/h5'
tr = [[os.path.basename(f), 1.0] for f in sorted(glob.glob(f'{H}/speech_train_g6_*.hdf5'))] + [["noise_train.hdf5", 1.0], ["rir_train.hdf5", 1.0], ["noise_hiss_train.hdf5", 1.0], ["noise_transient_train.hdf5", 3.0]]
va = [["speech_valid_g6.hdf5", 1.0], ["noise_valid.hdf5", 1.0], ["rir_valid.hdf5", 1.0], ["noise_transient_valid.hdf5", 1.0]]
json.dump({"train": tr, "valid": va, "test": va}, open(f'{H}/g6ht.cfg', 'w'), indent=1); print('g6ht.cfg', len(tr), 'train entries')
EOF
step venv_patch; $PY $S/patch_df.py $D && $PY $S/patch_ddp.py $D && $PY -m py_compile $D/train_ddp.py $D/deepfilternet3.py $D/utils.py || fail venv_patch
step train; R=/workspace/runs/stage2; mkdir -p $R/checkpoints; cp $S/configs/stage2_gated_finetune.ini $R/config.ini; cp /workspace/runs/stage1/checkpoints/model_113.ckpt.best $R/checkpoints/model_0.ckpt
cd $H; (DF_ON_MASKED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 setsid nohup /workspace/opennc/venv/bin/torchrun --nproc_per_node=4 $D/train_ddp.py g6ht.cfg $H $R > $R/train.stdout.log 2>&1 < /dev/null &); sleep 120; grep -E "\[0\] \[ *[0-9]+/" $R/train.stdout.log | tail -2 | cut -c1-150
echo BOOTSTRAP_DONE
