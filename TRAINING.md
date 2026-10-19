# Training

Everything here was run on a rented 4x RTX 4090 box (vast.ai, Ubuntu, 500 GB disk). The scripts assume that layout:
`/workspace/opennc/venv` (Python 3.11 venv), `/workspace/data` (datasets and HDF5 sets), `/workspace/runs/<RUN>` (training runs),
`/root/scripts` (a copy of `training/`). Change the paths at the top of each script if your box differs.

Total compute for the released model: about 10.5 GPU-hours x 4 for stage 1 and 1.5 x 4 for stage 2.

## 1. Environment

```bash
bash training/setup.sh          # uv venv, torch 2.4.1 cu124, deepfilternet[train]==0.5.6, h5py, librosa, soundfile
uv pip install --python /workspace/opennc/venv/bin/python silero-vad
apt-get install -y aria2 lbzip2 ffmpeg
```

Two one-line changes to the installed trainer, plus a data-parallel copy of `train.py`:

```bash
D=/workspace/opennc/venv/lib/python3.11/site-packages/df
python training/patch_df.py  $D     # DF_ON_MASKED=1: deep filter runs on the masked spectrum; DF_DEVICE env override
python training/patch_ddp.py $D     # writes $D/train_ddp.py (torchrun, NCCL, rank-0 checkpoints/logs)
```

`DF_ON_MASKED=1` must be set for every training, enhancement and export run of these models. Without it the deep-filter
stage runs on the unmasked spectrum, which is not how the weights were trained.

## 2. Data

`training/select.txt` lists the 24 DNS Challenge 4 full-band shards used (download: `training/download.sh`, aria2c, about 120 GB
packed). MUSAN music is added as extra noise.

| role | source | license (as stated in the DNS Challenge README) |
|---|---|---|
| speech | DNS4 `clean_fullband/read_speech` shards 017-024 (LibriVox readings, MOS-filtered) | public domain (LibriVox) |
| speech | DNS4 `clean_fullband/vctk_wav48_silence_trimmed` 000-004 (VCTK, all speakers) | ODC-By 1.0 (CC BY 4.0 at the Edinburgh source) |
| speech | DNS4 `clean_fullband/emotional_speech` (CREMA-D) | ODC Database Contents License 1.0 |
| noise | DNS4 `noise_fullband/audioset` 000-006 | CC BY 4.0 |
| noise | DNS4 `noise_fullband/freesound` 000-001 | CC0 |
| noise | MUSAN music (openSLR 17, resampled to 48 kHz) | CC BY 4.0 |
| reverb | DNS4 `impulse_responses_000` (openSLR 26 and 28) | Apache 2.0 |

Microsoft redistributes these "under the original terms that Microsoft received such datasets". The released weights are
derived from them; attribution for VCTK, AudioSet, MUSAN and CREMA-D is given here and in the model card.

`training/prepare.sh` builds the DeepFilterNet HDF5 sets (48 kHz, int16, 1000 speech / 400 noise / 200 RIR clips held out as
`valid`), `training/fix_rir.sh` rewrites the RIRs that torchaudio's ffmpeg backend rejects. Check every HDF5 group is int16:
a float32 group mixed with int16 ones puts the target 90 dB below the noise and the model learns to output silence.

Note: VCTK speakers p232 and p257 (the VoiceBank-DEMAND test speakers) are part of this speech set. The VoiceBank-DEMAND
numbers in the README are therefore in-domain for speech (not for noise) and are marked as such.

## 3. Targets

The clean recordings carry their own noise floor (room, mic, codec). A model trained on them learns to keep that floor
between harmonics, which is audible as static under speech. Three deterministic passes clean the targets:

1. `training/gate_vad_energy.py` (Silero VAD on GPU, batched): outside speech and in 20 ms frames below max(-50 dBFS,
   clip p10 + 12 dB) inside speech, attenuate to -40 dB with 250 ms padding and 60 ms hangover. Run per shard in parallel
   (8 processes on 4 GPUs, 156k clips in about 20 min).
2. `training/gate_tf_shard.py` -> `training/tf_gate.py`: time-frequency gate with the model's own STFT (960/480, Vorbis window):
   cells more than 50 dB below the clip's speech level are attenuated by 40 dB with a 6 dB ramp. Stage 2 used the clip-level
   gate (`depth 50`); the frame-relative v2 (`45 35`) gives the same floor with less soft-passage loss but did not change the
   results in our ablation.
3. Extra noise sets: `training/make_hiss.py` (synthetic electrical hiss, hum, whine) and `training/make_transients.py`
   (4000 burstiest DNS noise clips + synthetic claps, key clicks, knocks; weight 3.0 in the dataset config).

Exact commands and the dataset config writer are in `training/bootstrap.sh` (the full box pipeline, download to launch).

## 4. Stage 1: from scratch

Config: `training/configs/stage1_scratch.ini` (DeepFilterNet3 trainer, `nb_df 160`, hidden 512/512, no lookahead,
8.17M parameters, batch 64 per GPU, AdamW lr 2e-3 cosine, 120 epochs, 3 s crops, SNR -5..40 dB, 25 % reverb).
Data: VAD/energy-gated speech (pass 1 only) + DNS noise + hiss + RIRs. Best validation at epoch 112 (0.9196).

```bash
R=/workspace/runs/stage1; mkdir -p $R/checkpoints; cp training/configs/stage1_scratch.ini $R/config.ini
cd /workspace/data/h5 && DF_ON_MASKED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 $D/train_ddp.py g5h.cfg /workspace/data/h5 $R
```

About 5.2 min per epoch on 4x4090 (10.5 h). Log: `training/logs/stage1_scratch.log`.

## 5. Stage 2: fine-tune on TF-gated targets

Config: `training/configs/stage2_gated_finetune.ini` (lr 3e-4, 30 epochs max, patience 10). Start from the stage-1 best
checkpoint as `model_0.ckpt` (fresh optimizer). Data: pass 1 + pass 2 targets, DNS noise + hiss + transients (3x weight).
Validation loss 0.75 after one epoch, best 0.748 at epoch 15 (`model_15.ckpt.best`), stopped at 17. This is the released
model. Log: `training/logs/stage2_gated_finetune.log`.

```bash
R=/workspace/runs/stage2; mkdir -p $R/checkpoints; cp training/configs/stage2_gated_finetune.ini $R/config.ini
cp /workspace/runs/stage1/checkpoints/model_113.ckpt.best $R/checkpoints/model_0.ckpt
cd /workspace/data/h5 && DF_ON_MASKED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 $D/train_ddp.py g6ht.cfg /workspace/data/h5 $R
```

What stage 2 buys and costs, measured against the stage-1 model: the residual level in time-frequency cells that should be
silent during speech drops 4 to 8 dB on real recordings, at the cost of about 10 % of 250 ms speech frames losing more
than 6 dB in 4-8 kHz (fricatives). In blind listening on real recordings the stage-2 model was preferred.

Ablations that did not move that trade-off (not released, configs on request): frame-relative gate (v2), intra-clip level
augmentation (random 12-30 dB dips so the model sees soft passages), asymmetric spectral loss (under-estimation x4), and
linear weight interpolation between stage 1 and stage 2 (a dial along the same frontier).

## 6. Export to the streaming graph

```bash
cp inference/{stream_meta.json,window960.npy,unit_norm_init160.npy} /workspace/runs/stage2/
DF_DEVICE=cpu DF_ON_MASKED=1 python export/dfn_stream.py /workspace/runs/stage2 some_speech.wav export
```

`export/dfn_stream.py` wraps the trained model as a per-frame graph with explicit state (2-frame ERB and spectral
history, 4-frame deep-filter history, GRU states), validates it against `df.enhance` on the probe file (expected max
difference about 1e-5) and writes `dfn_stream.onnx` (33 MB, opset 14). `inference/engine.py` is the numpy + onnxruntime
reference that runs it: 10 ms hop, no lookahead, about 1 ms per frame on one CPU thread.
