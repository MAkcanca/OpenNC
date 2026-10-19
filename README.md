# OpenNC

Open real-time speech denoiser: 48 kHz, 10 ms frames, no lookahead, 8.2M parameters, about 1 ms per frame on one CPU
thread. Trained from scratch on open data (DNS Challenge 4, VCTK, MUSAN) with the DeepFilterNet trainer, released with the
full training pipeline, the evaluation scripts and the weights. Apache 2.0.

This repository is the model. A desktop app that runs it as a virtual microphone is a separate project.

## Download

| file | what | sha256 |
|---|---|---|
| `opennc-v1-stream.zip` | `dfn_stream.onnx` + `window960.npy`, `unit_norm_init160.npy`, `stream_meta.json` (what `inference/engine.py` loads) | 939d6cf1c4b7c50dedb0efd639748d027775e85d90178ed67a24c44daf78dd11 |
| `opennc-v1-checkpoint.zip` | `model_15.ckpt.best` + `config.ini` for the DeepFilterNet trainer (fine-tuning, re-export) | 441251d1490a6aa63425283a35e1f0242421135ebb2627bc43d029d802d7fe91 |

Both are attached to the [v1 release](https://github.com/MAkcanca/OpenNC/releases/tag/v1).

## Quick start

```bash
pip install -r inference/requirements.txt
unzip opennc-v1-stream.zip -d model
python inference/engine.py model noisy.wav clean.wav        # whole file (any sample rate; resampled to 48 kHz internally)
```

As a library, frame by frame (what a live app does):

```python
from inference.engine import StreamingDenoiser
nc = StreamingDenoiser('model')            # 48 kHz, 480-sample hop
clean = nc.process_frame(frame480)         # float32 in [-1, 1], returns the same length
```

The engine keeps the recurrent state between calls, so feed it consecutive 10 ms frames. Output is delayed by one hop
relative to the input. Frames under -85 dBFS are gated to digital zero (set `nc.gate_open_db = -200` to disable).

## Results

Measured on 2026-10-19 with the scripts in `eval/`; every system ran through its own released inference path.
DeepFilterNet 2 and 3 reproduce their published PESQ within 0.03, which is the check that the pipeline is right.

**VoiceBank-DEMAND test set** (824 clips, 48 kHz, synthetic noise at 2.5 to 17.5 dB SNR; intrusive metrics vs the clean reference)

| system | params | PESQ-WB | STOI | SI-SDR dB |
|---|---|---|---|---|
| noisy input | - | 1.97 | 0.921 | 8.4 |
| DeepFilterNet2 (published 3.08 / 0.943) | 2.3M | 3.05 | 0.944 | 15.7 |
| DeepFilterNet3 (published 3.17 / 0.944) | 2.1M | 3.16 | 0.945 | 18.9 |
| OpenNC v1 (this release, stage 2) | 8.2M | 2.51 | 0.911 | 18.2 |
| OpenNC stage 1 (from scratch, lighter target gating) | 8.2M | 2.78 | 0.930 | 18.0 |

**DNS Challenge 5 development set, headset track** (641 real recordings, 10 s, 48 kHz; non-intrusive DNSMOS P.835 and P.808)

| system | SIG | BAK | OVRL | P.808 MOS |
|---|---|---|---|---|
| noisy input | 3.09 | 2.67 | 2.29 | 3.07 |
| DeepFilterNet2 | 3.22 | 3.97 | 2.90 | 3.60 |
| DeepFilterNet3 | 3.17 | 3.95 | 2.84 | 3.55 |
| OpenNC v1 (this release) | 3.09 | 3.90 | 2.76 | 3.62 |
| OpenNC stage 1 | 3.08 | 3.90 | 2.76 | 3.58 |

How to read this: on the synthetic VoiceBank mixtures the model is well below DeepFilterNet 2/3 on PESQ and STOI; on real
headset recordings it is close to DeepFilterNet3 on P.835 and slightly ahead on P.808. The VoiceBank gap holds at every input
SNR (PESQ 2.16 vs 2.63 below 5 dB, 2.85 vs 3.64 above 15 dB; `eval/results.md`), so it is not only low-SNR over-suppression.
Part of it is the training target: the stage-2 gate removes the recordings' own floor, which the VoiceBank clean references
keep, and gating the references themselves already costs PESQ 4.64 -> 4.00. The rest is the model being more aggressive
than DeepFilterNet3 on quiet speech components. The model was selected
by blind listening on real recordings, where its lower residual floor under speech was preferred over DeepFilterNet3-family
baselines and over two commercial denoisers; that preference is not what PESQ measures, and the numbers above are reported
as they are. Caveats: the VoiceBank test speakers are part of the training speech (see `MODEL_CARD.md`), and the DNS5
development set has no published numbers for other systems, so only rows measured here are comparable.

`eval/published_numbers.md` lists the paper numbers for other open models on the same VoiceBank set (RNNoise 2.33,
NSNet2 2.47, GTCRN 2.87, FullSubNet+ 2.88 PESQ-WB).

Known issue (measured, `eval/results.md`): on clean-ish input the stage-2 model lowers quiet high-frequency speech
components, about 7 dB in 4-8 kHz and 14 dB in 8-16 kHz relative to the reference on speech frames, because the clip-level
target gate (threshold = clip peak minus 50 dB) reaches into the high-frequency speech range. Stage 1 does not have this.
A band-aware gate is the planned fix for v2.

## Repository layout

| path | contents |
|---|---|
| `inference/` | `engine.py` reference streaming engine (numpy + onnxruntime), stream support files, `requirements.txt` |
| `export/` | `dfn_stream.py`: trainer checkpoint -> validated per-frame ONNX graph |
| `training/` | the complete box pipeline: setup, data download/preparation, target gating, trainer patches, configs, logs |
| `eval/` | VoiceBank-DEMAND metrics, DNSMOS scoring, batch rendering, published numbers |
| `TRAINING.md` | how the model was trained, step by step, with data licenses |
| `MODEL_CARD.md` | architecture, data, evaluation protocol, limitations |

## Reproduce

See `TRAINING.md`. Stage 1 is 120 epochs (about 10.5 h on 4x RTX 4090), stage 2 is 15 epochs (1.5 h) on time-frequency
gated targets. `training/bootstrap.sh` runs the whole thing on a fresh box.

## Acknowledgements

Built on the [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) trainer and architecture (Schröter et al.).
Data: Microsoft DNS Challenge 4 (LibriVox, VCTK, CREMA-D, AudioSet, Freesound, openSLR RIRs), MUSAN. Evaluation:
VoiceBank-DEMAND (Valentini-Botinhao et al.), DNS Challenge 5, DNSMOS (Reddy et al.).

## License

Apache License 2.0 for the code and the weights. Dataset licenses are listed in `TRAINING.md`.
