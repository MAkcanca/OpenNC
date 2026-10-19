# Model card: OpenNC v1

Real-time single-channel speech denoiser for voice chat and recording: 48 kHz, 10 ms hop, no lookahead, 8.17M parameters,
about 1 ms per frame on one CPU thread. Trained from scratch on open data with the DeepFilterNet trainer.

## Architecture

DeepFilterNet3 family (Schröter et al.), widened: 32 ERB bands, deep-filter on the lowest 160 bins (0-8 kHz, order 5),
encoder/decoder hidden size 512, GRU embedding, no convolutional or deep-filter lookahead. The deep filter is applied to
the mask-enhanced spectrum (the stock trainer filters the raw spectrum; see `training/patch_df.py`). Algorithmic latency is
one hop plus one window: 10 ms hop, 20 ms analysis window.

Files in a release:

| file | what | size |
|---|---|---|
| `dfn_stream.onnx` | per-frame streaming graph with explicit state (opset 14) | 33 MB |
| `window960.npy`, `unit_norm_init160.npy`, `stream_meta.json` | analysis window, spectral-norm init, ERB band widths | 5 KB |
| `model_15.ckpt.best`, `config.ini` | trainer checkpoint and config, for fine-tuning or re-export | 33 MB |

## Training data

DNS Challenge 4 full-band speech (LibriVox read speech, VCTK, CREMA-D), DNS4 AudioSet/Freesound noise, MUSAN music,
openSLR 26/28 room impulse responses, plus synthetic hiss/hum and transient (click, clap, knock) noise sets.
Mixtures at -5 to 40 dB SNR, 25 % reverberant, 3 s crops. Details, licenses and the exact shard list: `TRAINING.md`.

Targets were cleaned before training: VAD/energy gating outside speech and a time-frequency gate on cells far below the
speech level, so the model does not learn to reproduce the recordings' own noise floor under speech.

## Evaluation

The measured tables are in `README.md` (Results). Summary: on the synthetic VoiceBank-DEMAND mixtures the model scores
below DeepFilterNet 2/3 on PESQ and STOI; on real DNS5 headset recordings it is close to DeepFilterNet3 on DNSMOS P.835 and
slightly ahead on P.808. It was selected by blind listening on real recordings for its low residual floor under speech.

Protocol: VoiceBank-DEMAND test set (824 clips, 48 kHz; PESQ-WB on 16 kHz resamples, STOI, SI-SDR against the clean
references). DNS Challenge 5 development set, headset track (641 real recordings, 10 s, 48 kHz; DNSMOS P.835 SIG/BAK/OVRL
and P.808 MOS with Microsoft's non-personalized ONNX scorers). All systems were run through their own released
inference paths (DeepFilterNet 2/3 via the `deepFilter` CLI of deepfilternet 0.5.6; this model via `inference/engine.py`).
Scripts: `eval/`.

Caveat: VCTK speakers p232 and p257, the VoiceBank-DEMAND test speakers, are part of the DNS4 VCTK subset this model was
trained on (with different noise). Treat the VoiceBank speech as in-domain. The DNS5 recordings are fully unseen.

## Intended use and limitations

- Single near-field speaker over a headset or desk microphone; stationary and transient background noise.
- Not a dereverberation model; strong room reverb is only partially reduced.
- Does not separate competing speakers; background voices are attenuated only when they are much quieter.
- Fricatives in very quiet passages can lose high-frequency energy (about 10 % of 250 ms speech frames lose more than 6 dB
  in 4-8 kHz relative to a reference commercial denoiser on real recordings; this was the accepted trade for the lower
  residual floor).
- Trained on English-dominant speech; other languages were not evaluated.
- Near-silent output frames are gated to digital zero in the reference engine (below -85 dBFS); disable `gate_open_db` in
  `engine.py` if you need the raw output.

## License

Weights and code: Apache License 2.0. Training data licenses are listed in `TRAINING.md`; VCTK (ODC-By / CC BY 4.0),
AudioSet (CC BY 4.0), MUSAN (CC BY 4.0) and CREMA-D (ODC DbCL) require attribution, which this card provides.

## Citation

DeepFilterNet3: H. Schröter, T. Rosenkranz, A. Maier, "DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement",
Interspeech 2023. Trainer: https://github.com/Rikorose/DeepFilterNet (MIT / Apache 2.0).
