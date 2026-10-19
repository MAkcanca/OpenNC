# Measured results (2026-10-19)

Scripts: `eval_vbd.py`, `vbd_by_snr.py` (in the working folder), `eval_dnsmos.py`. Raw JSON next to this file.
Every system ran through its own released inference path: DeepFilterNet 2/3 via `deepFilter` (deepfilternet 0.5.6,
PyTorch, GPU), our models via `inference/engine.py` (numpy + onnxruntime, the shipped graph).

## VoiceBank-DEMAND test set (824 clips, 48 kHz)

| system | PESQ-WB | STOI | SI-SDR dB |
|---|---|---|---|
| noisy | 1.967 | 0.921 | 8.39 |
| DeepFilterNet2 | 3.054 | 0.944 | 15.69 |
| DeepFilterNet3 | 3.161 | 0.945 | 18.91 |
| stage 1 (from scratch) | 2.777 | 0.930 | 18.02 |
| stage 2 (released) | 2.514 | 0.911 | 18.19 |

Per input SNR (estimated per clip from clean and noisy; n = 219 / 210 / 204 / 191):

| system | < 5 dB PESQ / STOI | 5-10 dB | 10-15 dB | > 15 dB |
|---|---|---|---|---|
| DeepFilterNet3 | 2.63 / 0.912 | 3.08 / 0.945 | 3.36 / 0.959 | 3.64 / 0.969 |
| stage 1 | 2.29 / 0.893 | 2.67 / 0.925 | 2.95 / 0.947 | 3.26 / 0.959 |
| stage 2 | 2.16 / 0.882 | 2.44 / 0.907 | 2.65 / 0.922 | 2.85 / 0.937 |

Where the loss sits (`band_analysis.py`, the 191 clips above 15 dB input SNR, speech frames = clean frame > -40 dBFS;
mean output-minus-clean dB per band, then the share of speech frames more than 6 dB below the reference per band):

| system | 0-0.5k | 0.5-1k | 1-2k | 2-4k | 4-8k | 8-16k | > 6 dB loss %, same bands | clean-silent frames: out / ref |
|---|---|---|---|---|---|---|---|---|
| DeepFilterNet3 | -1.1 | -1.5 | -1.3 | -1.3 | -1.9 | -3.3 | 7 8 6 6 10 20 | -59 / -62 dB |
| stage 1 | -2.7 | -2.1 | -2.2 | -2.1 | -2.6 | -3.9 | 9 7 7 6 10 20 | -103 / -62 dB |
| stage 2 (released) | -4.5 | -3.7 | -4.2 | -4.5 | -7.1 | -14.4 | 10 10 12 13 27 62 | silence / -62 dB |

DeepFilterNet3 reproduces the reference including its floor between words; both our models silence the floor; stage 2
also removes quiet high-frequency speech components, which is the main PESQ/STOI cost and a real defect (see README).

Scored against gated references instead (the stage-2 training target applied to the clean test set): noisy 1.52,
DeepFilterNet3 2.31, stage 1 2.29, stage 2 2.48 PESQ-WB. The reference definition alone flips the ranking, which is why
both views are reported.

Reference-floor check: applying the stage-2 target gate (clip-level, depth 50 dB) to the clean references and scoring them
against the untouched references gives PESQ-WB 4.00, STOI 0.991 (the gate removes energy 36.6 dB below the speech).
So the reference's own floor explains part of the gap, not all of it.

## DNS Challenge 5 development set, Track 1 headset (641 real recordings, 10 s, 48 kHz)

DNSMOS P.835 (sig_bak_ovr.onnx) and P.808 (model_v8.onnx), Microsoft's non-personalized scorers, 16 kHz resample.

| system | SIG | BAK | OVRL | P.808 MOS |
|---|---|---|---|---|
| noisy | 3.085 | 2.669 | 2.294 | 3.069 |
| DeepFilterNet2 | 3.217 | 3.970 | 2.901 | 3.599 |
| DeepFilterNet3 | 3.167 | 3.951 | 2.841 | 3.545 |
| stage 1 | 3.077 | 3.901 | 2.756 | 3.579 |
| stage 2 (released) | 3.085 | 3.900 | 2.762 | 3.615 |

## Not reported here

Real-recording probes used during development (residual level in cells a reference denoiser silences during speech,
fricative dropout rate, transient leak) were measured against two commercial denoisers and are not reproducible without
them, so they are left out of the public report. The blind listening rounds that selected stage 2 are described in
`MODEL_CARD.md`.
