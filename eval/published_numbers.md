# Published numbers used for comparison

Copied from the papers / READMEs on 2026-10-19 so the README comparison can be checked. Our own runs of DeepFilterNet 2 and 3
on the same test set reproduce the published PESQ within 0.03, which validates the metric pipeline in this folder.

## VoiceBank-DEMAND test set (824 clips)

| model | params | PESQ-WB | STOI | source |
|---|---|---|---|---|
| Noisy | - | 1.97 | 0.921 | DFN2 paper, Table 1 |
| RNNoise | 0.06M | 2.33 | 0.922 | DFN2 paper |
| NSNet2 | 6.17M | 2.47 | 0.903 | DFN2 paper |
| PercepNet | 8.0M | 2.73 | - | DFN2 paper |
| DCCRN | 3.7M | 2.54 | 0.938 | DFN2 paper |
| FullSubNet+ | 8.67M | 2.88 | 0.940 | DFN2 paper |
| GTCRN | 24K | 2.87 | 0.940 | github.com/Xiaobin-Rong/gtcrn |
| Demucs (causal, H=48) | - | 2.93 | 0.95 | arXiv 2006.12847 |
| DeepFilterNet | 1.78M | 2.81 | 0.942 | DFN2 paper |
| DeepFilterNet2 | 2.31M | 3.08 | 0.943 | arXiv 2205.05474 |
| DeepFilterNet3 | 2.1M | 3.17 | 0.944 | arXiv 2305.08227 |

## DNSMOS on DNS blind sets (not the same set as ours; listed for scale only)

DNS4 blind test, DFN2 paper Table 2: Noisy SIG 4.14 / BAK 2.94 / OVRL 3.29; RNNoise 3.88 / 3.69 / 3.38; NSNet2 3.87 / 4.21 / 3.59;
DeepFilterNet2 4.20 / 4.43 / 3.88. DeepFilterNet3 has no published DNSMOS. GTCRN on DNS3 blind: OVRL 2.70 (noisy 2.33).

Our DNSMOS numbers are on the DNS5 development set, headset track (641 real recordings), which no paper reports, so the
only valid comparison is between the systems we ran ourselves on it (noisy, DeepFilterNet 2 and 3, our models).

## Test-set sources

- VoiceBank-DEMAND: https://datashare.ed.ac.uk/handle/10283/2791 (clean/noisy test zips, 147 MB + 163 MB, 48 kHz).
  Test speakers p232, p257; the 28-speaker training split does not contain them.
- DNS5 dev set: `download-dns5-...` scripts in https://github.com/microsoft/DNS-Challenge (V5_dev_testset.zip).
- DNSMOS: `DNSMOS/DNSMOS/sig_bak_ovr.onnx` and `model_v8.onnx` in the same repository; `eval/dnsmos.py` fetches them.
