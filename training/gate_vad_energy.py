"""GPU-batched VAD gate + in-speech energy gate (box_gate3 plus: 20 ms frames below max(THR dBFS, clip p10+12 dB) inside VAD regions are gated too, HANG ms hangover).
Usage: box_gate4.py <src.hdf5> <dst.hdf5> <pad_ms> <min_silence_ms> <floor_gain> <a:b> <thr_dbfs> <hang_ms> [batch]"""
import sys, h5py, numpy as np, torch, torchaudio
from silero_vad import load_silero_vad
src, dst = sys.argv[1:3]; PAD, MINSIL, FLOOR = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
a, b = sys.argv[6].split(':'); THR, HANG = float(sys.argv[7]), int(sys.argv[8]); BATCH = int(sys.argv[9]) if len(sys.argv) > 9 else 256
CH = 512; MS = 1000 * CH / 16000; FR = 960  # 32 ms VAD chunks, 20 ms energy frames at 48 kHz
dev = 'cuda'; vad = load_silero_vad().to(dev); ramp = torch.hann_window(2 * 960, device=dev)[None, None]; ramp = ramp / ramp.sum()
H = HANG // 20; dil = torch.ones(1, 1, 2 * H + 1, device=dev)


def runs(mask):
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]]))); return idx[::2], idx[1::2]


def speech_mask(p):
    s = p > 0.5
    for st, en in zip(*runs(~s)):
        if (en - st) * MS < MINSIL and st > 0 and en < len(s): s[st:en] = True
    for st, en in zip(*runs(s)):
        if (en - st) * MS < 100: s[st:en] = False
    pad = int(round(PAD / MS)); out = s.copy()
    for st, en in zip(*runs(s)): out[max(0, st - pad):en + pad] = True
    return out


def energy_mask(x):  # x: (n,) on dev -> per-sample keep mask (n,)
    n = x.shape[0]; nf = (n + FR - 1) // FR; xp = torch.nn.functional.pad(x, (0, nf * FR - n))
    e = 20 * torch.log10(xp.reshape(nf, FR).pow(2).mean(1).sqrt() + 1e-9); thr = max(THR, float(torch.quantile(e, 0.1)) + 12)
    k = (e > thr).float(); k = (torch.nn.functional.conv1d(k[None, None], dil, padding=H)[0, 0] > 0).float()
    return k.repeat_interleave(FR)[:n]


with h5py.File(src, 'r') as hi:
    keys = list(hi['speech'].keys())[int(a):int(b) if b else None]; rattrs = dict(hi.attrs)
    lens = {k: hi['speech'][k].shape[1] for k in keys}
keys.sort(key=lens.get)
fr = []; ung = 0; done = 0
with h5py.File(src, 'r') as hi, h5py.File(dst, 'w') as ho, torch.no_grad():
    for k, v in rattrs.items(): ho.attrs[k] = v
    g = ho.create_group('speech')
    for i0 in range(0, len(keys), BATCH):
        ks = keys[i0:i0 + BATCH]; raws = [hi['speech'][k][()] for k in ks]; attrs = [dict(hi['speech'][k].attrs) for k in ks]
        L = max(r.shape[1] for r in raws); L = (L + 1535) // 1536 * 1536
        X = torch.zeros(len(ks), L, device=dev)
        for j, r in enumerate(raws): X[j, :r.shape[1]] = torch.from_numpy(r[0].astype(np.float32) / 32768).to(dev)
        X16 = torchaudio.functional.resample(X, 48000, 16000); n16 = X16.shape[1] // CH * CH; X16 = X16[:, :n16]
        vad.reset_states(); P = torch.stack([vad(X16[:, j * CH:(j + 1) * CH], 16000)[:, 0] for j in range(n16 // CH)], 1).cpu().numpy()
        for j, (k, r) in enumerate(zip(ks, raws)):
            n = r.shape[1]; nch = (n + 1535) // 1536; s = speech_mask(P[j, :nch])
            if not s.any(): y = r; ung += 1
            else:
                m = torch.from_numpy(np.repeat(s.astype(np.float32), 1536)[:n]).to(dev) * energy_mask(X[j, :n])
                m = torch.nn.functional.conv1d(m[None, None], ramp, padding=960)[0, 0, :n]; m = FLOOR + (1 - FLOOR) * m
                y = (X[j, :n] * m).clamp(-1, 1).mul(32767).round().to(torch.int16).cpu().numpy()[None]; fr.append(float((m < FLOOR + 0.01).float().mean()))
            o = g.create_dataset(k, data=y)
            for ak, av in attrs[j].items(): o.attrs[ak] = av
        done += len(ks)
        if (i0 // BATCH) % 20 == 0: print(done, len(keys), flush=True)
print(f'GATE_DONE {dst} clips {len(keys)} ungated {ung} silenced_frac {np.mean(fr):.3f}', flush=True)
