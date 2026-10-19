"""Frame-by-frame (streaming) wrapper for the S1D DeepFilterNet3-family model + numpy features; validates against df.enhance and exports ONNX.
Usage: dfn_stream.py <model_dir> <probe.wav> [export]   (probe.wav: any speech recording, used to validate stream vs offline)"""
import sys, os, json, time, numpy as np, torch, torchaudio, libdf
from torch import nn
from df.enhance import init_df, enhance
D = sys.argv[1]; model, state, _ = init_df(D, log_level='ERROR'); model = model.eval(); torch.set_num_threads(1)
meta = json.load(open(os.path.join(D, 'stream_meta.json'))); W = np.load(os.path.join(D, 'window960.npy')); FB = meta['erb_widths']; ALPHA = meta['alpha']
np.save(os.path.join(D, 'unit_norm_init160.npy'), np.asarray(libdf.unit_norm_init(160), dtype=np.float32).reshape(-1))


class StreamNet(nn.Module):
    """One frame in, one frame out. Histories: erb_hist [1,1,2,32], spec_hist [1,2,2,160], c0_hist [1,64,4,160]; GRU states h_enc [1,1,512], h_erb [2,1,512], h_df [2,1,512]."""
    def __init__(self, m):
        super().__init__(); self.enc, self.erb_dec, self.df_dec = m.enc, m.erb_dec, m.df_dec; self.register_buffer('erb_inv_fb', m.mask.erb_inv_fb.clone())

    def forward(self, feat_erb, feat_spec, erb_hist, spec_hist, c0_hist, h_enc, h_erb, h_df):
        e = self.enc
        x_erb = torch.cat([erb_hist, feat_erb], 2); x_spec = torch.cat([spec_hist, feat_spec], 2)  # 3 frames -> conv0 (k_t=3) -> take last
        e0 = e.erb_conv0(x_erb)[:, :, -1:]; e1 = e.erb_conv1(e0); e2 = e.erb_conv2(e1); e3 = e.erb_conv3(e2)
        c0 = e.df_conv0(x_spec)[:, :, -1:]; c1 = e.df_conv1(c0)
        cemb = e.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2)); emb = e.combine(e3.permute(0, 2, 3, 1).flatten(2), cemb)
        emb, h_enc = e.emb_gru(emb, h_enc)
        d = self.erb_dec; b, _, t, f8 = e3.shape
        embd, h_erb = d.emb_gru(emb, h_erb); embd = embd.view(b, t, f8, -1).permute(0, 3, 1, 2)
        y3 = d.convt3(d.conv3p(e3) + embd); y2 = d.convt2(d.conv2p(e2) + y3); y1 = d.convt1(d.conv1p(e1) + y2); m = d.conv0_out(d.conv0p(e0) + y1)  # [1,1,1,32]
        gain = m.reshape(1, 32).matmul(self.erb_inv_fb).reshape(481)
        g = self.df_dec; c, h_df = g.df_gru(emb, h_df); c = c + g.df_skip(emb)
        c0w = torch.cat([c0_hist, c0], 2); cp = g.df_convp(c0w)[:, :, -1:].permute(0, 2, 3, 1)  # 5 frames -> k_t=5 -> last
        coefs = (g.df_out(c).view(b, t, g.df_bins, g.df_out_ch) + cp).reshape(160, 5, 2).permute(1, 0, 2)  # [O, F, 2], tap 0 = oldest
        return gain, coefs, m.reshape(32), x_erb[:, :, 1:], x_spec[:, :, 1:], c0w[:, :, 1:], h_enc, h_erb, h_df


def features(x):
    """numpy: x float32 [N] at 48 kHz -> spec [T,481] complex, feat_erb [T,32], feat_spec [T,160] complex (matches libdf within 2e-6)."""
    xp = np.concatenate([np.zeros(480, np.float32), x]); T = (len(xp) - 960) // 480 + 1
    S = np.fft.rfft(np.stack([xp[i * 480:i * 480 + 960] for i in range(T)]) * W, 960) / 960
    idx = np.concatenate([[0], np.cumsum(FB)]).astype(int); P = np.abs(S) ** 2
    E = 10 * np.log10(np.stack([P[:, idx[i]:idx[i + 1]].mean(1) for i in range(32)], 1) + 1e-10)
    es = np.linspace(-60, -90, 32); us = np.load(os.path.join(D, 'unit_norm_init160.npy')).astype(np.float64); FE = np.zeros_like(E); FS = np.zeros((T, 160), np.complex64)
    for t in range(T):
        es = E[t] * (1 - ALPHA) + es * ALPHA; FE[t] = (E[t] - es) / 40
        us = np.abs(S[t, :160]) * (1 - ALPHA) + us * ALPHA; FS[t] = S[t, :160] / np.sqrt(us)
    return S.astype(np.complex64), FE.astype(np.float32), FS


def run_stream(net, x, mask_pow=1.0):
    S, FE, FS = features(x); T = len(S); out = np.zeros(480 * (T + 1), np.float32)
    st = [torch.zeros(1, 1, 2, 32), torch.zeros(1, 2, 2, 160), torch.zeros(1, 64, 4, 160), torch.zeros(1, 1, 512), torch.zeros(2, 1, 512), torch.zeros(2, 1, 512)]
    hist = np.zeros((5, 160), np.complex64)
    with torch.no_grad():
        for t in range(T):
            fe = torch.from_numpy(FE[t]).reshape(1, 1, 1, 32); fs = torch.view_as_real(torch.from_numpy(FS[t])).permute(1, 0).reshape(1, 2, 1, 160)
            gain, coefs, m, *st = net(fe, fs, *st)
            gain = gain.numpy() ** mask_pow if mask_pow != 1 else gain.numpy()
            Xm = S[t] * gain; hist = np.roll(hist, -1, 0); hist[-1] = Xm[:160]
            c = coefs.numpy(); cc = c[..., 0] + 1j * c[..., 1]; Xm[:160] = (cc * hist).sum(0)
            fr = np.fft.irfft(Xm * 960, 960) * W; out[t * 480:t * 480 + 960] += fr
    return out[480:480 + len(x)]


if __name__ == '__main__':
    a, sr = torchaudio.load(sys.argv[2]); a = torchaudio.functional.resample(a.mean(0, keepdim=True), sr, 48000)[:, :48000 * 6]
    ref = enhance(model, state, a.clone()).numpy()[0]  # on the trainer's device, before moving to CPU
    model = model.cpu(); net = StreamNet(model).eval()
    t0 = time.perf_counter(); y = run_stream(net, a[0].numpy()); dt = time.perf_counter() - t0
    print(f'stream vs enhance: maxdiff {np.abs(y - ref).max():.2e}  ref max {np.abs(ref).max():.3f}  torch per-frame {dt / (len(y) / 480) * 1000:.2f} ms (incl numpy features)')
    if len(sys.argv) > 3:
        args = (torch.zeros(1, 1, 1, 32), torch.zeros(1, 2, 1, 160), torch.zeros(1, 1, 2, 32), torch.zeros(1, 2, 2, 160), torch.zeros(1, 64, 4, 160), torch.zeros(1, 1, 512), torch.zeros(2, 1, 512), torch.zeros(2, 1, 512))
        names = ['feat_erb', 'feat_spec', 'erb_hist', 'spec_hist', 'c0_hist', 'h_enc', 'h_erb', 'h_df']
        p = os.path.join(D, 'dfn_stream.onnx')
        torch.onnx.export(net, args, p, input_names=names, output_names=['gain', 'coefs', 'mask', 'erb_hist_out', 'spec_hist_out', 'c0_hist_out', 'h_enc_out', 'h_erb_out', 'h_df_out'], opset_version=14)
        import onnxruntime as ort; so = ort.SessionOptions(); so.intra_op_num_threads = 1; sess = ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
        with torch.no_grad():
            ref_o = net(*args)
        o = sess.run(None, {n: v.numpy() for n, v in zip(names, args)}); print('onnx vs torch maxdiff', max(np.abs(r.numpy() - q).max() for r, q in zip(ref_o, o)), 'size MB', os.path.getsize(p) / 1e6)
        x = {n: v.numpy() for n, v in zip(names, args)}; t0 = time.perf_counter()
        for _ in range(300):
            sess.run(None, x)
        print(f'onnxruntime per-frame {(time.perf_counter() - t0) / 300 * 1000:.2f} ms (1 thread, WSL)')
