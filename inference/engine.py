r"""
Streaming reference engine: the released DeepFilterNet3-family model as a per-frame ONNX graph with explicit recurrent state.
Pure numpy + onnxruntime, 48 kHz, 10 ms hop, no lookahead. Usage as a library:

  nc = StreamingDenoiser(model_dir)
  cleaned = nc.process_frame(480_samples)   # float32[-1,1] at 48 kHz in, same out

Model dir holds dfn_stream.onnx, window960.npy, unit_norm_init160.npy and
stream_meta.json (ERB band widths, norm alpha). Features are numpy
re-implementations of the Rust libdf functions (verified to 2e-6).
"""
import os, json
import numpy as np
import onnxruntime as ort

SR, FFT_SIZE, HOP, N_BINS, N_DF, DF_ORDER, N_ERB = 48000, 960, 480, 481, 160, 5, 32


class StreamingDenoiser:
    SR = SR
    HOP = HOP

    HOPS_PER_CHUNK = 2   # 20 ms device buffers: at 10 ms the Windows audio loop underran about once a second (one-hop zero gaps, 2026-10-19)

    def __init__(self, model_dir, strength=100.0, mask_pow=1.0):
        """mask_pow=2 squares the band gains (stronger suppression); it costs 5-10 points of fricative dropouts on the gated fine-tune, so the released model uses plain gains (1.0)."""
        meta = json.load(open(os.path.join(model_dir, 'stream_meta.json')))
        self.alpha = float(meta['alpha']); widths = np.asarray(meta['erb_widths'], dtype=int)
        self.band_idx = np.concatenate([[0], np.cumsum(widths)])
        self.window = np.load(os.path.join(model_dir, 'window960.npy')).astype(np.float32)
        self.strength, self.mask_pow = float(strength), float(mask_pow)
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL; opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(os.path.join(model_dir, 'dfn_stream.onnx'), opts, providers=['CPUExecutionProvider'])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        # state
        self.frame = np.zeros(FFT_SIZE, np.float32)
        self.erb_state = np.linspace(-60, -90, N_ERB)                      # libdf erb_norm init
        self.unit_state = np.load(os.path.join(model_dir, 'unit_norm_init160.npy')).astype(np.float64)
        self.st = [np.zeros((1, 1, 2, N_ERB), np.float32), np.zeros((1, 2, 2, N_DF), np.float32), np.zeros((1, 64, 4, N_DF), np.float32),
                   np.zeros((1, 1, 512), np.float32), np.zeros((2, 1, 512), np.float32), np.zeros((2, 1, 512), np.float32)]
        self.hist = np.zeros((DF_ORDER, N_DF), np.complex64)   # masked spectrum history, oldest first
        self.oa = np.zeros(FFT_SIZE, np.float32)
        self.gate_close_db, self.gate_open_db = -85.0, -80.0
        self.frame_count = 0

    def set_strength(self, s):
        self.strength = float(s)

    def process_frame(self, new_samples):
        if len(new_samples) != HOP:
            raise ValueError(f"expected {HOP} samples, got {len(new_samples)}")
        fb = self.frame; fb[:HOP] = fb[HOP:]; fb[HOP:] = new_samples
        X = np.fft.rfft(fb * self.window, FFT_SIZE) / FFT_SIZE                       # libdf analysis scaling
        P = np.abs(X) ** 2; a = self.alpha
        E = 10 * np.log10(np.array([P[self.band_idx[i]:self.band_idx[i + 1]].mean() for i in range(N_ERB)]) + 1e-10)
        self.erb_state = E * (1 - a) + self.erb_state * a; feat_erb = ((E - self.erb_state) / 40).astype(np.float32)
        self.unit_state = np.abs(X[:N_DF]) * (1 - a) + self.unit_state * a; fs = (X[:N_DF] / np.sqrt(self.unit_state)).astype(np.complex64)
        feat_spec = np.stack([fs.real, fs.imag]).reshape(1, 2, 1, N_DF).astype(np.float32)
        out = self.sess.run(None, dict(zip(self.in_names, [feat_erb.reshape(1, 1, 1, N_ERB), feat_spec] + self.st)))
        gain, coefs = out[0], out[1]; self.st = out[3:]
        if self.mask_pow != 1.0:
            gain = gain ** self.mask_pow
        if self.strength < 100.0:
            s = self.strength / 100.0; gain = gain * s + (1.0 - s)
        X = X * gain
        self.hist[:-1] = self.hist[1:]; self.hist[-1] = X[:N_DF]
        cc = coefs[..., 0] + 1j * coefs[..., 1]                                       # [5,160], tap 0 = oldest
        X[:N_DF] = (cc * self.hist).sum(0)
        y = np.fft.irfft(X * FFT_SIZE, FFT_SIZE) * self.window
        oa = self.oa; oa[:HOP] = oa[HOP:]; oa[HOP:] = 0.0; oa += y
        out = oa[:HOP].astype(np.float32)
        # Output silence gate: the model's gains never reach exactly 0, so near-silent gaps carry a few LSB of residual that some
        # playback chains (Bluetooth, loudness enhancers) turn into audible static. Frames under -85 dBFS go to exact zero, -85..-80 ramp.
        rms_db = 20 * np.log10(np.sqrt(np.mean(out * out)) + 1e-12)
        if rms_db < self.gate_open_db:
            out *= np.float32(max(0.0, (rms_db - self.gate_close_db) / (self.gate_open_db - self.gate_close_db)))
        self.frame_count += 1
        return out

    @classmethod
    def process_file(cls, audio, model_dir, strength=100.0, mask_pow=1.0):
        """Whole-file helper; output is aligned with the input (the one-hop analysis delay is removed), like df.enhance."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1: audio = audio.mean(axis=1)
        nc = cls(model_dir, strength=strength, mask_pow=mask_pow)
        n_hops = (len(audio) + HOP - 1) // HOP + 1
        work = np.zeros(n_hops * HOP, np.float32); work[:len(audio)] = audio
        out = np.concatenate([nc.process_frame(work[i:i + HOP]) for i in range(0, len(work), HOP)])
        return out[HOP:HOP + len(audio)]

def _cli():
    import sys, time, soundfile as sf
    if len(sys.argv) < 4:
        print("usage: python engine.py model_dir input.wav output.wav [mask_pow]"); sys.exit(1)
    audio, sr = sf.read(sys.argv[2], dtype='float32')
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != SR:
        import librosa; audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    t0 = time.perf_counter(); y = StreamingDenoiser.process_file(audio, sys.argv[1], mask_pow=float(sys.argv[4]) if len(sys.argv) > 4 else 1.0); dt = time.perf_counter() - t0
    sf.write(sys.argv[3], y, SR)
    print(f"wrote {sys.argv[3]} ({len(y) / SR:.2f}s)  {dt / (len(y) / HOP) * 1000:.2f} ms per 10 ms frame, real-time factor {dt / (len(y) / SR):.3f}")


if __name__ == "__main__":
    _cli()
