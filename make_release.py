"""Package the release assets: <name>-v1-stream.zip (streaming graph + support files) and <name>-v1-checkpoint.zip
(trainer checkpoint + config), print their sha256 for the README.
Usage: make_release.py <name> <stream_model_dir> <checkpoint_file> <config_ini> <out_dir>"""
import sys, os, zipfile, hashlib
name, sdir, ckpt, cfg, out = sys.argv[1:6]; os.makedirs(out, exist_ok=True)
def z(path, files):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files: zf.write(src, arc)
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest(); print(f'{os.path.basename(path)}  {os.path.getsize(path) / 1e6:.1f} MB  sha256 {h}')
z(os.path.join(out, f'{name}-v1-stream.zip'), [(os.path.join(sdir, f), f) for f in ('dfn_stream.onnx', 'window960.npy', 'unit_norm_init160.npy', 'stream_meta.json')])
z(os.path.join(out, f'{name}-v1-checkpoint.zip'), [(ckpt, 'checkpoints/' + os.path.basename(ckpt)), (cfg, 'config.ini')])
