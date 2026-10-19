#!/usr/bin/env bash
set -e
mkdir -p /workspace/opennc && cd /workspace/opennc
export UV_CACHE_DIR=/workspace/uv-cache
uv python install 3.11 2>&1 | tail -1
[ -d venv ] || uv venv venv --python 3.11
uv pip install --python venv/bin/python torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python venv/bin/python "deepfilternet[train]==0.5.6" h5py librosa soundfile
venv/bin/python -c "import torch, df; print(torch.__version__, torch.cuda.device_count(), df.__file__)"
venv/bin/python -c "from df.deepfilternet3 import ModelParams; print('df ok')"
echo SETUP_DONE
