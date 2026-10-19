"""DNSMOS proxies following Microsoft's non-personalized local scorer.

Reference: microsoft/DNS-Challenge/DNSMOS/dnsmos_local.py.
These predicted scores are not human listening ratings.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import onnxruntime as ort
import requests
from scipy.signal import resample_poly

MODEL_DIR=Path(__file__).resolve().parent/'dnsmos_models'   # sig_bak_ovr.onnx + model_v8.onnx, fetched from microsoft/DNS-Challenge on first use

def setup_models():
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    manifest=MODEL_DIR/'manifest.json'
    if manifest.exists():
        info=json.loads(manifest.read_text())
        if all((MODEL_DIR/r['file']).exists() and hashlib.sha256((MODEL_DIR/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in info['models']):return info
    response=requests.get('https://api.github.com/repos/microsoft/DNS-Challenge/commits/master',timeout=30)
    response.raise_for_status();commit=response.json()['sha']
    records=[]
    for name in ('sig_bak_ovr.onnx','model_v8.onnx'):
        url=f'https://raw.githubusercontent.com/microsoft/DNS-Challenge/{commit}/DNSMOS/DNSMOS/{name}'
        r=requests.get(url,timeout=60);r.raise_for_status()
        target=MODEL_DIR/name;temporary=target.with_suffix('.onnx.part')
        temporary.write_bytes(r.content)
        # Reject HTML, partial transfers or an invalid model before promotion.
        options=ort.SessionOptions();options.intra_op_num_threads=1
        ort.InferenceSession(str(temporary),options,providers=['CPUExecutionProvider'])
        temporary.replace(target)
        records.append({'file':name,'source':url,'sha256':hashlib.sha256(r.content).hexdigest()})
    info={'reference_commit':commit,'models':records,'reference_script':f'https://github.com/microsoft/DNS-Challenge/blob/{commit}/DNSMOS/dnsmos_local.py'}
    manifest.write_text(json.dumps(info,indent=2));return info

class DNSMOS:
    def __init__(self):
        self.provenance=setup_models()
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.primary=ort.InferenceSession(str(MODEL_DIR/'sig_bak_ovr.onnx'),options,providers=['CPUExecutionProvider'])
        self.p808=ort.InferenceSession(str(MODEL_DIR/'model_v8.onnx'),options,providers=['CPUExecutionProvider'])

    def score(self,audio,rate):
        import librosa
        audio=np.asarray(audio,dtype=np.float32)
        if not len(audio):raise ValueError('Empty audio')
        if rate!=16000:
            import math
            divisor=math.gcd(rate,16000)
            audio=resample_poly(audio,16000//divisor,rate//divisor).astype(np.float32)
        # Microsoft's short-clip policy repeats audio to a >=9.01 s segment.
        while len(audio)<144160:audio=np.concatenate((audio,audio))
        count=int(np.floor(len(audio)/16000)-9.01)+1
        scores=[]
        for index in range(count):
            segment=audio[index*16000:index*16000+144160]
            mel=librosa.feature.melspectrogram(y=segment[:-160],sr=16000,n_fft=321,hop_length=160,n_mels=120)
            features=((librosa.power_to_db(mel,ref=np.max)+40)/40).T.astype(np.float32)
            sig,bak,ovr=self.primary.run(None,{'input_1':segment[None].astype(np.float32)})[0][0]
            p808=self.p808.run(None,{'input_1':features[None]})[0][0][0]
            scores.append([np.polyval([-.08397278,1.22083953,.0052439],sig),
                           np.polyval([-.13166888,1.60915514,-.39604546],bak),
                           np.polyval([-.06766283,1.11546468,.04602535],ovr),p808])
        if not scores:raise RuntimeError('No full DNSMOS segments')
        return dict(zip(('SIG','BAK','OVRL','P808_MOS'),map(float,np.mean(scores,axis=0))))

if __name__=='__main__':print(json.dumps(setup_models(),indent=2))
