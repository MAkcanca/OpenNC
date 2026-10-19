"""Two one-line patches to the installed DeepFilterNet 0.5.6 trainer (site-packages/df), applied in place with .orig backups:
  1. deepfilternet3.py: apply the deep-filter stage to the MASKED spectrum when DF_ON_MASKED=1 (the released models are trained
     and run this way; the stock trainer filters the raw spectrum).
  2. utils.py: honour a DF_DEVICE env var (e.g. DF_DEVICE=cpu) so exports/validation can run on CPU regardless of config.ini.
Usage: patch_df.py <site-packages/df>   (idempotent)"""
import sys, os, shutil
d = sys.argv[1]
def patch(name, old, new):
    p = os.path.join(d, name); s = open(p).read()
    if new in s: print(f'{name}: already patched'); return
    assert s.count(old) == 1, f'{name}: expected exactly one occurrence of the stock line'
    if not os.path.exists(p + '.orig'): shutil.copy(p, p + '.orig')
    open(p, 'w').write(s.replace(old, new)); print(f'{name}: patched')
patch('deepfilternet3.py',
      '            spec_e = self.df_op(spec.clone(), df_coefs)\n',
      '            spec_e = self.df_op((spec_m if os.environ.get("DF_ON_MASKED") else spec).clone(), df_coefs)  # DF on the masked spectrum\n')
s = open(os.path.join(d, 'deepfilternet3.py')).read()
if not s.startswith('import os'): open(os.path.join(d, 'deepfilternet3.py'), 'w').write('import os\n' + s)
patch('utils.py',
      '    s = config("DEVICE", default="", section="train")\n',
      '    s = os.environ.get("DF_DEVICE") or config("DEVICE", default="", section="train")\n')
