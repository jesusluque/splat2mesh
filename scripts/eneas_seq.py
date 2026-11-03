"""ENEAS on an orbit sequence: for each phrase, ground it on the annotation frame and track it through all frames.
usage (in the ENEAS env): eneas_seq.py FRAMES_DIR OUT_NPZ ANNOTATION_FRAME[,FRAME2] "phrase" ["phrase" ...]
With two annotation frames (e.g. the input's view and the opposite one) the phrase is grounded and tracked from
each and the masks are united: tracking that fades halfway round the orbit is covered from the other side.
OUT_NPZ holds one bool array (n_frames, H, W) per phrase (key = phrase)."""
import sys, time
from pathlib import Path
import numpy as np
from eneas.segmentation import UniqueInstanceSegmenter
frames, out, ann = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
names = sorted(p.name for p in frames.glob("*.jpg"))
t = time.time(); seg = UniqueInstanceSegmenter(); print("loaded", round(time.time() - t), "s", flush=True)
res = {}
for ph in sys.argv[4:]:
    t = time.time()
    try:
        arr = None
        for a in ann.split(","):
            r = seg.segment(frames_path=str(frames), text=ph, annotation_frame=a)
            ks = sorted(r.masks)
            m = np.stack([np.asarray(r.masks[k]) > 127 for k in ks])
            if arr is None:
                arr = m
            else:
                n = min(len(arr), len(m)); arr = np.concatenate([arr[:n] | m[:n], arr[n:]])
        res[ph] = arr
        print(ph, "->", arr.shape, "mean cover", round(float(arr.mean()), 4), round(time.time() - t, 1), "s", flush=True)
    except Exception as e:
        print(ph, "ERROR", repr(e)[:300], flush=True)
np.savez_compressed(out, **res)
