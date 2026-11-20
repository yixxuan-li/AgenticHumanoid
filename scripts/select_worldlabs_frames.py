"""Select sharp time-ordered views across an extracted video without API calls."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--frames',type=Path,default=Path('outputs/inspection'))
p.add_argument('--workspace',type=Path,default=Path('outputs/worldlabs'))
p.add_argument('--count',type=int,default=12)
p.add_argument('--sample-fps',type=float,default=2.)
a=p.parse_args()
files=sorted(a.frames.glob('frame_[0-9][0-9][0-9][0-9].jpg'))
if not files or a.count<1 or a.sample_fps<=0:p.error('Need extracted frames, positive count and positive sample FPS')
destination=a.workspace/'selected_frames.json'
if destination.exists():raise FileExistsError('Frame selection already exists; inspect it or use a fresh workspace')
scores=[]
for f in files:
    image=cv2.imread(str(f),0)
    if image is None:raise ValueError(f'Cannot decode {f}')
    scores.append(float(cv2.Laplacian(image,cv2.CV_64F).var()))
chosen=[int(group[np.argmax(np.array(scores)[group])]) for group in np.array_split(np.arange(len(files)),min(a.count,len(files)))]
selected=[dict(path=str(files[i].resolve()),source_extracted_index=i+1,time_seconds=i/a.sample_fps,sharpness=scores[i]) for i in chosen]
a.workspace.mkdir(parents=True,exist_ok=True);a.workspace.chmod(0o700)
destination.write_text(json.dumps(selected,indent=2));destination.chmod(0o600)
print(f'Selected {len(selected)} views. Review overlap and scene coverage before uploading.')
