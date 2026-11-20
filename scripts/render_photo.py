"""Render source and shifted views of the dense geometry for honest visual review."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('EGL_PLATFORM','surfaceless')
import mujoco
import numpy as np
from PIL import Image,ImageDraw

p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,default=Path('outputs/geometry'));p.add_argument('--scene',type=Path);args=p.parse_args()
output=args.scene or args.geometry
data=np.load(args.geometry/'predictions.npz');m=mujoco.MjModel.from_xml_path(str(args.scene/'scene.xml' if args.scene else args.geometry/'photo_visual.xml'));d=mujoco.MjData(m)
index=min(range(len(data['frames'])),key=lambda i:abs(int(Path(str(data['frames'][i])).stem.split('_')[-1])-17));cid=m.camera(f'source_{index:03d}').id
path=Path(str(data['frames'][index]))
source=Image.open(path).convert('RGB') if path.exists() else Image.fromarray(data['colors'][index])
w=480;h=min(900,round(w*source.height/source.width));source=source.resize((w,h))
mujoco.mj_forward(m,d);right=d.cam_xmat[cid].reshape(3,3)[:,0];original=m.cam_pos[cid].copy()
images=[]
with mujoco.Renderer(m,height=h,width=w) as renderer:
    for label,shift in [('photo_source_view',0),('photo_shifted_20cm',.20),('photo_shifted_40cm',.40)]:
        m.cam_pos[cid]=original+right*shift;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cid)
        rgb=renderer.render().copy();Image.fromarray(rgb).save(output/f'{label}.png');images.append(Image.fromarray(rgb))
sheet=Image.new('RGB',(w*3,h+70),'#172326');draw=ImageDraw.Draw(sheet)
for i,(image,label) in enumerate([(source,'Original source pixels'),(images[0],'Reconstructed source camera'),(images[1],'Novel camera: shifted 0.20 provisional m')]):
    sheet.paste(image,(i*w,30));draw.text((i*w+8,10),label,fill='white')
draw.text((8,h+43),'Photo surfaces only; overlaps/occlusion holes may remain. Metric scale is assumed, and collision geometry is separate.',fill='white')
sheet.save(output/'photo_comparison.png')
print(json.dumps(dict(camera=index,source=str(path),outputs=['photo_source_view.png','photo_shifted_20cm.png','photo_shifted_40cm.png','photo_comparison.png']),indent=2))
