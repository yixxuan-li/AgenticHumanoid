"""Combine actual MuJoCo/PhysX navigation recordings, without changing states."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/agentic-humanoid-matplotlib')
from pathlib import Path
import imageio_ffmpeg,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib,numpy as np
paths=[Path('outputs/controller/worldlabs_final_explore/explore_mode0.mp4'),Path('outputs/controller/worldlabs_native_explore/isaac_explore.mp4')];readers=[imageio_ffmpeg.read_frames(str(p),pix_fmt='rgb24',output_params=['-vf','fps=25,scale=640:480']) for p in paths];metadata=[next(r) for r in readers];fontpath=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf';font=ImageFont.truetype(str(fontpath),18);small=ImageFont.truetype(str(fontpath),15);out=Path('outputs/worldlabs_scene/navigation_both_engines.mp4');last=None
with imageio.get_writer(out,fps=25,macro_block_size=1) as writer:
 for buffers in zip(*readers):
  sheet=Image.new('RGB',(1280,550),'#172326');draw=ImageDraw.Draw(sheet)
  for i,(buffer,meta) in enumerate(zip(buffers,metadata)):sheet.paste(Image.frombytes('RGB',tuple(meta['size']),buffer),(640*i,40))
  draw.text((12,10),'MuJoCo: released ScaleBFM, pelvis mode',font=font,fill='white');draw.text((652,10),'Isaac Lab / PhysX: same scene-bound A* route',font=font,fill='white');draw.text((12,527),'World Labs mesh + fitted collision boxes | Assumed 0.74 m table height | Map/state agent | Static furniture',font=small,fill='white');writer.append_data(np.asarray(sheet));last=sheet
for reader in readers:reader.close()
if last is not None:last.save(out.with_suffix('.png'))
print(str(out))
