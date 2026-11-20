"""Assemble recorded simulator pixels; does not alter episode trajectories."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/agentic-humanoid-matplotlib')
from pathlib import Path
import imageio_ffmpeg,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import matplotlib
paths=[Path('outputs/controller/pick/final_pick/dex3_pick_trial.mp4'),Path('outputs/controller/isaac_pick/final_sequence/pick.mp4')]
readers=[imageio_ffmpeg.read_frames(str(p),pix_fmt='rgb24',output_params=['-vf','fps=25,scale=640:480']) for p in paths];meta=[next(r) for r in readers];font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),18);small=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),15);output=Path('outputs/final_scene/navigation_pick_both_engines.mp4');last=None
with imageio.get_writer(output,fps=25,macro_block_size=1) as writer:
 for i,buffers in enumerate(zip(*readers)):
  sheet=Image.new('RGB',(1280,550),'#172326');draw=ImageDraw.Draw(sheet)
  for j,(buffer,m) in enumerate(zip(buffers,meta)):
   image=Image.frombytes('RGB',tuple(m['size']),buffer).resize((640,480));sheet.paste(image,(j*640,40))
  draw.text((12,10),'MuJoCo: ScaleBFM body + Dex3 finger control',font=font,fill='white');draw.text((652,10),'Isaac Lab / PhysX: same physical skill sequence',font=font,fill='white');draw.text((12,527),'Assumed 0.74 m table height | Added red benchmark block | Map/simulator-state agent | No object weld or pose forcing',font=small,fill='white');writer.append_data(np.asarray(sheet));last=sheet
for r in readers:r.close()
if last is not None:last.save('outputs/final_scene/navigation_pick_both_engines.png')
print(str(output))
