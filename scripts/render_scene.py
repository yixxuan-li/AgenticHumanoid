"""Render current reconstruction stills and a physical-scene walkthrough."""
import argparse
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('EGL_PLATFORM','surfaceless')
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/scene'));p.add_argument('--video',action='store_true');args=p.parse_args()
m=mujoco.MjModel.from_xml_path(str(args.scene/'scene.xml'));d=mujoco.MjData(m)
for name in ['wall_entry_left','wall_entry_right','entry_lintel','ceiling_visual','wall_front_observed','wall_entry_front','photo_cutaway_ceiling','photo_cutaway_front']:
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
    if bid<0:continue
    adr=m.body_geomadr[bid];count=m.body_geomnum[bid];m.geom_rgba[adr:adr+count,3]=0
for _ in range(2000):mujoco.mj_step(m,d)
with mujoco.Renderer(m,height=720,width=960) as renderer:
    for c in ['entry','window_view','right_view','overview']:
        renderer.update_scene(d,camera=c);Image.fromarray(renderer.render()).save(args.scene/f'{c}.png')
    # A comparison intentionally labels the reconstructed scene as provisional.
    source=Image.open('outputs/inspection/frame_0016.jpg').convert('RGB');source.thumbnail((340,600))
    result=Image.open(args.scene/'window_view.png');result.thumbnail((800,600))
    sheet=Image.new('RGB',(1180,660),'#172326');sheet.paste(source,(20,40));sheet.paste(result,(370,40))
    draw=ImageDraw.Draw(sheet);draw.text((20,15),'SOURCE: 7.5 s, upright phone video',fill='white');draw.text((370,15),'SEMANTIC RECONSTRUCTION: estimated scale, procedural + source textures',fill='white')
    draw.text((20,630),'Different viewpoints; visual comparison is qualitative. Geometry, scale and materials are not yet surveyed.',fill='white');sheet.save(args.scene/'comparison.png')
    if args.video:
        with imageio.get_writer(args.scene/'walkthrough.mp4',fps=24,macro_block_size=1) as writer:
            for yaw in np.linspace(150,420,240):
                cam=mujoco.MjvCamera();cam.lookat=[3.2,2.4,.8];cam.distance=6.0;cam.azimuth=yaw;cam.elevation=-28
                renderer.update_scene(d,camera=cam);writer.append_data(renderer.render())
