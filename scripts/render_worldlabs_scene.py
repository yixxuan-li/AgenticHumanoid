"""Review the exported Atlas mesh from real and generated camera viewpoints."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('EGL_PLATFORM','surfaceless')
import imageio.v2 as imageio,mujoco,numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('outputs/worldlabs'));p.add_argument('--scene',type=Path,default=Path('outputs/worldlabs_scene'));p.add_argument('--video',action='store_true');a=p.parse_args();mf=json.loads((a.scene/'manifest.json').read_text());m=mujoco.MjModel.from_xml_path(str(a.scene/'scene.xml'));d=mujoco.MjData(m)
for _ in range(600):mujoco.mj_step(m,d)
review=a.scene/'review';review.mkdir(exist_ok=True);selected=json.loads((a.workspace/'selected_frames.json').read_text())
with mujoco.Renderer(m,height=852,width=480) as renderer:
 sheet=Image.new('RGB',(480,1824),'#172326');draw=ImageDraw.Draw(sheet)
 for row,index in enumerate(np.unique(np.linspace(0,len(selected)-1,min(4,len(selected)),dtype=int))):
  renderer.update_scene(d,camera=f'source_{index:03d}');result=Image.fromarray(renderer.render());result.save(review/f'source_{index:03d}.png');source=Image.open(selected[index]['path']).convert('RGB');source.thumbnail((240,426));result.thumbnail((240,426));sheet.paste(source,(0,row*456+24));sheet.paste(result,(240,row*456+24));draw.text((4,row*456+5),'REAL: '+Path(selected[index]['path']).name,fill='white');draw.text((244,row*456+5),'ATLAS MESH',fill='white')
 sheet.save(a.scene/'source_comparison.jpg')
 conversion=mf['worldlabs_native_to_sim'];C=np.array(conversion['rotation']);s=conversion['scale'];offset=np.r_[conversion['translation_xy'],0.];floor=conversion['floor_y'];camid=m.camera('source_000').id;original_pos=m.cam_pos[camid].copy();original_quat=m.cam_quat[camid].copy();original_fovy=m.cam_fovy[camid];novel=Image.new('RGB',(720,460),'#172326');draw=ImageDraw.Draw(novel)
 for column,item in enumerate((json.loads((a.workspace/'curation.json').read_text()).get('retained_generated',[]) if (a.workspace/'curation.json').exists() else [])):
  f=json.loads((a.workspace/(item['stage']+'_operation.json')).read_text())['response']['frames'][item['index']];e=f['camera']['extrinsics'];k=f['camera']['intrinsics'];R=C@Rotation.from_quat(e['quaternion']).as_matrix();m.cam_pos[camid]=s*(C@np.array(e['position'])-[0,0,floor])+offset;m.cam_quat[camid]=Rotation.from_matrix(R).as_quat(scalar_first=True);m.cam_fovy[camid]=np.rad2deg(2*np.arctan(k['height']/(2*k['fy'])));mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camid);image=Image.fromarray(renderer.render());image.save(review/f'novel_{column:03d}.png');image.thumbnail((240,426));novel.paste(image,(column*240,24));draw.text((column*240+3,4),'ATLAS MESH: novel '+str(column+1),fill='white')
 novel.save(a.scene/'novel_views.jpg');m.cam_pos[camid]=original_pos;m.cam_quat[camid]=original_quat;m.cam_fovy[camid]=original_fovy
 if a.video:
  ids=[m.camera(f'source_{i:03d}').id for i in range(len(selected))];positions=m.cam_pos[ids].copy();rotations=Rotation.from_quat(m.cam_quat[ids],scalar_first=True);times=np.array([x.get('time_seconds',x['source_extracted_index']*.5) for x in selected]);orient=Slerp(times,rotations);fovy=m.cam_fovy[ids].copy()
  with imageio.get_writer(a.scene/'walkthrough.mp4',fps=24,macro_block_size=1) as writer:
   for t in np.linspace(times[0],times[-1],int((times[-1]-times[0])*24)+1):
    m.cam_pos[camid]=[np.interp(t,times,positions[:,axis]) for axis in range(3)];m.cam_quat[camid]=orient(t).as_quat(scalar_first=True);m.cam_fovy[camid]=np.interp(t,times,fovy);mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camid);writer.append_data(renderer.render())
for name in ['photo_cutaway_ceiling','photo_cutaway_front']:
 b=m.body(name);m.geom_rgba[b.geomadr[0]:b.geomadr[0]+b.geomnum[0],3]=0
with mujoco.Renderer(m,height=900,width=1200) as renderer:
 camera=mujoco.MjvCamera();camera.lookat=[mf['dimensions'][0]/2,mf['dimensions'][1]/2,.8];camera.distance=max(mf['dimensions'][:2])*1.15;camera.azimuth=-90;camera.elevation=-55;renderer.update_scene(d,camera=camera);Image.fromarray(renderer.render()).save(a.scene/'overview.png')
print('Saved source, novel-view, overview and optional video review in '+str(a.scene))
