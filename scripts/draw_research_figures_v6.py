"""Compose version 6 using only native Isaac Sim Gaussian render images."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from figure_layout import Teaser, system, INK, MUTED, BLUE, TEAL
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/figures_v6'

def teaser(out):
    a=out/'assets';c=Teaser();crop=(0,80,2200,1208)
    c.image(a/'hero_sequence.png',(30,30,2670,1370),region=crop)
    meta=json.loads((out/'hero_sequence_render.json').read_text())
    eye=np.array(meta['camera_position']);forward=np.array(meta['camera_target'])-eye
    forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    f=675/np.tan(np.deg2rad(meta['vertical_fov_degrees'])/2)
    poses=[[2.96,2.70],[2.30,3.70],[3.65,4.95],[4.75,4.90]]
    for word,xy in zip(['Pick','Turn','Carry','Place'],poses):
        rel=np.r_[xy,.05]-eye;z=rel@forward
        px=1100+f*(rel@right)/z;py=675-f*(rel@up)/z
        x=30+px*2670/2200;y=30+(py-crop[1])*1370/(crop[3]-crop[1])+12
        c.label(min(max(x-100,50),2440),min(max(y,400),1300),word,200,dot=True)
    c.box(55,55,390,265,color='none',edge='white',radius=15,lw=2)
    c.image(a/'onboard_pick.png',(66,66,368,243),layer=10);c.label(78,255,'Onboard',180)
    c.box(2243,1108,425,292,color='none',edge='white',radius=15,lw=2)
    c.image(a/'onboard_place.png',(2254,1119,403,270),layer=10);c.label(2266,1328,'Onboard',180)
    c.box(30,1416,2670,60,color='#F1F3F4',radius=8)
    c.text(1365,1429,'Move the cube from the table into the drawer.',13,INK,ha='center',style='italic')
    for i,(name,label) in enumerate(zip(['posture','reach','push','open_drawer','place','close_drawer'],['Posture','Reach','Push','Open drawer','Place','Close drawer'])):
        row,col=divmod(i,2);x=2740+col*715;y=30+row*385
        c.image(a/(name+'.png'),(x,y,695,370));c.label(x+12,y+306,label,245 if 'drawer' in label else 170)
    c.text(2745,1211,'AgenticHumanoid',14,INK,'bold')
    c.box(2740,1280,665,90,color='#EEF5FC',radius=10);c.text(3072,1307,'Skill programs',15,BLUE,'bold',ha='center')
    c.box(3470,1280,685,90,color='#FBF3E7',radius=10);c.text(3812,1307,'Symbolic knowledge',14,'#AB7429','bold',ha='center')
    c.text(3438,1301,'↔',24,TEAL,ha='center');c.text(2745,1424,'Explore → validate → reuse',12,TEAL)
    c.text(4155,1430,'Isaac Sim · 3DGS · authored poses',9,MUTED,ha='right')
    c.fig.canvas.draw();r=c.fig.canvas.get_renderer()
    for item in c.texts:
        box=item.get_window_extent(r)
        assert box.x0>=0 and box.y0>=0 and box.x1<=4200 and box.y1<=1500,item.get_text()
    for ext in ['png','pdf','svg']:c.fig.savefig(out/('teaser_v6.'+ext),dpi=200,facecolor='white')
    plt.close(c.fig)


def main():
    out=OUT;a=out/'assets'
    shutil.copyfile(ROOT/'outputs/figures_v5/assets/captured_13s.png',a/'captured_13s.png')
    shutil.copyfile(a/'room_probe.png',a/'reconstructed_room.png')
    teaser(out);system(out, version=6)
    native=json.loads((out/'scene/room_nurec.json').read_text())
    names=['hero_sequence','posture','reach','push','open_drawer','place','close_drawer','onboard_pick','onboard_place','room_probe']
    reports={n:json.loads((out/(n+'_render.json')).read_text()) for n in names}
    assert all(v['renderer'].startswith('Isaac Sim') and not v['room_mesh_appearance'] for v in reports.values())
    report=dict(figure_version=6,language='English',room_appearance='Atlas Gaussian splats rendered natively in Isaac Sim RTX / NuRec',
        conversion=native,views=reports,robot='Official G1 URDF-derived static visual snapshots; not a PhysX articulation import',
        pose_evidence='Authored figure poses only; no controller execution or learned skill claims',
        occlusion='Approximate authored tabletop matte proxy; no extracted room mesh appearance',
        original_room_and_splats_unchanged=True,external_image_generation=False,
        design_reference='https://arxiv.org/pdf/2611.08039, Figure 1 layout principles only',
        assets={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.glob('*.png'))})
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(out/'teaser_v6.png');print(out/'system_v6.png')


if __name__=='__main__':main()
