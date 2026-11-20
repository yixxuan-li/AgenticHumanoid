"""Plan exploration waypoints on the settled scene; no WBC execution implied."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from scene_reconstruction.navigation import collision_grid,plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/scene'));p.add_argument('--radius',type=float,default=.28);args=p.parse_args()
    manifest=json.loads((args.scene/'manifest.json').read_text());m=mujoco.MjModel.from_xml_path(str(args.scene/'scene.xml'));d=mujoco.MjData(m)
    for _ in range(2000):mujoco.mj_step(m,d)
    grid=collision_grid(m,d,manifest['dimensions'],radius=args.radius)
    start=manifest['navigation']['start'];goals=manifest['navigation']['goals'];results=[]
    img=np.zeros((*grid['blocked'].shape,3),np.uint8);img[:]=[234,232,221];img[grid['occupied']]=[194,179,154];img[grid['blocked'].astype(bool)]=[49,56,59]
    for goal in goals:
        try:
            path=plan(grid,start,goal);indices=np.round(path/grid['resolution']).astype(int)
            clearance=min(grid['clearance'][y,x] for x,y in indices)
            results.append(dict(goal=goal,status='planned',length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),min_center_clearance_m=float(clearance),waypoints_m=path.tolist()))
            cv2.polylines(img,[indices],False,(33,167,143),2)
        except ValueError as exc:results.append(dict(goal=goal,status='unreachable',reason=str(exc)))
    img=Image.fromarray(np.flipud(img)).resize((1280,960),Image.Resampling.NEAREST)
    draw=ImageDraw.Draw(img);draw.rectangle((0,0,1280,40),fill='#172326');draw.text((12,12),f'Collision-space A* | body radius {args.radius:.2f} m | layout scale provisional | planned paths only',fill='white')
    img.save(args.scene/'navigation.png')
    report=dict(scene=str(args.scene),body_radius_m=args.radius,resolution_m=grid['resolution'],start=start,results=results,execution='NOT executed by a humanoid controller')
    Path('reports/navigation_plan.json').write_text(json.dumps(report,indent=2));print(json.dumps([{k:v for k,v in x.items() if k!='waypoints_m'} for x in results],indent=2))


if __name__=='__main__':main()
