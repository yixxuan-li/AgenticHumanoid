"""Screen manipulation preconditions against the actual collision scene."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scene_reconstruction.navigation import collision_grid,plan

manifest=json.loads(Path('outputs/scene/manifest.json').read_text());m=mujoco.MjModel.from_xml_path('outputs/scene/scene.xml');d=mujoco.MjData(m)
for _ in range(2000):mujoco.mj_step(m,d)
grid=collision_grid(m,d,manifest['dimensions'],radius=.32)
start=np.array([1.85,.65]);obj=d.xpos[m.body('pick_box').id];candidates=[]
for distance in [.40,.45,.50,.55,.60]:
    for angle in np.linspace(0,2*np.pi,48,endpoint=False):
        point=obj[:2]+distance*np.array([np.cos(angle),np.sin(angle)])
        try:
            path=plan(grid,start,point);candidates.append(dict(base_xy_m=point.tolist(),reach_xy_m=distance,path_length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())))
        except ValueError:pass
report=dict(skill='pick',target='pick_box',target_position_m=obj.tolist(),reachable_approach_candidates=candidates,
    gripper_available=False,status='rejected',reasons=['Loaded g1_29dof asset has rubber hands and no finger actuators.']+([] if candidates else ['No base approach was found within the assumed 0.60 m horizontal reach envelope at 0.32 m collision radius.']),
    assumptions=['Reach envelope is an estimated precondition, not a robot IK/reachability proof.','Layout is provisional.'],next_step='Use g1_29dof_dex3 with independently mapped finger control, collision-aware arm targets, and contact-based grasp evaluation.')
Path('reports/pick_preconditions.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
