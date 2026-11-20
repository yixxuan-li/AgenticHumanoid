"""Load exported USD in Isaac Lab, step actual PhysX, and check free objects.

Run with an installed Isaac Lab interpreter, e.g.:
  python scripts/isaaclab_smoke.py --headless --scene outputs/scene/scene.usda
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--scene',type=Path,default=Path('outputs/scene/scene.usda'))
parser.add_argument('--report',type=Path,default=Path('reports/isaaclab_validation.json'))
parser.add_argument('--steps',type=int,default=300)
parser.add_argument('--screenshot',action='store_true')
parser.add_argument('--camera',help='Source camera name from scene.xml, e.g. source_016; screenshot uses its pose and vertical FOV')
parser.add_argument('--fast-exit',action='store_true',help='Exit after flushing artifacts; avoids a Kit 5.1 shutdown hang on this runtime')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args)
app=launcher.app

import numpy as np
import omni.usd
from pxr import Gf,Usd,UsdGeom,UsdPhysics
import isaaclab.sim as sim_utils
from isaacsim.core.utils.prims import create_prim
from scene_reconstruction.provenance import scene_fingerprint


def main():
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=args.device))
    create_prim('/World/Reconstruction',usd_path=str(args.scene.resolve()))
    stage=omni.usd.get_context().get_stage()
    # A composed scene must have one active PhysicsScene, owned by SimulationContext.
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()).startswith('/World/Reconstruction'):
            prim.SetActive(False)
    sim_utils.DomeLightCfg(intensity=750.).func('/World/Light',sim_utils.DomeLightCfg(intensity=750.))
    sim.set_camera_view([3.2,-1.3,3.2],[3.2,2.6,.8])
    # RigidObject views report PhysX state directly; USD transforms are not guaranteed
    # to update when Fabric/GPU simulation is enabled.
    from isaaclab.assets import RigidObject,RigidObjectCfg
    manifest=json.loads((args.scene.parent/'manifest.json').read_text());dynamic={b['name']:b for b in manifest['bodies'] if b['kind']!='fixed'}
    objects={name:RigidObject(RigidObjectCfg(prim_path=f'/World/Reconstruction/Scene/{name}',spawn=None)) for name in dynamic}
    camera=None
    if args.screenshot:
        from isaaclab.sensors import Camera,CameraCfg
        width,height,focal=1280,960,20.
        if args.camera:
            import xml.etree.ElementTree as ET,math
            source=ET.parse(args.scene.with_suffix('.xml')).getroot().find(f"worldbody/camera[@name='{args.camera}']")
            if source is None:raise ValueError('Review camera not found in matching MJCF')
            camera_pos=np.fromstring(source.get('pos'),sep=' ');axes=np.fromstring(source.get('xyaxes'),sep=' ');camera_target=camera_pos-np.cross(axes[:3],axes[3:])
            width,height=480,852;focal=36.*height/width/(2*math.tan(math.radians(float(source.get('fovy')))/2))
        camera=Camera(CameraCfg(prim_path='/World/ReviewCamera',update_period=0.,height=height,width=width,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=focal,horizontal_aperture=36.,clipping_range=(.05,100.))))
        for name in ['wall_entry_left','wall_entry_right','entry_lintel','ceiling_visual']:
            prim=stage.GetPrimAtPath('/World/Reconstruction/Scene/'+name)
            if prim.IsValid():UsdGeom.Imageable(prim).MakeInvisible()
        if not args.camera:
            for path in ['Scene/wall_front_observed','Scene/wall_entry_front','PhotoBackground/photo_cutaway_ceiling','PhotoBackground/photo_cutaway_front']:
                prim=stage.GetPrimAtPath('/World/Reconstruction/'+path)
                if prim.IsValid():UsdGeom.Imageable(prim).MakeInvisible()
    sim.reset()
    if camera is not None:
        import torch
        eye=camera_pos if args.camera else [3.2,-2.,5.8];target=camera_target if args.camera else [3.2,2.5,.7]
        if args.camera:
            from scipy.spatial.transform import Rotation
            orientation=Rotation.from_matrix(np.column_stack([axes[:3],axes[3:],np.cross(axes[:3],axes[3:])])).as_quat(scalar_first=True)
            camera.set_world_poses(torch.tensor([list(eye)],device=sim.device,dtype=torch.float32),torch.tensor([orientation.tolist()],device=sim.device,dtype=torch.float32),convention='opengl')
        else:camera.set_world_poses_from_view(torch.tensor([list(eye)],device=sim.device,dtype=torch.float32),torch.tensor([list(target)],device=sim.device,dtype=torch.float32))
    for _ in range(args.steps):
        sim.step()
        for obj in objects.values():obj.update(sim.get_physics_dt())
        if camera is not None:camera.update(sim.get_physics_dt())
    states={name:obj.data.root_pos_w[0].detach().cpu().tolist() for name,obj in objects.items()}
    drift={name:float(np.linalg.norm(np.array(states[name])-b['pos'])) for name,b in dynamic.items() if b['kind']=='free'}
    masses={name:float(obj.root_physx_view.get_masses().reshape(-1)[0]) for name,obj in objects.items()}
    block=next(b for b in manifest['bodies'] if b['name']=='test_block')
    expected=manifest['desk_height']+block['parts'][0]['size'][2]/2
    error=abs(states['test_block'][2]-expected)
    report=dict(runtime='Isaac Lab / PhysX',scene=str(args.scene),steps=args.steps,
        scene_fingerprint=scene_fingerprint(args.scene.parent)['sha256'],
        dt_seconds=sim.get_physics_dt(),positions_m=states,table_support_error_m=error,
        checks=dict(finite_states=bool(np.isfinite(list(states.values())).all()),table_support=error<.015,free_body_drift_under_6cm=all(v<.06 for v in drift.values()),mass_consistency=all(abs(masses[n]-b['mass'])<1e-5 for n,b in dynamic.items())),free_body_drift_m=drift,masses_kg=masses,
        limitations=['Scene scale and physical parameters remain estimates.','No humanoid policy was run by this smoke test.'])
    if 'cabinet_door' in objects:
        import torch
        door=objects['cabinet_door'];spec=dynamic['cabinet_door'];q=door.data.root_quat_w[0].cpu().numpy();rest=float((2*np.arctan2(q[3],q[0])-spec['yaw']+np.pi)%(2*np.pi)-np.pi)
        force=torch.zeros((1,1,3),device=sim.device);torque=force.clone();torque[0,0,2]=-3.
        for _ in range(600):
            door.set_external_force_and_torque(force,torque);door.write_data_to_sim();sim.step();door.update(sim.get_physics_dt())
            if camera is not None:camera.update(sim.get_physics_dt())
        q=door.data.root_quat_w[0].cpu().numpy();angle=float((2*np.arctan2(q[3],q[0])-spec['yaw']+np.pi)%(2*np.pi)-np.pi);position_error=float(np.linalg.norm(door.data.root_pos_w[0].cpu().numpy()-spec['pos']));limits=spec['joint']['range']
        door.set_external_force_and_torque(force,force);door.write_data_to_sim()
        report['cabinet_hinge']=dict(rest_angle_relative_rad=rest,torque_nm=-3.,probe_steps=600,final_angle_relative_rad=angle,anchor_position_error_m=position_error,limits_rad=limits)
        report['checks'].update(hinge_rest_frame=abs(rest)<.05,hinge_moves_under_torque=angle<-1.2,hinge_limits=limits[0]-.03<=angle<=limits[1]+.03,hinge_anchor_stays_fixed=position_error<.01)
    args.report.write_text(json.dumps(report,indent=2))
    print('ISAACLAB_VALIDATION '+json.dumps(report),flush=True)
    if not all(report['checks'].values()):raise RuntimeError('PhysX validation failed')
    if args.screenshot:
        from PIL import Image
        rgb=camera.data.output['rgb'][0].detach().cpu().numpy()
        if rgb.size==0:raise RuntimeError('RTX screenshot returned no image')
        screenshot=args.scene.parent/f'isaaclab_{args.camera or "overview"}.png';Image.fromarray(rgb).save(screenshot)
        print('RTX_SCREENSHOT '+str(screenshot),flush=True)


try:main()
except BaseException:
    import traceback,os,sys
    traceback.print_exc()
    sys.stdout.flush();sys.stderr.flush()
    if args.fast_exit:os._exit(1)
    app.close(wait_for_replicator=False)
    raise
else:
    if args.fast_exit:
        import os,sys
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    app.close(wait_for_replicator=False)
