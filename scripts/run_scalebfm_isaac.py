"""Execute the same released BFM weights with Isaac Lab / PhysX articulation state.

Run with Isaac Lab's Python interpreter; this is simulation only.
"""
import argparse
import ast
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--task',choices=['squat','navigate'],default='navigate')
parser.add_argument('--run-name',default='')
parser.add_argument('--robot-usd',type=Path,help='Optional converted G1/Dex3 asset for a separate body-control check')
parser.add_argument('--seconds',type=float,default=12.)
parser.add_argument('--scene',type=Path,default=Path('outputs/scene/scene.usda'))
parser.add_argument('--fast-exit',action='store_true')
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
launcher=AppLauncher(args);app=launcher.app

import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import omni.usd
from pxr import UsdPhysics
from isaacsim.core.utils.prims import create_prim
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from scene_reconstruction.scalebfm import ScaleBFM
from scene_reconstruction.provenance import scene_fingerprint


def main():
    torch.set_num_threads(2);source=Path('vendor/ScaleBFM');output=Path('outputs/controller')/args.run_name;output.mkdir(parents=True,exist_ok=True)
    robotxml=source/'ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml'
    robotusd=robotxml.with_suffix('.usda').resolve()
    policy=ScaleBFM(source,Path('checkpoints/scalebfm_m.pt'),Path('checkpoints/scalebfm_m_metadata.json'),robotxml,output/'isaac',device=args.device)
    manifest=json.loads((args.scene.parent/'manifest.json').read_text());start=manifest['navigation']['start']
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=args.device))
    create_prim('/World/Reconstruction',usd_path=str(args.scene.resolve()))
    stage=omni.usd.get_context().get_stage()
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()).startswith('/World/Reconstruction'):prim.SetActive(False)
    meta=policy.metadata
    configpath=source/'ScaleTrack/source/scaletrack/scaletrack/robots/g1_29dof.py'
    tree=ast.parse(configpath.read_text(),filename=str(configpath))
    # Resolve only the package's ASSET_DIR locally; retain the official motor
    # armatures, gains, limits, collision and damping settings verbatim.
    tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.module=='scaletrack.assets')]
    namespace=dict(ASSET_DIR=str(robotxml.parent.parent.parent.resolve()))
    exec(compile(tree,str(configpath),'exec'),namespace)
    cfg=namespace['G1_29DOF_CYLINDER_CFG'].replace(prim_path='/World/G1')
    cfg.spawn.usd_path=str(robotusd);cfg.init_state.pos=(*start,.76)
    if args.robot_usd:
        cfg.spawn.usd_path=str(args.robot_usd.resolve());cfg.init_state.joint_pos['.*_hand_.*_joint']=0.
        cfg.actuators['hands']=ImplicitActuatorCfg(joint_names_expr=['.*_hand_.*_joint'],effort_limit_sim={'.*_hand_thumb_0_joint':2.45,'.*_hand_(thumb_[12]|middle_[01]|index_[01])_joint':.7},velocity_limit_sim=10.,stiffness=6.,damping=.20,armature=.001)
    robot=Articulation(cfg);sim.reset()
    joint_idx=[robot.joint_names.index(n) for n in meta['joint_names']];body_idx=[robot.body_names.index(n) for n in meta['selected_body_names']]
    motion=np.load(source/'ScaleBridge/scalebridge/data/motion/g1_29dof/squat.npz')
    selected=[meta['body_names'].index(n) for n in meta['selected_body_names']]
    src_pos=motion['body_pos_w'][:,selected].astype(float);src_quat=motion['body_quat_w'][:,selected].astype(float)
    heading=Rotation.from_quat(src_quat[0,0],scalar_first=True).as_euler('xyz')[2];align=Rotation.from_euler('z',-heading)
    offset=src_pos[0,0].copy();offset[2]=0
    refs=align.apply((src_pos-offset).reshape(-1,3)).reshape(src_pos.shape)+np.r_[start,0.]
    rots=(align*Rotation.from_quat(src_quat.reshape(-1,4),scalar_first=True)).as_quat(scalar_first=True).reshape(src_quat.shape)
    if args.task=='squat':
        root=robot.data.default_root_state.clone();root[0,:3]=torch.tensor(refs[0,0],device=sim.device);root[0,3:7]=torch.tensor(rots[0,0],device=sim.device)
        root[0,7:10]=torch.tensor(align.apply(motion['body_lin_vel_w'][0,0]),device=sim.device)
        root[0,10:13]=torch.tensor(align.apply(motion['body_ang_vel_w'][0,0]),device=sim.device)
        q=robot.data.default_joint_pos.clone();qd=torch.zeros_like(q);q[:,joint_idx]=torch.tensor(motion['joint_pos'][0],device=sim.device);qd[:,joint_idx]=torch.tensor(motion['joint_vel'][0],device=sim.device)
        robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(q,qd)
    else:
        robot.write_root_state_to_sim(robot.data.default_root_state);robot.write_joint_state_to_sim(robot.data.default_joint_pos,robot.data.default_joint_vel)
    robot.reset();sim.step(render=False);robot.update(.005)
    if args.task=='navigate':
        refs=np.broadcast_to(robot.data.body_pos_w[0,body_idx].cpu().numpy(),(len(src_pos),len(body_idx),3)).copy()
        rots=np.broadcast_to(robot.data.body_quat_w[0,body_idx].cpu().numpy(),(len(src_pos),len(body_idx),4)).copy()
    initial=robot.data.root_pos_w[0].cpu().numpy().copy();rows=[];fall=False;max_torque=0.
    mode=7 if args.task=='squat' else 0
    for step in range(int(args.seconds/.02)):
        inds=np.minimum(step+np.array(meta['future_idx']),len(refs)-1);targetpos=refs[inds].copy();targetrot=rots[inds].copy()
        if args.task=='navigate':
            targetpos=refs[:6].copy();targetrot=rots[:6].copy();t=np.maximum(0,(step+np.array(meta['future_idx']))*.02-2.)
            targetpos[:,:,0]+=np.minimum(t*.12,1.10)[:,None]
        rootpos=robot.data.root_pos_w[0].cpu().numpy();rootquat=robot.data.root_quat_w[0].cpu().numpy();r=Rotation.from_quat(rootquat,scalar_first=True)
        localpos=r.inv().apply((targetpos-rootpos).reshape(-1,3)).reshape(targetpos.shape)
        localquat=(r.inv()*Rotation.from_quat(targetrot.reshape(-1,4),scalar_first=True)).as_quat(scalar_first=True).reshape(targetrot.shape)
        target,_=policy.act(rootquat,robot.data.root_ang_vel_b[0].cpu().numpy(),robot.data.joint_pos[0,joint_idx].cpu().numpy(),robot.data.joint_vel[0,joint_idx].cpu().numpy(),localpos,localquat,mode)
        target=torch.tensor(target,device=sim.device)
        for _ in range(4):
            positions=robot.data.default_joint_pos.clone();positions[0,joint_idx]=target
            robot.set_joint_position_target(positions);robot.write_data_to_sim();sim.step(render=False);robot.update(.005)
            max_torque=max(max_torque,float(robot.data.applied_torque.abs().max()))
        actual=robot.data.root_pos_w[0].cpu().numpy();quat=robot.data.root_quat_w[0].cpu().numpy();up=Rotation.from_quat(quat,scalar_first=True).as_matrix()[2,2]
        error=float(np.linalg.norm(actual-targetpos[0,0]));rows.append(np.r_[(step+1)*.02,actual,targetpos[0,0],up,error])
        if actual[2]<.12 or up<.25 or not np.isfinite(actual).all():fall=True;break
    rows=np.array(rows);np.savetxt(output/f'isaac_{args.task}_trajectory.csv',rows,delimiter=',',header='time,x,y,z,target_x,target_y,target_z,up_z,root_error_m',comments='')
    rmse=float(np.sqrt(np.mean(rows[:,-1]**2)));displacement=float(np.linalg.norm(rows[-1,1:3]-initial[:2]))
    report=dict(runtime='Isaac Lab 0.47.2 / Isaac Sim 5.1 / PhysX',task=args.task,mode=mode,simulated_seconds=float(rows[-1,0]),fall_detected=fall,
        success=bool(not fall and (rmse<.3 if args.task=='squat' else displacement>.3 and rows[-1,-1]<.3)),root_tracking_rmse_m=rmse,root_displacement_m=displacement,final_root_m=rows[-1,1:4].tolist(),max_applied_torque_nm=max_torque,
        root_teleport_during_execution=False,reference_forcing=False,scene_scale=manifest['scale_status'],
        scene_fingerprint=scene_fingerprint(args.scene.parent)['sha256'],
        motor_model='official G1_29DOF_CYLINDER_CFG implicit PD, armature, effort/velocity limits and damping; 50 Hz BFM / 200 Hz physics',
        robot_asset=str(args.robot_usd) if args.robot_usd else str(robotusd),
        limitations=['Scripted reference test, not learned visual exploration.','No pick/grasp success is claimed.'])
    Path(f'reports/{args.run_name+"_" if args.run_name else ""}scalebfm_isaac_{args.task}.json').write_text(json.dumps(report,indent=2));print('ISAAC_BFM_RESULT '+json.dumps(report),flush=True)


try:main()
except BaseException:
    import traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
    if args.fast_exit:os._exit(1)
    app.close(wait_for_replicator=False);raise
else:
    if args.fast_exit:sys.stdout.flush();sys.stderr.flush();os._exit(0)
    app.close(wait_for_replicator=False)
