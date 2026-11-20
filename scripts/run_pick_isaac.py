"""Native PhysX Dex3 grasp driven by the released BFM and finger drives.

No object weld, root pose forcing, or object force is used during execution.
This is a simulator-state scripted skill experiment, not visual manipulation.
"""
import argparse,ast,json,math,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/final_scene/scene.usda'));p.add_argument('--robot-usd',type=Path,default=Path('outputs/robots/g1_dex3_v4/g1_dex3.usd'));p.add_argument('--run-name',default='trial');p.add_argument('--seconds',type=float,default=20.);p.add_argument('--navigate-first',action='store_true');p.add_argument('--navigation-speed',type=float,default=.12);p.add_argument('--plan',type=Path,default=Path('reports/skill_agent_final_plan.json'));p.add_argument('--video',action='store_true');p.add_argument('--fast-exit',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);app=launcher.app
import numpy as np,torch
from scipy.spatial.transform import Rotation,Slerp
import omni.usd
from pxr import UsdPhysics,UsdGeom
from isaacsim.core.utils.prims import create_prim
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,RigidObject,RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from scene_reconstruction.scalebfm import ScaleBFM
from scene_reconstruction.provenance import scene_fingerprint

def main():
 torch.set_num_threads(2);out=Path('outputs/controller/isaac_pick')/a.run_name;out.mkdir(parents=True,exist_ok=True);source=Path('vendor/ScaleBFM');manifest=json.loads((a.scene.parent/'manifest.json').read_text());bench=manifest['benchmarks']['pick'];profile=bench['controller_profile'];approach=bench['approach'];target_body=bench['target'];objectpath='/World/Reconstruction/Scene/'+target_body
 robotxml=source/'ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml';policy=ScaleBFM(source,Path('checkpoints/scalebfm_m.pt'),Path('checkpoints/scalebfm_m_metadata.json'),robotxml,out,device=a.device);meta=policy.metadata
 sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=a.device));create_prim('/World/Reconstruction',usd_path=str(a.scene.resolve()));stage=omni.usd.get_context().get_stage()
 for prim in list(stage.Traverse()):
  if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()).startswith('/World/Reconstruction'):prim.SetActive(False)
 tree=ast.parse((source/'ScaleTrack/source/scaletrack/scaletrack/robots/g1_29dof.py').read_text());tree.body=[node for node in tree.body if not(isinstance(node,ast.ImportFrom) and node.module=='scaletrack.assets')];namespace=dict(ASSET_DIR=str(robotxml.parent.parent.parent.resolve()));exec(compile(tree,'official_g1_cfg','exec'),namespace);cfg=namespace['G1_29DOF_CYLINDER_CFG'].replace(prim_path='/World/G1');cfg.spawn.usd_path=str(a.robot_usd.resolve());cfg.init_state.pos=(*approach[:2],profile['robot_root_height_m']);cfg.init_state.rot=Rotation.from_euler('z',bench['yaw_rad']).as_quat(scalar_first=True).tolist();cfg.init_state.joint_pos['.*_hand_.*_joint']=0.
 cfg.actuators['hands']=ImplicitActuatorCfg(joint_names_expr=['.*_hand_.*_joint'],effort_limit_sim={'.*_hand_thumb_0_joint':2.45,'.*_hand_(thumb_[12]|middle_[01]|index_[01])_joint':.7},velocity_limit_sim=10.,stiffness=6.,damping=.20,armature=.001)
 if a.navigate_first:cfg.init_state.pos=(*manifest['navigation']['start'],.76);cfg.init_state.rot=(1.,0.,0.,0.)
 robot=Articulation(cfg);obj=RigidObject(RigidObjectCfg(prim_path=objectpath,spawn=None));hand_links=['right_hand_thumb_0_link','right_hand_thumb_1_link','right_hand_thumb_2_link','right_hand_middle_0_link','right_hand_middle_1_link','right_hand_index_0_link','right_hand_index_1_link']
 sensors={name:ContactSensor(ContactSensorCfg(prim_path='/World/G1/pelvis/'+name,filter_prim_paths_expr=[objectpath],update_period=0.)) for name in hand_links}
 camera=None;writer=None
 if a.video:
  from isaaclab.sensors import Camera,CameraCfg
  import imageio.v2 as imageio
  sim_utils.DomeLightCfg(intensity=750.).func('/World/Light',sim_utils.DomeLightCfg(intensity=750.))
  camera=Camera(CameraCfg(prim_path='/World/Camera',height=480,width=640,data_types=['rgb'],update_period=0.,spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=36.,clipping_range=(.03,100.))))
  writer=imageio.get_writer(out/'pick.mp4',fps=25,macro_block_size=1)
  for path in ['Scene/ceiling_visual','Scene/wall_front_observed','Scene/wall_entry_front','PhotoBackground/photo_cutaway_ceiling','PhotoBackground/photo_cutaway_front']:
   prim=stage.GetPrimAtPath('/World/Reconstruction/'+path)
   if prim.IsValid():UsdGeom.Imageable(prim).MakeInvisible()
 sim.reset();robot.write_root_state_to_sim(robot.data.default_root_state);robot.write_joint_state_to_sim(robot.data.default_joint_pos,robot.data.default_joint_vel);robot.reset();sim.step(render=False);robot.update(.005);obj.update(.005)
 camera_close=not a.navigate_first
 if camera is not None:
  eye=[5.1,1.6,1.45] if camera_close else [3.15,-2.,5.8];look=[bench['block'][0],bench['block'][1],.9] if camera_close else [3.15,2.9,.8]
  camera.set_world_poses_from_view(torch.tensor([eye],device=sim.device),torch.tensor([look],device=sim.device))
 joints=[robot.joint_names.index(n) for n in meta['joint_names']];bodies=[robot.body_names.index(n) for n in meta['selected_body_names']];wrist_index=meta['selected_body_names'].index('right_wrist_yaw_link');wrist_body=robot.body_names.index('right_wrist_yaw_link');initial_pos=robot.data.body_pos_w[0,bodies].cpu().numpy().copy();initial_quat=robot.data.body_quat_w[0,bodies].cpu().numpy().copy();initial_root=robot.data.root_pos_w[0].cpu().numpy().copy();initial_prop=obj.data.root_pos_w[0].cpu().numpy().copy()
 desired_closed={'right_hand_thumb_0_joint':0.,'right_hand_thumb_1_joint':-.8,'right_hand_thumb_2_joint':-1.5,'right_hand_middle_0_joint':1.,'right_hand_middle_1_joint':1.2,'right_hand_index_0_joint':1.,'right_hand_index_1_joint':1.2};finger_indices={name:robot.joint_names.index(name) for name in desired_closed}
 rotation=Rotation.from_matrix(np.array([[0,0,-1],[1,0,0],[0,-1,0]]));slerp=Slerp([0,1],Rotation.from_quat([initial_quat[wrist_index],rotation.as_quat(scalar_first=True)],scalar_first=True));grasp=initial_prop+[profile['wrist_lateral_offset_m'],-profile['wrist_forward_offset_m'],profile['wrist_height_offset_m']];pregrasp=grasp+[0,0,.12]
 pick_start=0.;pick_initialized=not a.navigate_first;navigation_error=None;navigation_success=not a.navigate_first;path=None;nav_pos=initial_pos.copy();nav_quat=initial_quat.copy()
 if a.navigate_first:
  plan_report=json.loads(a.plan.read_text())
  if plan_report['scene_fingerprint']!=scene_fingerprint(a.scene.parent)['sha256']:raise ValueError('Re-run the agent dry-run for this exact scene before transferring its route')
  path=np.array(plan_report['steps'][0]['waypoints_m']);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))];navigation_end=2.+arc[-1]/a.navigation_speed+2.;pick_start=navigation_end+2.
 def reference(t):
  if a.navigate_first and (t<pick_start or not pick_initialized):
   distance=min(max(0,t-2)*a.navigation_speed,arc[-1]);xy=np.array([np.interp(distance,arc,path[:,j]) for j in range(2)]);ahead=min(distance+.2,arc[-1]);behind=max(0,distance-.1);tangent=np.array([np.interp(ahead,arc,path[:,j])-np.interp(behind,arc,path[:,j]) for j in range(2)]);yaw=math.atan2(tangent[1],tangent[0]) if np.linalg.norm(tangent)>1e-6 else bench['yaw_rad']
   if t>=2.+arc[-1]/a.navigation_speed:yaw=bench['yaw_rad']
   pos=nav_pos.copy();pos[:,:2]+=xy-initial_root[:2];pos[0,2]=.76+np.clip((t-navigation_end)/2,0,1)*(profile['robot_root_height_m']-.76);quat=(Rotation.from_euler('z',yaw)*Rotation.from_quat(nav_quat,scalar_first=True)).as_quat(scalar_first=True);return pos,quat,0.,0
  t-=pick_start
  pos=initial_pos.copy();quat=initial_quat.copy()
  if t<2:return pos,quat,0.,0
  blend=np.clip((t-2)/5.,0,1);pos[wrist_index]=(1-blend)*initial_pos[wrist_index]+blend*pregrasp;quat[wrist_index]=slerp([blend]).as_quat(scalar_first=True)[0]
  if t>9:pos[wrist_index]=pregrasp+np.clip((t-9)/3.,0,1)*(grasp-pregrasp)
  close=float(np.clip((t-12)/2.,0,1))
  if t>14:pos[wrist_index]=grasp+[0,0,.18*np.clip((t-14)/3.,0,1)]
  return pos,quat,close,2
 root_correction=np.zeros(3);wrist_correction=np.zeros(3);rows=[];contact_frames=0;lift_frames=0;contact_parts=set();fall=False;maximum=float(initial_prop[2]);max_torque=0.;trace=[]
 try:
  for step in range(round((a.seconds+pick_start)/.02)):
   t=step*.02
   if a.navigate_first and t>=pick_start and not pick_initialized:
    navigation_error=float(np.linalg.norm(robot.data.root_pos_w[0,:2].cpu().numpy()-path[-1]));navigation_success=navigation_error<.15
    if not navigation_success:break
    initial_pos=robot.data.body_pos_w[0,bodies].cpu().numpy().copy();initial_quat=robot.data.body_quat_w[0,bodies].cpu().numpy().copy();initial_pos[0]=[*approach[:2],profile['robot_root_height_m']];initial_quat[0]=Rotation.from_euler('z',bench['yaw_rad']).as_quat(scalar_first=True);slerp=Slerp([0,1],Rotation.from_quat([initial_quat[wrist_index],rotation.as_quat(scalar_first=True)],scalar_first=True));pick_initialized=True
   pos,quat,close,mode=reference(t);future=[reference(t+k*.02) for k in meta['future_idx']];fp=np.array([x[0] for x in future]);fq=np.array([x[1] for x in future]);rp=robot.data.root_pos_w[0].cpu().numpy();rq=robot.data.root_quat_w[0].cpu().numpy();rr=Rotation.from_quat(rq,scalar_first=True);wrist=robot.data.body_pos_w[0,wrist_body].cpu().numpy()
   if t>=pick_start:root_correction=np.clip(root_correction+.6*(np.array([*approach[:2],profile['settled_root_height_m']])-rp)*.02,-.1,.1);fp[:,0]+=root_correction
   if t-pick_start>7.25:wrist_correction=np.clip(wrist_correction+.8*(pos[wrist_index]-wrist)*.02,-.1,.1);fp[:,wrist_index]+=wrist_correction
   lp=rr.inv().apply((fp-rp).reshape(-1,3)).reshape(fp.shape);lq=(rr.inv()*Rotation.from_quat(fq.reshape(-1,4),scalar_first=True)).as_quat(scalar_first=True).reshape(fq.shape)
   target,_=policy.act(rq,robot.data.root_ang_vel_b[0].cpu().numpy(),robot.data.joint_pos[0,joints].cpu().numpy(),robot.data.joint_vel[0,joints].cpu().numpy(),lp,lq,mode);positions=robot.data.default_joint_pos.clone();positions[0,joints]=torch.tensor(target,device=sim.device)
   for name,index in finger_indices.items():positions[0,index]=desired_closed[name]*close
   touching=set();normal_force=0.
   for _ in range(4):
    robot.set_joint_position_target(positions);robot.write_data_to_sim();sim.step(render=False);robot.update(.005);obj.update(.005)
    for name,sensor in sensors.items():
     sensor.update(.005);forces=sensor.data.force_matrix_w
     if forces is not None:
      force=float(torch.linalg.vector_norm(forces,dim=-1).sum());normal_force=max(normal_force,force)
      if force>.05:touching.add(name)
   prop=obj.data.root_pos_w[0].cpu().numpy().copy();actual=robot.data.body_pos_w[0,wrist_body].cpu().numpy();rootpos=robot.data.root_pos_w[0].cpu().numpy();height=float(prop[2]);maximum=max(maximum,height);contact_frames+=bool(touching);contact_parts.update(touching)
   if height>initial_prop[2]+.12 and touching:lift_frames+=1
   up=Rotation.from_quat(robot.data.root_quat_w[0].cpu().numpy(),scalar_first=True).as_matrix()[2,2];error=float(np.linalg.norm(actual-pos[wrist_index]));rows.append(np.r_[(step+1)*.02,rootpos,actual,pos[wrist_index],prop,error,len(touching),close,normal_force]);trace.append(np.r_[t,root_correction,wrist_correction]);max_torque=max(max_torque,float(robot.data.applied_torque.abs().max()))
   if camera is not None and step%2==0:
    if t>=pick_start and not camera_close:
     camera.set_world_poses_from_view(torch.tensor([[5.1,1.6,1.45]],device=sim.device),torch.tensor([[bench['block'][0],bench['block'][1],.9]],device=sim.device));camera_close=True
    sim.render();camera.update(.04);rgb=camera.data.output['rgb'][0].cpu().numpy();writer.append_data(rgb)
    if step==0 or step==round((a.seconds+pick_start)/.02)-2:
     from PIL import Image
     Image.fromarray(rgb).save(out/('initial.png' if step==0 else 'final.png'))
   if rootpos[2]<.2 or up<.25 or not np.isfinite(rootpos).all():fall=True;break
 finally:
  if writer is not None:writer.close()
 rows=np.array(rows);np.savetxt(out/'trajectory.csv',rows,delimiter=',',header='time,root_x,root_y,root_z,wrist_x,wrist_y,wrist_z,target_x,target_y,target_z,prop_x,prop_y,prop_z,wrist_error_m,contact_links,close,max_filtered_contact_force_n',comments='');np.savetxt(out/'servo.csv',trace,delimiter=',')
 report=dict(runtime='Isaac Lab / PhysX',task='Dex3 navigation and contact grasp' if a.navigate_first else 'Dex3 contact grasp',success=bool(navigation_success and not fall and lift_frames>=50 and rows[-1,12]>initial_prop[2]+.12),fall_detected=fall,simulated_seconds=float(rows[-1,0]),max_prop_height_m=maximum,lifted_contact_seconds=lift_frames*.02,hand_contact_seconds=contact_frames*.02,contact_parts=sorted(contact_parts),final_prop_position_m=rows[-1,10:13].tolist(),initial_prop_position_m=initial_prop.tolist(),initial_root_m=initial_root.tolist(),final_wrist_error_m=float(rows[-1,13]),max_applied_torque_nm=max_torque,scene_fingerprint=scene_fingerprint(a.scene.parent)['sha256'],robot_asset=str(a.robot_usd),object_weld_or_external_force=False,root_teleport_during_execution=False,body_controller='Released ScaleBFM medium; official G1 motor configuration',hand_controller='Dex3 implicit finger drives: 6 stiffness, 0.2 damping, 0.7 Nm / thumb-base 2.45 Nm limits',state_access='Simulator body/wrist/object state; scripted references and task-space integral correction',limitations=['Scene scale and material/mass parameters are provisional.','Simulator-state skill experiment; no learned visual perception policy.'])
 if a.navigate_first:report.update(navigation_success=navigation_success,navigation_goal_error_m=navigation_error,pick_start_seconds=pick_start,planned_path_m=path.tolist())
 Path(f'reports/isaac_dex3_pick_{a.run_name}.json').write_text(json.dumps(report,indent=2));print('ISAAC_DEX3_PICK '+json.dumps(report),flush=True)
try:main()
except BaseException:
 import traceback
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
 if a.fast_exit:os._exit(1)
 app.close(wait_for_replicator=False);raise
else:
 if a.fast_exit:sys.stdout.flush();sys.stderr.flush();os._exit(0)
 app.close(wait_for_replicator=False)
