"""Exploratory Dex3 contact-grasp benchmark driven by the released BFM.

The added red benchmark block is reset near an accessible table edge. The video-
observed prop is not moved. No weld, object force, or kinematic root motion is used.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('EGL_PLATFORM','surfaceless')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation,Slerp
from scene_reconstruction.scalebfm import ScaleBFM,RobotState,build_robot_scene
from scene_reconstruction.provenance import scene_fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--video',action='store_true');p.add_argument('--seconds',type=float,default=15.)
    p.add_argument('--root-y',type=float,default=1.58);p.add_argument('--wrist-forward-offset',type=float,default=.16);p.add_argument('--wrist-height-offset',type=float,default=.035)
    p.add_argument('--wrist-lateral-offset',type=float,default=0.)
    p.add_argument('--root-x',type=float,default=3.95);p.add_argument('--root-height',type=float,default=.76)
    p.add_argument('--block-x',type=float,default=3.95);p.add_argument('--block-y',type=float,default=2.03);p.add_argument('--run-name',default='trial')
    p.add_argument('--scene',type=Path,default=Path('outputs/worldlabs_scene/scene.xml'))
    p.add_argument('--navigate-first',action='store_true');p.add_argument('--navigation-speed',type=float,default=.12)
    p.add_argument('--slow-grasp',action='store_true')
    p.add_argument('--pose-servo',action='store_true',help='Integral task-space correction through BFM references, using simulator body state')
    p.add_argument('--settled-root-height',type=float,default=.78)
    args=p.parse_args();torch.set_num_threads(2);source=Path('vendor/ScaleBFM');out=Path('outputs/controller/pick')/args.run_name;out.mkdir(parents=True,exist_ok=True)
    robotdir=source/'ScaleBridge/scalebridge/data/robot/g1_29dof';policy=ScaleBFM(source,Path('checkpoints/scalebfm_m.pt'),Path('checkpoints/scalebfm_m_metadata.json'),robotdir/'g1_29dof.xml',out,device=args.device)
    m=build_robot_scene(robotdir/'g1_29dof_dex3.xml',args.scene,out/'g1_dex3_scene.xml');d=mujoco.MjData(m);state=RobotState(m,policy.metadata)
    rootqa=state.root_qadr;rootva=state.root_vadr;heading=Rotation.from_euler('z',math.pi/2)
    manifest=json.loads((args.scene.parent/'manifest.json').read_text())
    startroot=[*manifest['navigation']['start'],.76] if args.navigate_first else [args.root_x,args.root_y,args.root_height]
    startquat=[1,0,0,0] if args.navigate_first else heading.as_quat(scalar_first=True)
    d.qpos[rootqa:rootqa+3]=startroot;d.qpos[rootqa+3:rootqa+7]=startquat;d.qpos[state.qadr]=policy.metadata['default_dof_pos']
    blockjoint=m.joint('env_test_block_free').id;blockqa=m.jnt_qposadr[blockjoint];blockid=m.body('env_test_block').id
    d.qpos[blockqa:blockqa+3]=[args.block_x,args.block_y,.766];d.qpos[blockqa+3:blockqa+7]=[1,0,0,0]
    # Explicit finger indexing; adding hand joints does not shift the BFM channels.
    fingerjoints=[i for i in range(m.njnt) if (m.joint(i).name or '').startswith(('left_hand_','right_hand_'))]
    fingers={m.joint(j).name:j for j in fingerjoints};actuators={int(m.actuator_trnid[i,0]):i for i in range(m.nu)}
    desired_closed={'right_hand_thumb_0_joint':0.,'right_hand_thumb_1_joint':-.8,'right_hand_thumb_2_joint':-1.5,'right_hand_middle_0_joint':1.,'right_hand_middle_1_joint':1.2,'right_hand_index_0_joint':1.,'right_hand_index_1_joint':1.2}
    mujoco.mj_forward(m,d);initial_pos=d.xpos[state.selected_bodies].copy();initial_quat=d.xquat[state.selected_bodies].copy();initial_root=d.qpos[rootqa:rootqa+3].copy();initial_prop=d.xpos[blockid].copy()
    wristidx=policy.metadata['selected_body_names'].index('right_wrist_yaw_link');wristid=m.body('right_wrist_yaw_link').id
    # Right palm faces down, with the fingers extending toward +Y.
    grasp_rotation=Rotation.from_matrix(np.array([[0,0,-1],[1,0,0],[0,-1,0]]))
    slerp=Slerp([0,1],Rotation.from_quat(np.array([initial_quat[wristidx],grasp_rotation.as_quat(scalar_first=True)]),scalar_first=True))
    grasp_wrist=initial_prop+np.array([args.wrist_lateral_offset,-args.wrist_forward_offset,args.wrist_height_offset]);pregrasp=grasp_wrist+[0,0,.12]
    path=None;pick_start=0.;navigation_error=None;navigation_success=not args.navigate_first;pick_initialized=not args.navigate_first
    nav_initial_pos=initial_pos.copy();nav_initial_quat=initial_quat.copy()
    if args.navigate_first:
        from scene_reconstruction.navigation import collision_grid,plan
        sm=mujoco.MjModel.from_xml_path(str(args.scene));sd=mujoco.MjData(sm)
        for _ in range(2000):mujoco.mj_step(sm,sd)
        grid=collision_grid(sm,sd,manifest['dimensions'],radius=.32)
        path=plan(grid,initial_root[:2],[args.root_x,args.root_y]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        navigation_end=2.+arc[-1]/args.navigation_speed+2.;pick_start=navigation_end+2.
    rows=[];frames=[];states=[];servo_rows=[];root_correction=np.zeros(3);wrist_correction=np.zeros(3);renderer=None;fall=False;max_prop_height=float(initial_prop[2]);contact_frames=0;lift_contact_frames=0;contact_parts=set();force=np.zeros(6);environment_contacts={'navigation':{},'manipulation':{}}
    if args.video:
        for name in ['env_wall_entry_left','env_wall_entry_right','env_entry_lintel','env_ceiling_visual','env_wall_front_observed','env_wall_entry_front','env_photo_cutaway_ceiling','env_photo_cutaway_front']:
            bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
            if bid<0:continue
            a=m.body_geomadr[bid];m.geom_rgba[a:a+m.body_geomnum[bid],3]=0
        renderer=mujoco.Renderer(m,height=600,width=800)
    def targets(t):
        if args.navigate_first and (t<pick_start or not pick_initialized):
            distance=min(max(0,t-2)*args.navigation_speed,arc[-1]);xy=np.array([np.interp(distance,arc,path[:,j]) for j in range(2)])
            ahead=min(distance+.20,arc[-1]);behind=max(0,distance-.10)
            tangent=np.array([np.interp(ahead,arc,path[:,j])-np.interp(behind,arc,path[:,j]) for j in range(2)])
            yaw=math.atan2(tangent[1],tangent[0]) if np.linalg.norm(tangent)>1e-6 else math.pi/2
            if t>=2.+arc[-1]/args.navigation_speed:yaw=math.pi/2
            pos=nav_initial_pos.copy();pos[:,:2]+=xy-initial_root[:2]
            pos[0,2]=.76+np.clip((t-navigation_end)/2.,0,1)*(args.root_height-.76)
            quat=(Rotation.from_euler('z',yaw)*Rotation.from_quat(nav_initial_quat,scalar_first=True)).as_quat(scalar_first=True)
            return pos,quat,0.,0
        t-=pick_start
        pos=initial_pos.copy();quat=initial_quat.copy()
        if t<2:return pos,quat,0.,0
        reach_end,hold_end,descend_end,close_end=(7.,9.,12.,14.) if args.slow_grasp else (5.,5.,7.,9.)
        blend=np.clip((t-2)/(reach_end-2),0,1);pos[wristidx]=(1-blend)*initial_pos[wristidx]+blend*pregrasp
        quat[wristidx]=slerp([blend]).as_quat(scalar_first=True)[0]
        if t>hold_end:pos[wristidx]=pregrasp+np.clip((t-hold_end)/(descend_end-hold_end),0,1)*(grasp_wrist-pregrasp)
        close=float(np.clip((t-descend_end)/(close_end-descend_end),0,1))
        if t>close_end:pos[wristidx]=grasp_wrist+[0,0,.18*np.clip((t-close_end)/3,0,1)]
        return pos,quat,close,2
    try:
        for step in range(int((args.seconds+pick_start)/.02)):
            t=step*.02
            if args.navigate_first and t>=pick_start and not pick_initialized:
                navigation_error=float(np.linalg.norm(d.qpos[rootqa:rootqa+2]-path[-1]));navigation_success=navigation_error<.15
                if not navigation_success:break
                initial_pos=d.xpos[state.selected_bodies].copy();initial_quat=d.xquat[state.selected_bodies].copy()
                initial_pos[0]=[*path[-1],args.root_height];initial_quat[0]=heading.as_quat(scalar_first=True)
                slerp=Slerp([0,1],Rotation.from_quat(np.array([initial_quat[wristidx],grasp_rotation.as_quat(scalar_first=True)]),scalar_first=True))
                pick_initialized=True
            pos,quat,close,mode=targets(t)
            rp=d.qpos[rootqa:rootqa+3].copy();rr=Rotation.from_quat(d.qpos[rootqa+3:rootqa+7],scalar_first=True)
            future=[targets(t+k*.02) for k in policy.metadata['future_idx']]
            futurepos=np.array([x[0] for x in future]);futurequat=np.array([x[1] for x in future])
            if args.pose_servo and t>=pick_start:
                root_error=np.array([args.root_x,args.root_y,args.settled_root_height])-rp
                root_correction=np.clip(root_correction+.6*root_error*.02,-.10,.10)
                futurepos[:,0]+=root_correction
                reach_end=7. if args.slow_grasp else 5.
                if t-pick_start>reach_end+.25:
                    wrist_error=pos[wristidx]-d.xpos[wristid]
                    wrist_correction=np.clip(wrist_correction+.8*wrist_error*.02,-.10,.10)
                    futurepos[:,wristidx]+=wrist_correction
                servo_rows.append(np.r_[t,root_correction,wrist_correction])
            localpos=rr.inv().apply((futurepos-rp).reshape(-1,3)).reshape(futurepos.shape);localquat=(rr.inv()*Rotation.from_quat(futurequat.reshape(-1,4),scalar_first=True)).as_quat(scalar_first=True).reshape(futurequat.shape)
            target,_=policy.act(d.qpos[rootqa+3:rootqa+7],d.qvel[rootva+3:rootva+6],d.qpos[state.qadr],d.qvel[state.vadr],localpos,localquat,mode)
            current_parts=set()
            for _ in range(4):
                state.apply_pd(d,target)
                for name,j in fingers.items():
                    a=actuators[j];q=m.jnt_qposadr[j];v=m.jnt_dofadr[j];goal=desired_closed.get(name,0)*close
                    torque=6*(goal-d.qpos[q])-.15*d.qvel[v];lo,hi=m.actuator_ctrlrange[a];d.ctrl[a]=np.clip(torque,lo,hi)
                mujoco.mj_step(m,d)
                for ci,c in enumerate(d.contact):
                    bodies=[m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]]
                    body_names=[m.body(b).name or '' for b in bodies]
                    env=[name for name in body_names if name.startswith('env_')]
                    if len(env)==1 and not env[0].startswith('env_floor') and env[0]!='env_test_block':
                        mujoco.mj_contactForce(m,d,ci,force)
                        if force[0]>1.:
                            phase='navigation' if t<pick_start else 'manipulation';key=' / '.join(body_names)
                            record=environment_contacts[phase].setdefault(key,dict(peak_force_n=0.,normal_impulse_ns=0.))
                            record['peak_force_n']=max(record['peak_force_n'],float(force[0]));record['normal_impulse_ns']+=float(force[0]*m.opt.timestep)
                    if blockid not in bodies:continue
                    other=bodies[1] if bodies[0]==blockid else bodies[0];name=m.body(other).name or ''
                    if not name.startswith('right_hand_'):continue
                    mujoco.mj_contactForce(m,d,ci,force)
                    if force[0]>.05:current_parts.add(name)
            prop=d.xpos[blockid].copy();height=float(prop[2]);max_prop_height=max(max_prop_height,height)
            if current_parts:contact_frames+=1;contact_parts.update(current_parts)
            if height>.88 and current_parts:lift_contact_frames+=1
            up=float(d.xmat[m.body('pelvis').id].reshape(3,3)[2,2]);error=float(np.linalg.norm(d.xpos[wristid]-pos[wristidx]))
            rows.append(np.r_[d.time,d.qpos[rootqa:rootqa+3],d.xpos[wristid],pos[wristidx],prop,error,len(current_parts),close])
            states.append(d.qpos.copy())
            if renderer is not None:
                camera=mujoco.MjvCamera()
                if t<pick_start:camera.lookat=[3.2,2.2,.65];camera.distance=6.2;camera.azimuth=95;camera.elevation=-52
                else:camera.lookat=[args.block_x,args.block_y-.08,.80];camera.distance=1.55;camera.azimuth=-30;camera.elevation=-25
                renderer.update_scene(d,camera=camera);frames.append(renderer.render().copy())
            if d.qpos[rootqa+2]<.20 or up<.25 or not np.isfinite(d.qpos).all():fall=True;break
    finally:
        if renderer is not None:renderer.close()
    rows=np.array(rows);np.savetxt(out/'trajectory.csv',rows,delimiter=',',header='time,root_x,root_y,root_z,wrist_x,wrist_y,wrist_z,target_x,target_y,target_z,prop_x,prop_y,prop_z,wrist_error_m,hand_contact_parts,finger_close',comments='')
    np.savez_compressed(out/'states.npz',qpos=np.array(states),time=rows[:,0])
    if servo_rows:np.savetxt(out/'servo_corrections.csv',servo_rows,delimiter=',',header='time,root_dx,root_dy,root_dz,wrist_dx,wrist_dy,wrist_dz',comments='')
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(out/'dex3_pick_trial.mp4',frames,fps=50,macro_block_size=1)
    report=dict(task='Dex3 pick benchmark',success=bool(navigation_success and not fall and lift_contact_frames>=50 and rows[-1,12]>.88),fall_detected=fall,
        simulated_seconds=float(d.time),max_prop_height_m=max_prop_height,final_prop_position_m=rows[-1,10:13].tolist(),hand_contact_seconds=contact_frames*.02,lifted_contact_seconds=lift_contact_frames*.02,contact_parts=sorted(contact_parts),
        mean_wrist_tracking_error_m=float(rows[:,13].mean()),final_wrist_tracking_error_m=float(rows[-1,13]),initial_robot_root_m=initial_root.tolist(),initial_benchmark_prop_m=initial_prop.tolist(),
        mean_manipulation_wrist_error_m=float(rows[rows[:,0]>=pick_start+2,13].mean()),environment_contacts_above_1n=environment_contacts,
        scene_fingerprint=scene_fingerprint(args.scene.parent)['sha256'],benchmark_variant='Only added red test_block is reset to an accessible table edge; initial robot pose is set near it. Video-observed objects remain in place.',
        object_weld_or_external_force=False,root_teleport_during_execution=False,controller='Released 29-DoF BFM controls body; separate torque-limited Dex3 finger PD',
        task_space_servo=bool(args.pose_servo),state_access='Simulator root/wrist/object state; no visual perception policy',
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},limitations=['Exploratory scripted grasp, not a trained autonomous manipulation agent.','Scene dimensions and object physical properties are provisional.'])
    if args.navigate_first:report.update(agent='A* base approach -> lower -> BFM hand reach -> finger close -> lift and hold',navigation_success=navigation_success,navigation_goal_error_m=navigation_error,pick_start_seconds=float(pick_start),planned_path_m=path.tolist())
    Path(f'reports/dex3_pick_{args.run_name}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
