"""Execute released ScaleBFM weights in actual MuJoCo scene dynamics.

Tasks: upstream squat tracking, root-controlled navigation, or hand reaching.
Success metrics use simulated state; references never move the simulated robot.
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
from scipy.spatial.transform import Rotation
from scene_reconstruction.scalebfm import ScaleBFM,RobotState,build_robot_scene
from scene_reconstruction.provenance import scene_fingerprint


def wxyz(r):return r.as_quat(scalar_first=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['squat','navigate','explore','reach'],default='squat')
    p.add_argument('--mode',type=int,default=None);p.add_argument('--device',default='cpu')
    p.add_argument('--seconds',type=float,default=8.0);p.add_argument('--video',action='store_true');p.add_argument('--speed',type=float,default=.12)
    p.add_argument('--run-name',default='');p.add_argument('--source',type=Path,default=Path('vendor/ScaleBFM'));p.add_argument('--scene',type=Path,default=Path('outputs/worldlabs_scene/scene.xml'))
    args=p.parse_args();torch.set_num_threads(2)
    output=Path('outputs/controller')/args.run_name;output.mkdir(parents=True,exist_ok=True)
    robot=args.source/'ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof.xml'
    manifest=json.loads((args.scene.parent/'manifest.json').read_text());scale=manifest.get('global_scale',1.)
    policy=ScaleBFM(args.source,Path('checkpoints/scalebfm_m.pt'),Path('checkpoints/scalebfm_m_metadata.json'),robot,output,device=args.device)
    m=build_robot_scene(robot,args.scene,output/'g1_scene.xml');d=mujoco.MjData(m);state=RobotState(m,policy.metadata)
    if args.video:
        # Hide the inferred entry closure only in review renders. It remains a
        # collision surface, and hiding it does not affect simulated dynamics.
        for name in ['env_wall_entry_left','env_wall_entry_right','env_entry_lintel','env_ceiling_visual','env_wall_front_observed','env_wall_entry_front','env_photo_cutaway_ceiling','env_photo_cutaway_front']:
            bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
            if bid<0:continue
            adr=m.body_geomadr[bid];count=m.body_geomnum[bid];m.geom_rgba[adr:adr+count,3]=0
    motion=np.load(args.source/'ScaleBridge/scalebridge/data/motion/g1_29dof/squat.npz')
    selected=[policy.metadata['body_names'].index(n) for n in policy.metadata['selected_body_names']]
    src_pos=motion['body_pos_w'][:,selected].astype(float);src_quat=motion['body_quat_w'][:,selected].astype(float)
    root0=Rotation.from_quat(src_quat[0,0],scalar_first=True);heading=root0.as_euler('xyz')[2]
    align=Rotation.from_euler('z',-heading)
    # Exploration starts in the observed open floor. The reach task uses the
    # same start and reports the distance, rather than teleporting near a prop.
    origin=np.r_[manifest['navigation']['start'],0.];offset=src_pos[0,0].copy();offset[2]=0.
    refs=align.apply((src_pos-offset).reshape(-1,3)).reshape(src_pos.shape)+origin
    rots=wxyz(align*Rotation.from_quat(src_quat.reshape(-1,4),scalar_first=True)).reshape(src_quat.shape)
    qa=state.root_qadr;va=state.root_vadr
    if args.task=='squat':
        d.qpos[qa:qa+3]=refs[0,0];d.qpos[qa+3:qa+7]=rots[0,0];d.qpos[state.qadr]=motion['joint_pos'][0];d.qvel[state.vadr]=motion['joint_vel'][0]
        d.qvel[va:va+3]=align.apply(motion['body_lin_vel_w'][0,0])
        d.qvel[va+3:va+6]=Rotation.from_quat(rots[0,0],scalar_first=True).inv().apply(align.apply(motion['body_ang_vel_w'][0,0]))
    else:
        d.qpos[qa:qa+3]=[*origin[:2],.76];d.qpos[qa+3:qa+7]=[1,0,0,0];d.qpos[state.qadr]=policy.metadata['default_dof_pos']
    mujoco.mj_forward(m,d)
    if args.task!='squat':
        refs=np.broadcast_to(d.xpos[state.selected_bodies],(len(motion['joint_pos']),len(selected),3)).copy()
        rots=np.broadcast_to(d.xquat[state.selected_bodies],(len(motion['joint_pos']),len(selected),4)).copy()
    mode=args.mode if args.mode is not None else (7 if args.task=='squat' else 0 if args.task in ['navigate','explore'] else 2)
    agent_decisions=[];path=None
    if args.task=='explore':
        from scene_reconstruction.navigation import collision_grid,plan
        manifest=json.loads((args.scene.parent/'manifest.json').read_text())
        scene_model=mujoco.MjModel.from_xml_path(str(args.scene));scene_data=mujoco.MjData(scene_model)
        for _ in range(2000):mujoco.mj_step(scene_model,scene_data)
        grid=collision_grid(scene_model,scene_data,manifest['dimensions'],radius=.32)
        # A map-based symbolic agent screens semantic exploration goals and chooses
        # the longest reachable route. It has no access to hidden policy rewards.
        candidates=[]
        for goal in manifest['navigation']['goals']:
            try:
                candidate=plan(grid,d.qpos[qa:qa+2],goal);length=np.linalg.norm(np.diff(candidate,axis=0),axis=1).sum()
                candidates.append((length,candidate));agent_decisions.append(dict(skill='navigate',goal=goal,status='reachable',length_m=float(length)))
            except ValueError as exc:agent_decisions.append(dict(skill='navigate',goal=goal,status='rejected',reason=str(exc)))
        if not candidates:raise RuntimeError('No reachable exploration target')
        _,path=max(candidates,key=lambda x:x[0]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        Path(f'reports/{args.run_name+"_" if args.run_name else ""}agent_decisions.json').write_text(json.dumps(agent_decisions,indent=2))
    initial=d.qpos[qa:qa+3].copy();rows=[];max_torque=0.;fall=False;frames=[]
    collision_pairs={};nonfloor_impulse=0.;peak_nonfloor_force=0.;force=np.zeros(6)
    renderer=None
    if args.video:
        renderer=mujoco.Renderer(m,height=480,width=640)
    def reference(step):
        inds=np.minimum(step+np.array(policy.metadata['future_idx']),len(refs)-1)
        if args.task=='squat':return refs[inds],rots[inds]
        pos=refs[0:6].copy();rot=rots[0:6].copy()
        if args.task=='explore':
            times=np.maximum(0,(step+np.array(policy.metadata['future_idx']))*.02-2.)
            for i,t in enumerate(times):
                distance=min(t*args.speed,arc[-1]);xy=np.array([np.interp(distance,arc,path[:,j]) for j in range(2)])
                ahead=min(distance+.20,arc[-1]);behind=max(0,distance-.10)
                tangent=np.array([np.interp(ahead,arc,path[:,j])-np.interp(behind,arc,path[:,j]) for j in range(2)])
                yaw=math.atan2(tangent[1],tangent[0]) if np.linalg.norm(tangent)>1e-6 else 0.
                pos[i,:,:2]+=xy-refs[0,0,:2]
                rot[i]=wxyz(Rotation.from_euler('z',yaw)*Rotation.from_quat(rots[0],scalar_first=True))
        elif args.task=='navigate':
            # Ramp movement after a two-second standing period, stopping by the
            # reachable floor goal before the next chair cluster.
            t=np.maximum(0,(step+np.array(policy.metadata['future_idx']))*.02-2.)
            pos[:,:,0]+=np.minimum(t*args.speed,1.10)[:,None]
        else:
            # Reach is an exploratory command, not a grasp. A rubber-hand G1 has
            # no finger actuator to close around the target object.
            blend=min(1.,max(0.,(step*.02-2.)/3.))
            prop=d.xpos[m.body('env_pick_box').id]
            for handname in ['left_wrist_yaw_link','right_wrist_yaw_link']:
                i=policy.metadata['selected_body_names'].index(handname)
                target=prop+np.array([0,.08 if handname.startswith('left') else -.08,.08])
                pos[:,i]=(1-blend)*pos[:,i]+blend*target
        return pos,rot
    try:
        for step in range(int(args.seconds/.02)):
            target_pos,target_rot=reference(step)
            rootpos=d.qpos[qa:qa+3].copy();rootrot=Rotation.from_quat(d.qpos[qa+3:qa+7],scalar_first=True)
            localpos=rootrot.inv().apply((target_pos-rootpos).reshape(-1,3)).reshape(target_pos.shape)
            localquat=wxyz(rootrot.inv()*Rotation.from_quat(target_rot.reshape(-1,4),scalar_first=True)).reshape(target_rot.shape)
            target,action=policy.act(d.qpos[qa+3:qa+7],d.qvel[va+3:va+6],d.qpos[state.qadr],d.qvel[state.vadr],localpos,localquat,mode)
            if not np.isfinite(target).all():raise RuntimeError('Nonfinite policy action')
            for _ in range(4):
                state.apply_pd(d,target);max_torque=max(max_torque,float(np.abs(d.ctrl).max()));mujoco.mj_step(m,d)
                for ci,contact in enumerate(d.contact):
                    g1,g2=int(contact.geom1),int(contact.geom2)
                    b1=m.body(int(m.geom_bodyid[g1])).name or '';b2=m.body(int(m.geom_bodyid[g2])).name or ''
                    if b1.startswith('env_')==b2.startswith('env_'):continue
                    environment=b1 if b1.startswith('env_') else b2
                    if environment.startswith('env_floor'):continue
                    mujoco.mj_contactForce(m,d,ci,force)
                    if force[0]<1.:continue
                    pair=' | '.join(sorted([m.geom(g1).name or str(g1),m.geom(g2).name or str(g2)]))
                    collision_pairs[pair]=collision_pairs.get(pair,0)+1
                    nonfloor_impulse+=float(force[0])*m.opt.timestep;peak_nonfloor_force=max(peak_nonfloor_force,float(force[0]))
            up=float(d.xmat[m.body('pelvis').id].reshape(3,3)[2,2]);error=float(np.linalg.norm(d.qpos[qa:qa+3]-target_pos[0,0]))
            rows.append(np.r_[d.time,d.qpos[qa:qa+3],target_pos[0,0],up,error])
            if renderer is not None:
                camera=mujoco.MjvCamera();camera.lookat=[3.2,2.2,.65];camera.distance=6.2;camera.azimuth=95;camera.elevation=-52
                renderer.update_scene(d,camera=camera);frames.append(renderer.render().copy())
            if d.qpos[qa+2]<.12 or up<.25 or not np.isfinite(d.qpos).all():
                fall=True;break
    finally:
        if renderer is not None:renderer.close()
    rows=np.array(rows);np.savetxt(output/f'{args.task}_mode{mode}_trajectory.csv',rows,delimiter=',',header='time,x,y,z,target_x,target_y,target_z,up_z,root_error_m',comments='')
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(output/f'{args.task}_mode{mode}.mp4',frames,fps=50,macro_block_size=1)
    displacement=float(np.linalg.norm(d.qpos[qa:qa+2]-initial[:2]));finalerror=float(rows[-1,-1]);rmse=float(np.sqrt((rows[:,-1]**2).mean()))
    success=not fall and (rmse<.3 if args.task=='squat' else displacement>.3 and finalerror<.3 if args.task in ['navigate','explore'] else False)
    report=dict(task=args.task,mode=mode,mode_name=policy.mode_names[mode],success=success,fall_detected=fall,simulated_seconds=float(d.time),requested_seconds=args.seconds,
        root_displacement_m=displacement,root_tracking_rmse_m=rmse,final_root_error_m=finalerror,max_applied_torque_nm=max_torque,
        initial_root_m=initial.tolist(),final_root_m=d.qpos[qa:qa+3].tolist(),checkpoint='released humanoid_transformer_m/model_22200.pt',
        controller='upstream PyTorch architecture and export wrapper; named joint mapping; 200 Hz torque-limited PD; 50 Hz BFM',
        reference_forcing=False,root_teleport_during_execution=False,
        environment_contact_pairs=collision_pairs,nonfloor_contact_impulse_ns=nonfloor_impulse,peak_nonfloor_contact_force_n=peak_nonfloor_force,
        collision_free_above_1n=len(collision_pairs)==0,
        scene_scale='provisional',limitations=['This is a scripted reference/task experiment, not an autonomous learned exploration agent.','No grasp success is claimed for the rubber-hand robot.'])
    report['scene_fingerprint']=scene_fingerprint(args.scene.parent)['sha256']
    report['scene_scale']=manifest['scale_status']
    if args.task=='explore':
        report['agent']='map-based symbolic goal screening + A* waypoints + closed-loop ScaleBFM root tracking'
        report['planned_goal_m']=path[-1].tolist();report['goal_error_m']=float(np.linalg.norm(d.qpos[qa:qa+2]-path[-1]))
        report['success']=not fall and report['goal_error_m']<.20
    Path(f'reports/{args.run_name+"_" if args.run_name else ""}scalebfm_{args.task}_mode{mode}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
