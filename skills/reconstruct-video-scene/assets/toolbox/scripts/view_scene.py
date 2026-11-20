"""Native scene viewer: physical perturbations, cabinet control and visual review."""
import argparse
import os
import sys
import queue
import time
from pathlib import Path
import mujoco
import mujoco.viewer

p=argparse.ArgumentParser();p.add_argument('--fast-exit',action='store_true',help='Skip interpreter graphics teardown after the viewer closes');p.add_argument('--collision',action='store_true');p.add_argument('--scene',type=Path,default=Path('outputs/scene/scene.xml'));args=p.parse_args()
m=mujoco.MjModel.from_xml_path(str(args.scene));d=mujoco.MjData(m);events=queue.SimpleQueue()
paused=False;door_target=None;cutaway=False;collision_view=False
original_rgba=m.geom_rgba.copy();original_matid=m.geom_matid.copy()
photo_cameras=[i for i in range(m.ncam) if m.camera(i).name.startswith('source_')]
cutaway_bodies=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['ceiling_visual','wall_front_observed','wall_entry_front','photo_cutaway_ceiling','photo_cutaway_front']]
ceiling_geom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'ceiling_collision')
def set_collision(viewer,enabled):
    m.geom_rgba[:]=original_rgba;m.geom_matid[:]=original_matid
    viewer.opt.geomgroup[1]=not enabled
    if enabled:
        mask=(m.geom_contype!=0)|(m.geom_conaffinity!=0);m.geom_rgba[mask]=[.35,.65,.85,.65];m.geom_matid[mask]=-1
    if cutaway:
        for bid in cutaway_bodies:
            if bid>=0:
                adr=m.body_geomadr[bid];count=m.body_geomnum[bid];m.geom_rgba[adr:adr+count,3]=0
        if ceiling_geom>=0:m.geom_rgba[ceiling_geom,3]=0

jid=None
for name in ['cabinet_door_hinge','env_cabinet_door_hinge']:
    try:jid=m.joint(name).id;break
    except KeyError:pass
print('V cutaway | F collision view | [ / ] source cameras | P pause | R reset | O open cabinet | C close cabinet | 1–4 fixed cameras. MuJoCo mouse controls support view rotation, selection and physical perturbation.')
with mujoco.viewer.launch_passive(m,d,key_callback=events.put) as viewer:
    if m.ncam:viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED;viewer.cam.fixedcamid=min(1,m.ncam-1)
    if args.collision:collision_view=True;set_collision(viewer,True)
    while viewer.is_running():
        start=time.perf_counter()
        with viewer.lock():
            while not events.empty():
                key=events.get()
                if key in [ord('P'),ord('p')]:paused=not paused
                elif key in [ord('R'),ord('r')]:mujoco.mj_resetData(m,d);door_target=None
                elif key in [ord('O'),ord('o')]:door_target=-1.4
                elif key in [ord('C'),ord('c')]:door_target=0.
                elif key in [ord('F'),ord('f')]:collision_view=not collision_view;set_collision(viewer,collision_view)
                elif key in [ord('V'),ord('v')]:
                    cutaway=not cutaway;set_collision(viewer,collision_view)
                elif key in [ord('['),ord(']')] and photo_cameras:
                    current=photo_cameras.index(viewer.cam.fixedcamid) if viewer.cam.fixedcamid in photo_cameras else 0
                    viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED;viewer.cam.fixedcamid=photo_cameras[(current+(1 if key==ord(']') else -1))%len(photo_cameras)]
                elif m.ncam and ord('1')<=key<=ord('4'):
                    viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED;viewer.cam.fixedcamid=min(key-ord('1'),m.ncam-1)
            d.qfrc_applied[:]=0
            if door_target is not None and jid is not None:
                qa=m.jnt_qposadr[jid];va=m.jnt_dofadr[jid]
                d.qfrc_applied[va]=max(-4.,min(4.,10*(door_target-d.qpos[qa])-2*d.qvel[va]))
            if not paused:mujoco.mj_step(m,d)
        viewer.sync()
        time.sleep(max(0,m.opt.timestep-(time.perf_counter()-start)))

if args.fast_exit:
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
