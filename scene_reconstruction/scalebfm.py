"""Eager PyTorch adapter using the pinned upstream architecture/export wrapper.

No TensorRT binary is reused across GPU generations. All state, action, and
body indexes are resolved by names so environment joints cannot shift them.
"""
import ast
import importlib.util
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def load_export_helpers(source):
    path=source/'ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py'
    tree=ast.parse(path.read_text(),filename=str(path))
    names={'quat_apply','quat_apply_inverse','quat_mul','quat_mul_inverse_left','quat_mul_inverse_right','HumanoidTransformerPolicyWrapperWithMode','build_mode_mappings','parse_xml'}
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if len(selected)!=len(names):raise RuntimeError('Upstream export helpers changed; inspect before adapting')
    # Importing the whole export script starts Isaac Sim and requires TensorRT.
    # Keep its pure functions/classes verbatim, executing eagerly without decorators.
    for node in selected:
        if isinstance(node,ast.FunctionDef):node.decorator_list=[]
    namespace=dict(torch=torch,nn=nn,np=np,ET=ET,math=math)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def build_robot_scene(robot_path,scene_path,output):
    import mujoco
    robot_path=robot_path.resolve();scene_path=scene_path.resolve();output=output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    robot=mujoco.MjSpec.from_file(str(robot_path));scene=mujoco.MjSpec.from_file(str(scene_path))
    for mesh in robot.meshes:
        if mesh.file:mesh.file=str(robot_path.parent/'meshes'/mesh.file)
    robot.meshdir=''
    for texture in scene.textures:
        if texture.file:texture.file=str(scene_path.parent/texture.file)
    scene.texturedir=''
    for mesh in scene.meshes:
        if mesh.file:mesh.file=str(scene_path.parent/mesh.file)
    scene.meshdir=''
    if (scene_path.parent/'photo_ownership.json').exists():
        for key in ['ambient','diffuse','specular']:getattr(robot.visual.headlight,key)[:]=0
        for material in robot.materials:material.emission=.8
        review=robot.add_material(name='robot_review_unlit',emission=.8)
        for geom in robot.geoms:
            if not geom.material:geom.material=review.name
    floor=robot.geom('floor')
    if floor is not None:robot.delete(floor)
    robot.option.timestep=.005;scene.option.timestep=.005
    robot.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;robot.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
    robot.visual.quality.shadowsize=2048
    frame=robot.worldbody.add_frame(name='scene_frame');robot.attach(scene,prefix='env_',frame=frame)
    model=robot.compile()
    # Save references relative to the composed XML so the repository is movable.
    import os
    root=ET.fromstring(robot.to_xml())
    for asset in root.findall('./asset/*'):
        if asset.get('file') and Path(asset.get('file')).is_absolute():asset.set('file',os.path.relpath(asset.get('file'),output.parent))
    ET.indent(root);ET.ElementTree(root).write(output,encoding='unicode')
    return model


class ScaleBFM:
    def __init__(self,source,checkpoint,metadata,robot_xml,output,device='cpu',expected_sha256='88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3',expected_metadata_sha256='3024f18136314fade2676e22d6eafb1a8b2befa388da55b0885a53fcb1368d16'):
        self.metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest()
        if expected_metadata_sha256 is not None and self.metadata_sha256!=expected_metadata_sha256:raise RuntimeError('ScaleBFM deployment metadata digest mismatch; review gains, joint order and observation configuration before changing the pin')
        self.metadata=json.loads(metadata.read_text());self.device=device
        with checkpoint.open('rb') as f:self.checkpoint_sha256=hashlib.file_digest(f,'sha256').hexdigest()
        if expected_sha256 is not None and self.checkpoint_sha256!=expected_sha256:raise RuntimeError('ScaleBFM checkpoint digest does not match the selected official medium model')
        network=load_module(source/'ScaleTrack/source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py','scalebfm_network')
        cfg=self.metadata['policy_architecture']
        actor=network.HumanoidTransformer(cfg['prop_obs_dim'],cfg['action_dim'],cfg['output_dim'],cfg['embedding_dim'],cfg['num_heads'],cfg['ff_dim'],cfg['num_layers'])
        embedder=network.TaskEmbedder(cfg['task_obs_dim'],cfg['embedding_dim'],cfg['reduced_task_dim'],cfg['task_embedder_hidden_dims'])
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
        actor.load_state_dict({k.removeprefix('actor.'):v for k,v in state.items() if k.startswith('actor.')},strict=True)
        embedder.load_state_dict({k.removeprefix('actor_task_embedder.'):v for k,v in state.items() if k.startswith('actor_task_embedder.')},strict=True)
        helper=load_export_helpers(source);self.helper=helper
        # Fixed visual/sensor leaf bodies in ScaleBridge are absent in the training
        # FK tree. Remove only those leaves; retain every articulated transform.
        tree=ET.parse(robot_xml)
        for parent in tree.getroot().iter():
            for child in list(parent):
                if child.tag=='body' and not child.findall('joint') and not child.findall('freejoint') and not child.findall('body'):parent.remove(child)
        output.mkdir(parents=True,exist_ok=True);fk_path=output/'fk_robot.xml';tree.write(fk_path)
        names,joints,parents,axis,translation,rotation=helper['parse_xml'](fk_path,device)
        local_rot=rotation/rotation.norm(dim=-1,keepdim=True)
        selected=torch.tensor([names.index(n) for n in self.metadata['selected_body_names']],device=device)
        lab_to_xml=torch.tensor([self.metadata['joint_names'].index(n) for n in joints],device=device)
        modefile=source/'ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/config/g1_29dof/flat_env_cfg.py'
        candidates=[ast.literal_eval(n.value) for n in ast.walk(ast.parse(modefile.read_text())) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='mode_candidates' for t in n.targets)]
        if len(candidates)!=1:raise RuntimeError('Mode configuration changed')
        table=torch.tensor([[float(name in links) for name in self.metadata['selected_body_names']] for links in candidates[0].values()],device=device)
        mapping=helper['build_mode_mappings'](table,self.metadata['mode_feature_dims'],self.metadata['mode_mapping_with_time'])
        def tensor(value):return torch.as_tensor(value,dtype=torch.float32,device=device)
        self.wrapper=helper['HumanoidTransformerPolicyWrapperWithMode'](SimpleNamespace(actor=actor,actor_task_embedder=embedder),mapping,table,
            tensor(self.metadata['default_dof_pos'])[None],tensor(self.metadata['action_scale'])[None],self.metadata['history_buffer_size'],len(self.metadata['future_idx']),
            translation,local_rot,parents,axis,selected,lab_to_xml).eval().to(device)
        self.actions=torch.zeros(1,self.metadata['history_buffer_size'],29,device=device)
        self.history={};self.mode_names=list(candidates[0])

    def reset(self):self.history={};self.actions.zero_()

    @torch.inference_mode()
    def act(self,root_quat,base_ang_vel,joint_pos,joint_vel,target_pos_local,target_quat_local,mode=7):
        context=self.metadata['history_buffer_size']
        current=dict(root=torch.as_tensor(root_quat,dtype=torch.float32,device=self.device),angular=torch.as_tensor(base_ang_vel,dtype=torch.float32,device=self.device),pos=torch.as_tensor(joint_pos,dtype=torch.float32,device=self.device),vel=torch.as_tensor(joint_vel,dtype=torch.float32,device=self.device))
        for key,value in current.items():
            if key not in self.history:self.history[key]=value[None,None].expand(1,context,-1).clone()
            else:self.history[key]=torch.cat([self.history[key][:,1:],value[None,None]],dim=1)
        pos=torch.as_tensor(target_pos_local,dtype=torch.float32,device=self.device)[None]
        quat=torch.as_tensor(target_quat_local,dtype=torch.float32,device=self.device)[None]
        offsets=torch.as_tensor(self.metadata['future_idx'],device=self.device)[None,:,None]
        target,action=self.wrapper(self.history['root'],self.history['angular'],self.history['pos'],self.history['vel'],self.actions,pos,quat,torch.tensor([mode],device=self.device),offsets)
        self.actions=torch.cat([self.actions[:,1:],action[:,None]],dim=1)
        return target[0].cpu().numpy(),action[0].cpu().numpy()


class RobotState:
    def __init__(self,model,metadata):
        self.model=model;self.metadata=metadata
        self.joints=np.array([model.joint(n).id for n in metadata['joint_names']]);self.qadr=model.jnt_qposadr[self.joints];self.vadr=model.jnt_dofadr[self.joints]
        self.actuators=np.array([model.actuator(n).id for n in metadata['action_names']])
        rootjoint=model.body_jntadr[model.body('pelvis').id];self.root_qadr=model.jnt_qposadr[rootjoint];self.root_vadr=model.jnt_dofadr[rootjoint]
        self.selected_bodies=np.array([model.body(n).id for n in metadata['selected_body_names']])

    def apply_pd(self,data,target):
        torque=(target-data.qpos[self.qadr])*np.array(self.metadata['stiffness'])-data.qvel[self.vadr]*np.array(self.metadata['damping'])
        limit=np.array(self.metadata['torque_limit']);data.ctrl[self.actuators]=np.clip(torque,-limit,limit)
