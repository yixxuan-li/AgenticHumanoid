import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from humanoid_skill_lab.integrations.scalebridge.reference_provider import ReferenceProvider
from humanoid_skill_lab.types import Observation
from humanoid_skill_lab.perception.state_estimator import StateEstimator
from humanoid_skill_lab.types import Pose


def test_root_transforms_match_independent_scipy():
    from humanoid_skill_lab.integrations.scalebridge.backend import components
    rng=np.random.default_rng(20)
    root=Rotation.from_euler('xyz',[.2,-.3,1.4])
    root_position=np.array([2.,3.,.7])
    positions=rng.normal(size=(1,6,14,3))
    rotations=Rotation.random(84,random_state=rng).as_quat(scalar_first=True).reshape(1,6,14,4)
    buffer={'root_quat_wxyz_buffer':torch.tensor(root.as_quat(scalar_first=True)[None,None]),
            'root_pos_buffer':torch.tensor(root_position[None,None]),'body_pos_w_future':torch.tensor(positions),
            'body_quat_w_wxyz_future':torch.tensor(rotations)}
    result=components.target_body_pos_future_to_robot_base(buffer).numpy()
    expected=root.inv().apply((positions-root_position).reshape(-1,3)).reshape(1,6,14,3)
    np.testing.assert_allclose(result,expected,atol=1e-6)
    actual=components.target_body_rot_future_to_robot_base(buffer).numpy()
    expected_rot=(root.inv()*Rotation.from_quat(rotations.reshape(-1,4),scalar_first=True)).as_quat(scalar_first=True).reshape(1,6,14,4)
    np.testing.assert_allclose(actual,expected_rot,atol=1e-6)


def test_future_offsets_and_uncontrolled_bodies_are_preserved():
    p=ReferenceProvider({'selected_body_names':['pelvis','hand','foot']})
    initial=np.array([[0,0,.76],[.1,-.2,.8],[0,-.1,.03]])
    rotations=np.tile([1.,0,0,0],(3,1))
    p.reset(initial,rotations)
    p.command(1,'hand',[.2,-.2,.8],2)
    positions,quats=p.sample(1,[0,1,2,3,4,5])
    np.testing.assert_allclose(positions[:,0],np.tile(initial[0],(6,1)))
    np.testing.assert_allclose(positions[:,2],np.tile(initial[2],(6,1)))
    assert np.all(np.diff(positions[:,1,0])>0)
    np.testing.assert_allclose(quats,np.broadcast_to(rotations,(6,3,4)))
    end,_=p.sample(3,[0]);np.testing.assert_allclose(end[0,1],[.2,-.2,.8])
    with pytest.raises(ValueError):p.command(3,'hand',[1,-.2,.8],.1)


def test_vision_rejects_oracle_truth():
    obs=Observation(0,np.zeros((24,32,3),dtype=np.uint8),np.eye(3),np.eye(4),{},'camera',
                    np.ones((24,32)),{'object':Pose((0,0,0),(1,0,0,0),'world',0,'oracle')})
    with pytest.raises(ValueError):StateEstimator('onboard_vision').update(obs)
    obs.oracle_poses={};obs.depth=None
    result=StateEstimator('onboard_vision').update(obs)
    assert not result.poses


@pytest.mark.integration
def test_official_loop_uses_live_references_and_preserves_history(tmp_path):
    from humanoid_skill_lab.integrations.scalebridge.backend import EagerBFMAgent,OnlineEnv
    from humanoid_skill_lab.runner import make_scene
    agent=EagerBFMAgent(tmp_path,device='cpu',mode=2)
    provider=ReferenceProvider(agent.get_meta_data())
    env=OnlineEnv(agent.get_meta_data(),make_scene(tmp_path,{}),provider,device='cpu')
    obs=env.reset()
    assert obs['target_body_pos_future_to_robot_base'].shape==(1,6,14,3)
    for step in range(3):
        target,action=agent.get_action(obs)
        obs=env.step(target,action)
        np.testing.assert_allclose(obs['actions_buffer'][0,-1].detach().numpy(),action[0].detach().numpy())
    assert int(env.episode_length_buf[0])==3
    assert env.simulator.mujoco_data.time==pytest.approx(.06)
    assert obs['future_time_offsets'].flatten().tolist()==[0,1,2,3,4,5]
    import mujoco
    before=env.simulator.mujoco_data.xpos.copy()
    mujoco.mj_kinematics(env.simulator.mujoco_model,env.simulator.mujoco_data)
    np.testing.assert_allclose(before,env.simulator.mujoco_data.xpos,atol=1e-12)


@pytest.mark.integration
def test_controlled_rgbd_localizes_from_images_without_object_truth(tmp_path):
    import json
    from humanoid_skill_lab.provenance import ROOT
    from humanoid_skill_lab.runner import make_scene
    from humanoid_skill_lab.perception.onboard_camera import OnboardCamera
    import mujoco
    metadata=json.loads((ROOT/'checkpoints/scalebfm_m_metadata.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(make_scene(tmp_path,{})));d=mujoco.MjData(m)
    d.qpos[:7]=[0,0,.76,1,0,0,0]
    for name,value in zip(metadata['joint_names'],metadata['default_dof_pos']):
        d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    mujoco.mj_forward(m,d)
    camera=OnboardCamera(m)
    try:
        obs=camera.observe(d,'onboard_vision')
        assert obs.oracle_poses=={}
        belief=StateEstimator('onboard_vision').update(obs)
        assert belief.poses['object'].source=='camera'
        assert belief.poses['target'].source=='camera'
        # Evaluation-only comparison is outside the estimator's input boundary.
        assert np.linalg.norm(np.array(belief.poses['object'].position)-d.xpos[m.body('env_cube').id])<.01
        assert np.linalg.norm(np.array(belief.poses['target'].position)[:2]-d.xpos[m.body('env_target').id,:2])<.01
    finally:
        camera.close()
