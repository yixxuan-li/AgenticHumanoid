"""Compatibility and coordinate tests for the merged reconstruction/skill paths."""
import json
from pathlib import Path
import numpy as np
import pytest
from humanoid_skill_lab.integrations.reconstructed_scene import load_scene
from humanoid_skill_lab.integrations.scalebridge.reference_provider import ReferenceProvider
from humanoid_skill_lab.skills.primitives import start
from humanoid_skill_lab.types import BeliefState
from humanoid_skill_lab.provenance import ROOT


@pytest.fixture
def scene(tmp_path):
    source=tmp_path/'export';source.mkdir()
    (source/'scene.xml').write_text('<mujoco><worldbody><body name="floor_main" pos="5 3 -.06"><geom name="floor_support" type="box" size="4 4 .06"/></body></worldbody></mujoco>')
    (source/'scene.usda').write_text('#usda 1.0\n')
    (source/'manifest.json').write_text(json.dumps({'navigation':{'start':[5.,3.]},'scale_status':'assumed_test_scale','limitations':['structural export fixture']}))
    return source


def test_reuses_existing_scene_fingerprint_and_does_not_invent_affordances(scene):
    from scene_reconstruction.provenance import scene_fingerprint
    context=load_scene(scene)
    assert context['identity']==scene_fingerprint(scene)
    assert context['robot_start']==[5.,3.,.76]
    assert context['manipulation_validated'] is False
    assert context['declared_pick_profiles'] is False
    assert context['scale_status']=='assumed_test_scale'


def test_episode_frame_translates_program_targets_not_relative_offsets():
    provider=ReferenceProvider({'selected_body_names':['pelvis','right_wrist_yaw_link']})
    provider.frame_origin=np.array([5.,3.,0.])
    provider.reset(np.array([[5,3,.76],[5.1,2.8,.8]]),np.array([[1.,0,0,0],[1.,0,0,0]]))
    stage={'primitive':'move_hand_world','parameters':{'position':[.32,-.24,.85],'duration':3.}}
    start(stage,BeliefState(0,{},{}),provider,0,frame='episode')
    target,_=provider.sample(3,[0])
    np.testing.assert_allclose(target[0,1],[5.32,2.76,.85])
    np.testing.assert_allclose(target[0,0],[5.,3.,.76])


@pytest.mark.integration
def test_online_environment_uses_old_scene_spawn_without_adding_task_objects(scene,tmp_path):
    from humanoid_skill_lab.runner import make_scene
    from humanoid_skill_lab.integrations.scalebridge.backend import EagerBFMAgent,OnlineEnv
    from humanoid_skill_lab.perception.onboard_camera import OnboardCamera
    output=tmp_path/'episode';output.mkdir()
    context=load_scene(scene)
    path=make_scene(output,{'scene_directory':str(scene)})
    agent=EagerBFMAgent(output,device='cpu',mode=2)
    provider=ReferenceProvider(agent.get_meta_data())
    env=OnlineEnv(agent.get_meta_data(),path,provider,device='cpu',initial_root_position=context['robot_start'])
    obs=env.reset()
    np.testing.assert_allclose(env.simulator.mujoco_data.qpos[:3],[5,3,.76])
    assert env.simulator.mujoco_model.body('env_floor_main').id>=0
    with pytest.raises(KeyError):env.simulator.mujoco_model.body('env_cube')
    camera=OnboardCamera(env.simulator.mujoco_model,object_bodies={})
    try:
        observation=camera.observe(env.simulator.mujoco_data,'oracle_debug')
        assert observation.oracle_poses=={}
    finally:camera.close()
    assert load_scene(scene)['identity']==context['identity']


def test_imported_push_requires_declared_benchmark_instead_of_fake_fixture(scene,tmp_path):
    from humanoid_skill_lab.runner import run_episode
    from humanoid_skill_lab.skills.schema import load
    with pytest.raises(ValueError,match='declared task bindings'):
        run_episode(load(ROOT/'humanoid_skill_lab/configs/push.json'),{'scene_directory':str(scene)},tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()
