"""Task placement/frame invariance and scene-bound evaluation/perception checks."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from humanoid_skill_lab.integrations.scalebridge.reference_provider import ReferenceProvider
from humanoid_skill_lab.integrations.task_scene import perception_workspace, validate_task
from humanoid_skill_lab.skills.primitives import start
from humanoid_skill_lab.types import BeliefState, Pose
from humanoid_skill_lab.provenance import ROOT


def task(yaw=0., height=.74):
    return dict(robot_position=[3.,4.,.76], yaw=yaw, support_height=height, support="added_table",
                object_xy=[.455,-.185], target_xy=[.585,-.185])


def test_rotated_camera_pose_offsets_and_retreat_share_episode_axes():
    p = ReferenceProvider({"selected_body_names":["pelvis","right_wrist_yaw_link"]})
    p.frame_origin = np.array([3.,4.,0.])
    p.frame_rotation = Rotation.from_euler("z",np.pi/2).as_matrix()
    p.reset(np.array([[3.,4.,.76],[3.2,4.1,.8]]),np.tile([1.,0,0,0],(2,1)))
    belief = BeliefState(0,{"object":Pose((3.4,4.4,.62),(1.,0,0,0),"world",0,"camera")},{})
    start({"primitive":"move_hand_relative","parameters":{"reference":"object","offset":[-.2,0,.1],"duration":2.,"quaternion_wxyz":[1.,0,0,0]}},belief,p,0,frame="episode")
    positions,quats = p.sample(2,[0])
    np.testing.assert_allclose(positions[0,1],[3.4,4.2,.72])
    np.testing.assert_allclose(Rotation.from_quat(quats[0,1],scalar_first=True).apply([1,0,0]),[0,1,0],atol=1e-8)
    start({"primitive":"retract_hand","parameters":{"offset":[-.1,0,.12],"duration":2.}},belief,p,2,frame="episode")
    np.testing.assert_allclose(p.sample(4,[0])[0][0,1],[3.4,4.1,.84])


def test_task_binding_rejects_missing_support_and_invalid_initial_pose():
    with pytest.raises(ValueError):validate_task({})
    bad=task();bad["robot_position"][2]=-.2
    with pytest.raises(ValueError):validate_task(bad)


def build(tmp_path, config):
    import mujoco
    from humanoid_skill_lab.runner import make_scene
    m=mujoco.MjModel.from_xml_path(str(make_scene(tmp_path,{"task_scene":config})))
    d=mujoco.MjData(m)
    d.qpos[:7]=[*config["robot_position"],*Rotation.from_euler("z",config["yaw"]).as_quat(scalar_first=True)]
    meta=json.loads((ROOT/"checkpoints/scalebfm_m_metadata.json").read_text())
    for name,value in zip(meta["joint_names"],meta["default_dof_pos"]):
        d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
    mujoco.mj_forward(m,d)
    return mujoco,m,d


@pytest.mark.integration
def test_goal_uses_declared_height_and_rotated_region_not_old_world_bounds(tmp_path):
    from humanoid_skill_lab.evaluation.oracle import Evaluator
    config=task(np.pi/2)
    mujoco,m,d=build(tmp_path,config)
    evaluator=Evaluator(m,d,support_height=.74)
    q=m.jnt_qposadr[m.joint("env_cube_free").id]
    target=d.xpos[m.body("env_target").id].copy()
    d.qpos[q:q+3]=[*target[:2],.78]
    d.qpos[q+3:q+7]=Rotation.from_euler("z",np.pi/2).as_quat(scalar_first=True)
    mujoco.mj_forward(m,d);evaluator.update(d,[])
    assert evaluator.rows[-1]["inside"]
    d.qpos[q:q+2]+=Rotation.from_euler("z",np.pi/2).apply([.023,0,0])[:2]
    d.qpos[q+3:q+7]=Rotation.from_euler("z",3*np.pi/4).as_quat(scalar_first=True)
    mujoco.mj_forward(m,d);evaluator.update(d,[])
    assert not evaluator.rows[-1]["inside"]


@pytest.mark.integration
@pytest.mark.parametrize("yaw",[0.,np.pi/2])
def test_onboard_rgbd_recovers_transformed_task_props_without_oracle(tmp_path,yaw):
    from humanoid_skill_lab.perception.onboard_camera import OnboardCamera
    from humanoid_skill_lab.perception.state_estimator import StateEstimator
    config=task(yaw,height=.575)
    _,m,d=build(tmp_path,config)
    camera=OnboardCamera(m)
    try:
        observation=camera.observe(d,"onboard_vision")
        assert observation.oracle_poses=={}
        belief=StateEstimator("onboard_vision",workspace=perception_workspace(config)).update(observation)
        assert {"object","target"} <= belief.poses.keys()
        for name,body in [("object","env_cube"),("target","env_target")]:
            assert belief.poses[name].source=="camera"
            assert np.linalg.norm(np.array(belief.poses[name].position)-d.xpos[m.body(body).id]) < .015
    finally:
        camera.close()
