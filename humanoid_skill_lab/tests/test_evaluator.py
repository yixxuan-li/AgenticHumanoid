from types import SimpleNamespace
import numpy as np
import pytest
from humanoid_skill_lab.evaluation.oracle import Evaluator


@pytest.fixture
def physics(tmp_path):
    import mujoco
    from humanoid_skill_lab.runner import make_scene
    m=mujoco.MjModel.from_xml_path(str(make_scene(tmp_path,{})))
    d=mujoco.MjData(m)
    d.qpos[:7]=[0,0,.76,1,0,0,0]
    mujoco.mj_forward(m,d)
    return mujoco,m,d


def test_unnamed_foot_geoms_do_not_become_forbidden(physics):
    mujoco,m,d=physics
    e=Evaluator(m,d)
    e.update(d,[{'timestamp':0,'geom1':'env_floor','geom2':'','body1':'world',
                 'body2':'right_ankle_roll_link','normal_force':50}])
    assert e.forbidden==[]
    e.update(d,[{'timestamp':0,'geom1':'env_table_top','geom2':'','body1':'env_table',
                 'body2':'right_wrist_yaw_link','normal_force':1}])
    assert len(e.forbidden)==1


def test_cube_center_inside_is_not_enough_for_rotated_footprint(physics):
    mujoco,m,d=physics
    e=Evaluator(m,d)
    q=m.jnt_qposadr[m.joint('env_cube_free').id]
    # Unit fixture initialization only; no controller rollout is claimed here.
    d.qpos[q:q+3]=[.62,-.18,.615]
    d.qpos[q+3:q+7]=[np.cos(np.pi/8),0,0,np.sin(np.pi/8)]
    mujoco.mj_forward(m,d)
    e.update(d,[])
    assert not e.rows[-1]['inside']


def test_program_completion_does_not_imply_task_success(physics):
    mujoco,m,d=physics
    e=Evaluator(m,d)
    e.update(d,[])
    result=e.report(SimpleNamespace(outcome='completed',reason=None),.02,18)
    assert not result['success']
    assert not result['goal_held']
