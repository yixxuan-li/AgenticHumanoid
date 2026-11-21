from copy import deepcopy
import json
import numpy as np
import pytest
from humanoid_skill_lab.provenance import ROOT
from humanoid_skill_lab.skills.schema import check, load, ValidationError
from humanoid_skill_lab.skills.runtime import SkillRuntime
from humanoid_skill_lab.integrations.scalebridge.reference_provider import ReferenceProvider
from humanoid_skill_lab.types import BeliefState, Pose, Truth


@pytest.fixture
def program():
    return load(ROOT/'humanoid_skill_lab/configs/push.json')


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(max_transitions=0),
    lambda x: x['stages'][0].update(primitive='open_drawer'),
    lambda x: x['stages'][0].update(primitive=['wait']),
    lambda x: x['stages'][1]['parameters'].update(reference={'object':True}),
    lambda x: x['stages'][0]['parameters'].update(duration=float('nan')),
    lambda x: x['stages'][0].update(next='missing'),
    lambda x: x['stages'][1]['parameters'].update(python='import os'),
    lambda x: x['stages'][1]['parameters'].update(quaternion_wxyz=[0,0,0,0]),
    lambda x: x['stages'][0].update(timeout=.01),
])
def test_reject_unsafe_or_unbounded_program(program, mutation):
    mutation(program)
    with pytest.raises(ValidationError):
        check(program)


def provider():
    p=ReferenceProvider({'selected_body_names':['pelvis','right_wrist_yaw_link']})
    p.reset(np.array([[0,0,.76],[.1,-.2,.8]]),np.array([[1.,0,0,0],[1.,0,0,0]]))
    return p


def test_retries_have_finite_global_bound():
    p={'name':'retry','version':1,'max_transitions':2,'stages':[
        {'id':'observe','primitive':'observe','parameters':{'duration':.1},'timeout':.2,
         'success_when':'visible(object)','on_timeout':'observe'}]}
    runtime=SkillRuntime(p,provider())
    belief=BeliefState(0,{}, {})
    for now in np.arange(0,2,.02):
        runtime.update(float(now),belief)
    assert runtime.outcome=='failure'
    assert runtime.reason=='transition_budget_exhausted'
    assert runtime.visits==2


def test_unknown_never_satisfies_condition():
    p={'name':'check','version':1,'max_transitions':1,'stages':[
        {'id':'look','primitive':'observe','parameters':{'duration':.1},'timeout':.2,'success_when':'visible(object)'}]}
    r=SkillRuntime(p,provider())
    r.update(0,BeliefState(0,{},{}))
    r.update(.2,BeliefState(.2,{},{}))
    assert r.outcome=='timeout'


def test_stale_pose_rejected(program):
    program['stages']=program['stages'][1:]
    r=SkillRuntime(program,provider())
    pose=Pose((.4,-.2,.6),(1,0,0,0),'world',0,'oracle')
    r.update(1,BeliefState(1,{'object':pose},{}))
    assert r.outcome=='failure'
    assert 'stale' in r.reason


def test_unknown_and_stale_beliefs():
    b=BeliefState(1,{}, {'visible(object)':Truth.TRUE})
    assert b.query('holding(right,cube)')==Truth.UNKNOWN
    assert b.query('visible(object)',now=2)==Truth.UNKNOWN
    assert b.query('visible(object)',now=.5)==Truth.UNKNOWN
