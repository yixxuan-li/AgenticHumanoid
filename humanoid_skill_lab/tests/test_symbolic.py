from copy import deepcopy
import pytest
from humanoid_skill_lab.symbolic.knowledge import Knowledge
from humanoid_skill_lab.symbolic.planner import ground,plan,apply_effects
from humanoid_skill_lab.symbolic.predicates import bind
from humanoid_skill_lab.types import Truth


def rule():
    return {'skill':'pick','skill_version':1,'preconditions':['visible(?object)','hand_free(right)'],
            'add_effects':['holding(right,?object)'],'delete_effects':['hand_free(right)'],
            'status':'validated','evidence':{}}


def test_all_preconditions_and_binding_required():
    operators=ground(rule(),{'?object':['cube','ball']},{'pick':1})
    assert plan({'visible(cube)':Truth.TRUE},['holding(right,cube)'],operators) is None
    assert plan({'visible(cube)':Truth.TRUE,'hand_free(right)':Truth.UNKNOWN},['holding(right,cube)'],operators) is None
    result=plan({'visible(cube)':Truth.TRUE,'hand_free(right)':Truth.TRUE},['holding(right,cube)'],operators)
    assert len(result)==1 and dict(result[0].bindings)=={'?object':'cube'}
    state=apply_effects({'hand_free(right)':Truth.TRUE},result[0])
    assert state['hand_free(right)']==Truth.FALSE
    assert state['holding(right,cube)']==Truth.TRUE


def test_effect_conflict_blocks_plan():
    r=rule(); r['delete_effects']=[]
    operators=ground(r,{'?object':['cube']},{'pick':1})
    assert plan({'visible(cube)':Truth.TRUE,'hand_free(right)':Truth.TRUE},['holding(right,cube)'],operators) is None


def test_candidate_and_old_version_not_plannable():
    r=rule(); r['status']='candidate'
    assert ground(r,{'?object':['cube']},{'pick':1})==[]
    r['status']='validated'
    assert ground(r,{'?object':['cube']},{'pick':2})==[]
    with pytest.raises(ValueError): bind('visible(?object)',{})


def test_rule_requires_contrastive_evidence_and_coverage():
    r=rule(); r['status']='candidate'; k=Knowledge([r])
    k.record(0,'val1','validation',True)
    with pytest.raises(ValueError): k.validate(0,[],1.)
    trials=[{'condition':c,'only_changed_condition':True,'present_success':True,
             'absent_success':False,'split':'validation'} for c in r['preconditions']]
    with pytest.raises(ValueError): k.validate(0,trials,.1)
    k.validate(0,trials,.8)
    assert k.rules[0]['status']=='validated'
    k.revise_skill('pick',2)
    assert k.rules[0]['status']=='revalidation_required'
    with pytest.raises(ValueError): k.record(0,'test1','test',True)


def test_three_skill_composition_and_consumed_state():
    picked=rule()
    place={'skill':'place','skill_version':1,'preconditions':['holding(right,?object)'],
           'add_effects':['object_in_region(?object,?region)','hand_free(right)'],
           'delete_effects':['holding(right,?object)'],'status':'validated','evidence':{}}
    close={'skill':'close','skill_version':1,'preconditions':['object_in_region(?object,?region)','hand_free(right)'],
           'add_effects':[],'delete_effects':['drawer_open(drawer)'],'status':'validated','evidence':{}}
    domains={'?object':['cube'],'?region':['drawer']}; versions={'pick':1,'place':1,'close':1}
    ops=ground(picked,domains,versions)+ground(place,domains,versions)
    result=plan({'visible(cube)':Truth.TRUE,'hand_free(right)':Truth.TRUE},['object_in_region(cube,drawer)','hand_free(right)'],ops)
    assert [op.skill for op in result]==['pick','place']
