import json
import pytest
from humanoid_skill_lab.exploration.promotion import compare,Registry
from humanoid_skill_lab.exploration.experiment import check_datasets
from humanoid_skill_lab.provenance import ROOT


def report(success,key='same'):
    return {'success':success,'configuration_sha256':key,'physics_executed':True,'fallen':False,'timeout':False,'forbidden_contacts':0}


def test_promotion_requires_success_and_no_regression():
    assert compare([report(True)]*3,[report(False)]*3)['accepted']
    assert not compare([report(False)]*3,[report(False)]*3)['accepted']
    assert not compare([report(True),report(False),report(True)],[report(False),report(True),report(False)])['accepted']
    with pytest.raises(ValueError): compare([report(True,'a')]*3,[report(False,'b')]*3)
    r=report(True);r['fallen']=True
    assert not compare([r]*3,[report(False)]*3)['accepted']


def test_snapshot_rollback_and_tamper_detection(tmp_path):
    registry=Registry(tmp_path)
    a=registry.commit({'version':1},{'rules':[]},{'reason':'baseline'},initial=True)
    b=registry.commit({'version':2},{'rules':[]},{'accepted':True})
    registry.select(a)
    assert registry.current()[0]==a
    registry.select(b)
    (tmp_path/'snapshots'/f'{b}.json').write_text('{}')
    with pytest.raises(ValueError):registry.current()
    with pytest.raises(ValueError):registry.select('../other')


def test_split_overlap_including_same_physics_different_seed():
    c=json.loads((ROOT/'humanoid_skill_lab/configs/experiment.json').read_text())
    check_datasets(c)
    c['test'][0]=dict(c['development'][0],seed=900)
    with pytest.raises(ValueError):check_datasets(c)
