"""Audit final real-run artifacts independently of task outcome claims."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from humanoid_skill_lab.provenance import source_hashes,digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cycle',type=Path,default=ROOT/'outputs/skill_lab/final_audit_cycle_v2')
    p.add_argument('--success',type=Path,default=ROOT/'outputs/skill_lab/isolated_calibrated_push_final')
    p.add_argument('--output',type=Path,default=ROOT/'reports/skill_lab_integrity.json')
    p.add_argument('--require-current-source',action='store_true',help='Also require historical records to match the current working tree')
    args=p.parse_args()
    current_sources=source_hashes()
    records=[]
    paths=sorted(path.parent for path in args.cycle.glob('*/metrics.json'))+[args.success]
    assert len(paths)==9,'Final cycle must have eight physical episodes plus the successful calibration repeat'
    expected_sources=json.loads((paths[0]/'manifest.json').read_text())['source_hashes']
    if args.require_current_source:
        assert expected_sources==current_sources,'This historical run predates the current implementation'
    required=['manifest.json','skill.json','knowledge.json','observations.jsonl','references.npz','robot_states.npz',
              'events.jsonl','onboard.mp4','overview.mp4','evaluation_trace.json','metrics.json']
    for path in paths:
        assert all((path/name).exists() for name in required),str(path)
        manifest=json.loads((path/'manifest.json').read_text())
        metrics=json.loads((path/'metrics.json').read_text())
        assert manifest['result_status']=='executed' and metrics['physics_executed']
        assert manifest['source_hashes']==expected_sources,'Source changed within the recorded release audit'
        refs=np.load(path/'references.npz');states=np.load(path/'robot_states.npz')
        n=len(refs['timestamp'])
        assert refs['world_positions'].shape==(n,6,14,3)
        assert refs['world_quaternions_wxyz'].shape==(n,6,14,4)
        assert states['joint_targets'].shape==(n,29)
        assert states['policy_actions'].shape==(n,29)
        assert states['body_positions'].shape==(n,14,3)
        np.testing.assert_allclose(np.diff(refs['timestamp']),.02,atol=1e-8)
        for name in refs.files:assert np.isfinite(refs[name]).all()
        for name in states.files:assert np.isfinite(states[name]).all()
        np.testing.assert_allclose(np.linalg.norm(refs['world_quaternions_wxyz'],axis=-1),1,atol=1e-5)
        for index in [0,n//2,n-1]:
            root=states['qpos_qvel'][index,:7]
            local=Rotation.from_quat(root[3:7],scalar_first=True).inv().apply((refs['world_positions'][index]-root[:3]).reshape(-1,3)).reshape(6,14,3)
            np.testing.assert_allclose(local,refs['root_positions'][index],atol=2e-6)
        observations=[json.loads(line) for line in (path/'observations.jsonl').read_text().splitlines()]
        for observation in observations:
            assert (path/observation['rgb']).exists() and (path/observation['depth']).exists()
            for pose in observation['poses'].values():
                assert pose['frame']=='world' and pose['timestamp']==observation['timestamp']
        records.append({'episode':str(path.relative_to(ROOT)),'policy_steps':n,'observations':len(observations),
                        'success':metrics['success'],'metrics_sha256':digest(path/'metrics.json')})
    summary=json.loads((args.cycle/'summary.json').read_text())
    assert summary['api_calls']==0 and summary['replayed_proposals']==1
    assert summary['history'][-1]['decision']['accepted'] is False
    assert json.loads((args.success/'metrics.json').read_text())['success'] is True
    result={'passed':True,'scope':'Artifact completeness, finite arrays, source identity, 50Hz timing, actual recorded frame transforms and selection integrity',
            'episodes':records,'physics_episodes':len(records),'source_files':len(expected_sources),
            'current_source_matches_recorded_release':expected_sources==current_sources,
            'controller_backend':'upstream eager ScaleBFM through official ScaleBridge methods',
            'new_api_calls_in_audit':0,'task_generalization_claimed':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'episodes':len(records),'source_files':len(expected_sources)}))


if __name__=='__main__':main()
