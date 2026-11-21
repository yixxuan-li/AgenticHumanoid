"""Verify physical merge regressions and preserve both versions' evidence."""
import json
import re
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from humanoid_skill_lab.integrations.reconstructed_scene import load_scene
from humanoid_skill_lab.provenance import source_hashes,digest


def main():
    scene=load_scene(ROOT/'outputs/worldlabs_scene')
    root=ROOT/'outputs/merged_system'
    reports={}
    current={k:v for k,v in source_hashes().items() if '/tests/' not in k}
    for name in ['atlas_reach','atlas_height','tabletop_regression']:
        directory=root/name
        manifest=json.loads((directory/'manifest.json').read_text())
        metrics=json.loads((directory/'metrics.json').read_text())
        assert manifest['result_status']=='executed' and metrics['physics_executed'] and metrics['success'],name
        assert not metrics['fallen'] and not metrics['forbidden_contacts']
        recorded={k:v for k,v in manifest['source_hashes'].items() if '/tests/' not in k}
        assert recorded==current,'Runtime changed after a merge regression was recorded'
        states=np.load(directory/'robot_states.npz')
        assert np.isfinite(states['qpos_qvel']).all()
        if name.startswith('atlas_'):
            assert manifest['source_scene']['identity']==scene['identity']
            assert metrics['manipulation_success'] is None
            np.testing.assert_allclose(states['qpos_qvel'][0,:3],scene['robot_start'])
            for line in (directory/'observations.jsonl').read_text().splitlines():
                assert not json.loads(line)['poses'],'Imported probe invented a task object binding'
        else:
            assert manifest['source_scene'] is None
        reports[name]={'success':metrics['success'],'metrics_sha256':digest(directory/'metrics.json'),
                       'manifest_sha256':digest(directory/'manifest.json'),
                       'right_hand_tracking_mean_m':metrics['right_hand_tracking_mean_m'],
                       'probe_diagnostics':metrics.get('probe_diagnostics'),
                       'final_target_distance_m':metrics['final_target_distance_m']}
    navigation_path=ROOT/'reports/merged_navigation_scalebfm_explore_mode0.json'
    navigation=json.loads(navigation_path.read_text())
    assert navigation['success'] and navigation['scene_fingerprint']==scene['identity']['sha256']
    tests=(root/'tests.log').read_text()
    passed=int(re.search(r'(\d+) passed',tests)[1])
    assert passed>=36 and not re.search(r'\d+ failed',tests)
    result={'passed':True,'tests_passed':passed,'new_api_calls':0,
            'scene_fingerprint':scene['identity']['sha256'],'source_scene_unchanged':True,
            'online_probes_and_tabletop':reports,
            'previous_navigation':{'success':True,'goal_error_m':navigation['goal_error_m'],'report_sha256':digest(navigation_path)},
            'scene_manipulation_claimed':False,'isaac_rerun':False,
            'scope':'Merged source and CLI; real navigation and online probes in the previous scene, plus tabletop regression'}
    (ROOT/'reports/merged_system_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
