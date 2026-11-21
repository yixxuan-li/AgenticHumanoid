"""Package maintained sources and selected evidence; exclude weights and credentials."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--merged',action='store_true',help='Include the previous tracked reconstruction version and merged validation')
    args=parser.parse_args()
    if args.output is None:
        args.output=ROOT/'deliverables'/('agentic_humanoid_merged.tar.gz' if args.merged else 'humanoid_skill_lab_prototype.tar.gz')
    files=set()
    def include(path):
        p=ROOT/path
        candidates=p.rglob('*') if p.is_dir() else [p]
        for item in candidates:
            if item.is_file() and '__pycache__' not in item.parts and item.suffix!='.pyc':
                if not item.resolve().is_relative_to(ROOT):raise ValueError('External symlinks cannot be packaged')
                files.add(item)
    for name in ['.gitignore','README.md','pyproject.toml','uv.lock','humanoid_skill_lab','docs',
                 'scene_reconstruction/__init__.py','scene_reconstruction/scalebfm.py',
                 'scripts/bootstrap_sources.py','scripts/download_sources.py','scripts/download_ranges.py',
                 'scripts/package_skill_lab.py','scripts/audit_skill_lab_delivery.py']:
        include(name)
    for item in (ROOT/'reports').glob('skill_lab*'):
        if item.name!='skill_lab_bundle.json':include(item.relative_to(ROOT))
    if args.merged:
        # Include the previous maintained version as well as all new modules.
        # Git's tracked-file list excludes private ignored downloads and keys.
        previous=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
        for name in previous:
            if name and Path(name).name not in {'worldlabs','skill_lab_bundle.json','merged_system_bundle.json'} and not Path(name).name.startswith('.env'):
                include(name)
        include('scripts/package_agentic_humanoid.py')
        include('scripts/validate_merged_system.py')
        include('outputs/merged_system')
        include('outputs/controller/merged_navigation')
        for item in (ROOT/'reports').glob('*merged_navigation*.json'):include(item.relative_to(ROOT))
        include('reports/merged_system_validation.json')
    runs=ROOT/'outputs/skill_lab'
    for name in ['final_audit_cycle_v2','isolated_calibrated_push_final','isolated_probe_reach','probe_stand_001',
                 'probe_height_001','replay_004','frozen_test_001','calibrated_onboard_002',
                 'exploration_003/proposal_00','exploration_003/proposal_00_context.json',
                 'isolated_environment.json','local_dependency_reuse.json','isolated_tests_final.log']:
        include(Path('outputs/skill_lab')/name)
    for name in ['exploration_002','exploration_003','ablation_fixed','ablation_skill_only',
                 'ablation_symbolic_only','ablation_joint','calibrated_validation']:
        for filename in ['experiment.json','summary.json','baseline_validation.json']:
            p=runs/name/filename
            if p.exists():include(p.relative_to(ROOT))
    # Detect likely accidentally pasted OpenAI credentials without displaying them.
    for p in files:
        if p.suffix in {'.py','.json','.md','.log','.txt','.csv','.toml'}:
            if re.search(rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}',p.read_bytes()):
                raise RuntimeError('Possible credential material detected; archive was not created')
    manifest={'scope':('Merged reconstruction and skill exploration sources with physical regression evidence' if args.merged else 'Maintained prototype and selected physical evidence; full experiment records remain in workspace'),
              'project_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'excluded':['API credentials','downloaded checkpoint weights','vendor source tree','virtual environments'],
              'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,'w:gz',compresslevel=5) as archive:
        for p in sorted(files):archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        encoded=(json.dumps(manifest,indent=2)+'\n').encode()
        info=tarfile.TarInfo('BUNDLE_MANIFEST.json');info.size=len(encoded)
        archive.addfile(info,io.BytesIO(encoded))
    sha=hashlib.sha256(args.output.read_bytes()).hexdigest()
    record={'archive':str(args.output.relative_to(ROOT)),'sha256':sha,'files':len(files),'bytes':args.output.stat().st_size}
    record_path=ROOT/'reports'/('merged_system_bundle.json' if args.merged else 'skill_lab_bundle.json')
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
