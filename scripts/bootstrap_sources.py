"""Fetch the pinned upstream source/assets required by this reconstruction workspace."""
import argparse
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--isaac',action='store_true',help='Also fetch official G1 USD layers and Isaac Lab motor configuration');args=p.parse_args()
cache=Path('outputs/inspection/sources');cache.mkdir(parents=True,exist_ok=True)
sources=[('zengweishuai/ScaleBFM','abd6f17c02fe0baabc14709feb8d9ea4959aa621','vendor/ScaleBFM',[
        'LICENSE','README.md',
        'ScaleBridge/scalebridge/',
        'ScaleTrack/source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py',
        'ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py',
        'ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/config/g1_29dof/flat_env_cfg.py'])]
if args.isaac:sources[0][3].extend(['ScaleTrack/source/scaletrack/scaletrack/robots/','ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/configuration/','ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.usda','ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml'])
for repo,commit,output,prefixes in sources:
    tree=cache/(repo.split('/')[-1]+'_tree.json')
    if not tree.exists():subprocess.run(['curl','-sSfL','--retry','8','--retry-all-errors',f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1','-o',str(tree)],check=True)
    command=[sys.executable,'scripts/download_sources.py','--repo',repo,'--tree',str(tree),'--output',output]
    for prefix in prefixes:command+=['--prefix',prefix]
    subprocess.run(command,check=True)
print('Pinned sources ready. Model weights are separate; see README.md.')
