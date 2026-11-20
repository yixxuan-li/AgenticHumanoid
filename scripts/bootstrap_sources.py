"""Fetch the pinned upstream source/assets required by this reconstruction workspace."""
import argparse
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--reconstruction-only',action='store_true',help='Fetch VGGT only');p.add_argument('--isaac',action='store_true',help='Also fetch official G1 USD layers and Isaac Lab motor configuration');args=p.parse_args()
cache=Path('outputs/inspection/sources');cache.mkdir(parents=True,exist_ok=True)
sources=[('facebookresearch/vggt','a288dd0f14786c93483e45524328726ab7b1b4ce','vendor/vggt',['vggt/','LICENSE','README.md']),
    ('zengweishuai/ScaleBFM','abd6f17c02fe0baabc14709feb8d9ea4959aa621','vendor/ScaleBFM',[
        'LICENSE','README.md',
        'ScaleBridge/scalebridge/data/robot/g1_29dof/','ScaleBridge/scalebridge/data/motion/g1_29dof/squat.npz',
        'ScaleTrack/source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py',
        'ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py',
        'ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/config/g1_29dof/flat_env_cfg.py'])]
if args.isaac:sources[1][3].extend(['ScaleTrack/source/scaletrack/scaletrack/robots/','ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/configuration/','ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.usda','ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml'])
if args.reconstruction_only:sources=sources[:1]
for repo,commit,output,prefixes in sources:
    tree=cache/(repo.split('/')[-1]+'_tree.json')
    if not tree.exists():subprocess.run(['curl','-sSfL','--retry','8','--retry-all-errors',f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1','-o',str(tree)],check=True)
    command=[sys.executable,'scripts/download_sources.py','--repo',repo,'--tree',str(tree),'--output',output]
    for prefix in prefixes:command+=['--prefix',prefix]
    subprocess.run(command,check=True)
print('Pinned sources ready. Model weights are separate; see README.md.')
