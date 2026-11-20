"""Download a pinned, selected subset of an upstream source tree."""
import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--tree',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',action='append',default=[]);p.add_argument('--max-file-mb',type=float,default=40)
args=p.parse_args();tree=json.loads(args.tree.read_text());revision=tree['sha']
files=[t for t in tree['tree'] if t['type']=='blob' and any(t['path'].startswith(prefix) for prefix in args.prefix) and t.get('size',0)<args.max_file_mb*1024**2]
def fetch(t):
    path=args.output/t['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==t['size']:return
    if t['size']==0:path.touch();return
    tmp=path.with_suffix(path.suffix+'.partial')
    subprocess.run(['curl','-sSfL','--retry','8','--retry-all-errors','--retry-delay','2','--connect-timeout','20','--max-time','300',f"https://raw.githubusercontent.com/{args.repo}/{revision}/{t['path']}",'-o',str(tmp)],check=True)
    if tmp.stat().st_size!=t['size']:raise RuntimeError(f'Unexpected length: {path}')
    tmp.replace(path);print(str(path),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(fetch,files))
(args.output/'UPSTREAM_REVISION.json').write_text(json.dumps(dict(repo=args.repo,commit=revision,files_in_last_fetch=len(files)),indent=2))
