"""Resume Atlas scene stages through the official CLI without exposing credentials."""
import argparse,concurrent.futures,hashlib,json,os,re,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/worldlabs'
KEY_FILE=ROOT/'worldlabs'
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2));Path(p).chmod(0o600)
def cli(args,path,timeout=300):
 env=os.environ.copy()
 if not env.get('WLT_API_KEY') and KEY_FILE.exists():env['WLT_API_KEY']=KEY_FILE.read_text().splitlines()[0].strip()
 secret=env.get('WLT_API_KEY');redact=lambda value:value.replace(secret,'[REDACTED]') if secret else value
 for attempt in range(4):
  p=subprocess.run(['worldlabs',*args,'--json'],env=env,capture_output=True,text=True,timeout=timeout,cwd=ROOT)
  if not p.returncode:break
  if not any(s in p.stderr+p.stdout for s in ['fetch failed','This operation was aborted']):break
  time.sleep(2**attempt)
 Path(str(path)+'.stderr').write_text(redact(p.stderr))
 if p.returncode:
  Path(str(path)+'.error').write_text(redact(p.stdout))
  raise RuntimeError(f'World Labs CLI failed ({p.returncode}); private details saved beside {path.name}')
 data=json.loads(p.stdout);save(path,data);return data
def summary(d):return {k:d.get(k) for k in ['id','done','error']}|{'state':(d.get('metadata') or {}).get('state')}
def submit(task,request,stage):
 path=OUT/(stage+'_operation.json')
 req=OUT/(stage+'_request.json')
 if path.exists():
  existing=read(path);submitted=read(OUT/(stage+'_submission.json'))
  expected='agentic-scene-'+hashlib.sha256((task+json.dumps(request,indent=2)).encode()).hexdigest()[:40]
  if submitted['task']!=task or submitted['idempotency_key']!=expected:raise ValueError('This stage already has a different submitted request; use a new stage name')
  return existing
 save(req,request)
 key='agentic-scene-'+hashlib.sha256((task+req.read_text()).encode()).hexdigest()[:40]
 save(OUT/(stage+'_submission.json'),dict(task=task,idempotency_key=key))
 return cli(['tasks','run',task,'--request',str(req),'--idempotency-key',key],path)
def asset_ref(x):return {'assetId':x['assetId']}
def clean_frame(f):
 return dict(imageAsset=asset_ref(f['imageAsset']),camera=f['camera'],depth={k:asset_ref(v) for k,v in f['depth'].items() if isinstance(v,dict) and 'assetId' in v})
def main():
 global ROOT,OUT,KEY_FILE
 p=argparse.ArgumentParser();p.add_argument('action',choices=['pose','status','download','submit']);p.add_argument('--stage',default='pose');p.add_argument('--task');p.add_argument('--request',type=Path)
 p.add_argument('--project-root',type=Path,default=Path.cwd());p.add_argument('--workspace',type=Path,default=Path('outputs/worldlabs'));p.add_argument('--key-file',type=Path);p.add_argument('--frames',type=Path);p.add_argument('--resolution',type=int,nargs=2,default=[720,1280],metavar=('WIDTH','HEIGHT'));a=p.parse_args()
 if not re.fullmatch(r'[A-Za-z0-9_-]+',a.stage):p.error('Stage must be a simple name without path separators')
 if min(a.resolution)<=0:p.error('Resolution must be positive')
 ROOT=a.project_root.resolve();OUT=a.workspace.resolve();KEY_FILE=a.key_file.resolve() if a.key_file else ROOT/'worldlabs'
 OUT.mkdir(parents=True,exist_ok=True);OUT.chmod(0o700)
 if a.action=='pose':
  selected=read(a.frames or OUT/'selected_frames.json')
  if not selected:p.error('Select at least one source frame')
  selection_path=OUT/'selected_frames.json'
  if selection_path.exists() and read(selection_path)!=selected:raise ValueError('Workspace already has different source frames; choose a new workspace')
  save(selection_path,selected)
  def upload(item):
   source=ROOT/item['path'];cache=OUT/(source.stem+'_'+hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:10]+'_asset.json')
   legacy=OUT/(source.stem+'_asset.json')
   if legacy.exists():cache=legacy
   digest=hashlib.sha256(source.read_bytes()).hexdigest()
   if cache.exists():
    data=read(cache);previous=data.get('_local_source_sha256')
    if previous is not None and previous!=digest:raise ValueError('Frame content changed; preserve this run and choose a fresh output workspace')
    if previous is None and source.stat().st_mtime>cache.stat().st_mtime:raise ValueError('Frame is newer than its cached upload; inspect before reuse')
   else:data=cli(['assets','upload',str(source)],cache)
   data['_local_source_sha256']=digest;save(cache,data)
   print('Uploaded',source.name,flush=True);return {'imageAsset':{'assetId':data['id']}}
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:frames=list(ex.map(upload,selected))
  d=submit('images2PosedRGBD',dict(frames=frames,targetResolution=a.resolution,refinePoses=True),a.stage)
 elif a.action=='submit':
  if not a.task or not a.request:p.error('--task and --request are required')
  d=submit(a.task,read(a.request),a.stage)
 else:
  d=read(OUT/(a.stage+'_operation.json'))
  if a.action=='status':d=cli(['operations','get',d['id']],OUT/(a.stage+'_operation.json'))
  else:
   folder=OUT/a.stage;folder.mkdir(exist_ok=True)
   manifest=folder/'operation.json'
   if manifest.exists():
    existing=read(manifest)
    if existing['id']!=d['id']:raise ValueError('Download folder belongs to another operation')
    def verify(value):
     if isinstance(value,dict):
      if 'path' in value and not (folder/value['path']).is_file():raise FileNotFoundError(value['path'])
      for child in value.values():verify(child)
     elif isinstance(value,list):
      for child in value:verify(child)
    verify(existing.get('response'));print('Reusing complete local download',a.stage,flush=True)
   else:cli(['operations','download',d['id'],'--out',str(folder),'--timeout','900'],OUT/(a.stage+'_download.json'),timeout=950)
 print(json.dumps(summary(d)),flush=True)
 if d.get('error'):raise RuntimeError('Atlas operation failed; see saved operation')
if __name__=='__main__':main()
