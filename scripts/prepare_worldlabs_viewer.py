"""Bundle a local-only Spark viewer; no API credentials or signed URLs included."""
import argparse,json,os,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('outputs/worldlabs'));p.add_argument('--node-modules',type=Path,default=Path('/tmp/worldlabs_viewer/node_modules'));p.add_argument('--output',type=Path,default=Path('outputs/worldlabs_viewer'));a=p.parse_args();root=a.workspace;out=a.output;out.mkdir(parents=True,exist_ok=True)
def copy(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
copy(Path(__file__).resolve().parents[1]/'scene_reconstruction/atlas_viewer.html',out/'index.html')
three=a.node_modules/'three';spark=a.node_modules/'@sparkjsdev/spark'
for name in ['three.module.js','three.core.js']:copy(three/'build'/name,out/'vendor/three'/name)
for name in ['controls/OrbitControls.js','postprocessing/Pass.js']:copy(three/'examples/jsm'/name,out/'vendor/three/addons'/name)
copy(spark/'dist/spark.module.js',out/'vendor/spark/spark.module.js');copy(three/'LICENSE',out/'vendor/three/LICENSE');copy(spark/'LICENSE',out/'vendor/spark/LICENSE')
selection=json.loads((root/'mesh_selection.json').read_text());source=root/'splats'/f"refinedSplats_{selection['splat_count']}.spz";target=out/'room.spz'
if target.exists() and target.read_bytes()!=source.read_bytes():target.unlink()
if not target.exists():
 try:os.link(source,target)
 except OSError:shutil.copy2(source,target)
frames=[]
for i,f in enumerate(json.loads((root/'pose_operation.json').read_text())['response']['frames']):frames.append(dict(name=f'source_{i:03d}',camera=f['camera']))
curation=json.loads((root/'curation.json').read_text()) if (root/'curation.json').exists() else {}
for n,item in enumerate(curation.get('retained_generated',[])):
 f=json.loads((root/(item['stage']+'_operation.json')).read_text())['response']['frames'][item['index']];frames.append(dict(name=f'novel_{n:03d}',camera=f['camera']))
extra=root/'heldout_cameras.json'
if extra.exists():frames+=json.loads(extra.read_text())
(out/'cameras.json').write_text(json.dumps(frames,indent=2));(out/'README.txt').write_text('Run from the project root:\npython3 -m http.server 8869 --bind 127.0.0.1 --directory outputs/worldlabs_viewer\nOpen http://127.0.0.1:8869/ in a browser with WebGL2.\nThis folder is self-contained. It contains no API key or private download URLs.\nSplats are visual geometry; physics is provided by the separate MJCF/USD export.\nScale and unseen areas remain estimates.\n');print('Bundled local Spark viewer:',out)
