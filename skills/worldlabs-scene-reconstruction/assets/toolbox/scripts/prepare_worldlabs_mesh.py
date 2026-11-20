"""Prepare one mesh extraction request from the completed refined Atlas run."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('outputs/worldlabs'));p.add_argument('--target-faces',type=int,default=300000);p.add_argument('--target-boxes',type=int,default=1000);p.add_argument('--margin',type=float,default=.35);a=p.parse_args()
out=a.workspace;operation=json.loads((out/'splats_operation.json').read_text())
if not operation.get('done') or operation.get('error'):raise RuntimeError('Wait for successful splat refinement before meshing')
splats=operation['response']['refinedSplats']
if not splats:raise RuntimeError('No refined splat asset returned')
count=min(splats,key=lambda key:abs(int(key)-2000000));asset=splats[count];points=np.load(out/'posed_points.npz')['points'];lo,hi=np.percentile(points,[.5,99.5],axis=0);lo-=a.margin;hi+=a.margin
request=dict(splatAsset={'assetId':asset['assetId']},cameras={'strategy':'auto_anchors'},orientedBoundingBox=dict(center=((lo+hi)/2).tolist(),halfExtents=((hi-lo)/2).tolist(),orientation=[0,0,0,1]),targetFaces=a.target_faces,textureMode='image_texture',textureSize=4096,colliderBoxes=dict(targetBoxes=a.target_boxes,heightAlignVoxels=1.))
(out/'mesh_input.json').write_text(json.dumps(request,indent=2));(out/'mesh_selection.json').write_text(json.dumps(dict(splat_count=int(count),splat_asset_id=asset['assetId'],bounds_source=f'Captured RGBD 0.5th to 99.5th percentiles, with {a.margin} native-unit margin',bounds_native=[lo.tolist(),hi.tolist()]),indent=2));print('Prepared textured mesh and upright collider request from',count,'splats')
