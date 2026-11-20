"""Uniformly rescale a scene and photographic meshes, preserving robot-free assets.

A supplied measurement is a single correspondence, not a scene-wide accuracy
certificate. --scale-factor is explicitly hypothetical and never marks calibration.
"""
import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from scene_reconstruction.export import values
from scene_reconstruction.provenance import scene_fingerprint

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
g=p.add_mutually_exclusive_group(required=True);g.add_argument('--calibration',type=Path);g.add_argument('--scale-factor',type=float)
a=p.parse_args()
if a.output.exists():p.error('Output must be a new directory to preserve previous scene/evidence')
manifest=json.loads((a.scene/'manifest.json').read_text());anchor=json.loads(a.calibration.read_text()) if a.calibration else None
if anchor:
 lengths=np.array([anchor['measured_length_m'],anchor['reconstructed_length_m']],float)
 if not np.isfinite(lengths).all() or np.min(lengths)<=0:p.error('Both corresponding lengths must be finite positive meters')
 if not str(anchor.get('description','')).strip():p.error('Describe the measured physical segment')
factor=anchor['measured_length_m']/anchor['reconstructed_length_m'] if anchor else a.scale_factor
if not np.isfinite(factor) or not .25<factor<4:p.error('Check positive meter units and corresponding physical segment; factor must be between .25 and 4')
source_identity=scene_fingerprint(a.scene)['sha256'];shutil.copytree(a.scene,a.output)
root=ET.parse(a.output/'scene.xml');r=root.getroot()
def multiply(element,key,power=1):
 if element.get(key) is not None:element.set(key,values(np.fromstring(element.get(key),sep=' ')*factor**power))
for element in r.iter():
 if element.tag in ['body','geom','site','camera','light','inertial']:multiply(element,'pos')
 if element.tag in ['geom','site']:multiply(element,'size');multiply(element,'margin');multiply(element,'gap')
 if element.tag in ['geom','inertial']:multiply(element,'mass',3)
 if element.tag=='inertial':multiply(element,'diaginertia',5);multiply(element,'fullinertia',5)
 if element.tag=='mesh':
  if element.get('file'):element.set('scale',values(np.fromstring(element.get('scale','1 1 1'),sep=' ')*factor))
  else:multiply(element,'vertex')
 if element.tag=='statistic':multiply(element,'center');multiply(element,'extent')
root.write(a.output/'scene.xml',encoding='unicode')
stage=Usd.Stage.Open(str(a.output/'scene.usda'))
for prim in stage.Traverse():
 if prim.HasCustomDataKey('scaleStatus'):prim.SetCustomDataByKey('scaleStatus','single_measured_anchor' if anchor else 'scaled_hypothesis')
 for attribute in prim.GetAttributes():
  name=attribute.GetName();value=attribute.Get()
  if value is None:continue
  if name.startswith('xformOp:translate') or name=='physics:localPos0' or name=='physics:localPos1':attribute.Set(value*factor)
  elif name.startswith('xformOp:scale'):
   # Exporter scales only unit native collision cubes, not parent assemblies.
   attribute.Set(value*factor)
  elif name=='points':attribute.Set([v*factor for v in value])
  elif name in ['radius','height','physics:contactOffset','physics:restOffset']:attribute.Set(value*factor)
  elif name=='size' and not prim.HasAttribute('xformOp:scale'):attribute.Set(value*factor)
  elif name=='physics:mass':attribute.Set(value*factor**3)
  elif name=='physics:diagonalInertia':attribute.Set(value*factor**5)
stage.GetRootLayer().Save()
for key in ['dimensions','source_geometry_shift_m','main_room_y_range']:
 if key in manifest:manifest[key]=(np.array(manifest[key])*factor).tolist()
manifest['desk_height']*=factor;manifest['global_scale']=manifest.get('global_scale',1.)*factor
if 'cameras' in manifest:manifest['cameras']=[[name,(np.array(pos)*factor).tolist(),(np.array(target)*factor).tolist()] for name,pos,target in manifest['cameras']]
for key in ['start','goals']:
 if key in manifest.get('navigation',{}):manifest['navigation'][key]=(np.array(manifest['navigation'][key])*factor).tolist()
for b in manifest['bodies']:
 b['pos']=(np.array(b['pos'])*factor).tolist();b['mass']*=factor**3
 for part in b['parts']:
  for key in ['pos','size']:part[key]=(np.array(part[key])*factor).tolist()
 if 'recovered_surface_bounds' in b:b['recovered_surface_bounds']=(np.array(b['recovered_surface_bounds'])*factor).tolist()
 if 'dimension_uncertainty_m' in b:b['dimension_uncertainty_m']=(np.array(b['dimension_uncertainty_m'])*factor).tolist()
for bench in manifest.get('benchmarks',{}).values():
 for key in ['approach','block']:
  if key in bench:
   point=np.array(bench[key])*factor
   if key=='approach' and 'controller_profile' in bench:point[2]=bench['controller_profile']['robot_root_height_m']
   bench[key]=point.tolist()
manifest['scale_status']='single_measured_anchor' if anchor else 'scaled_hypothesis'
manifest['metric_anchor']=dict(**anchor,measured=True) if anchor else dict(type='hypothetical sensitivity scaling',factor=factor,measured=False)
manifest['rescaled_from_sha256']=source_identity
manifest['validation_status']='Re-run dynamics, rendering and robot checks for this scene identity'
if manifest.get('room_envelope'):manifest['source_room_envelope']=manifest.pop('room_envelope')
for name in ['room_geometry_m','furniture_fit_m']:
 if name in manifest.get('uncertainty',{}):manifest['uncertainty'][name]*=factor
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
(a.output/'scale_transform.json').write_text(json.dumps(dict(factor=factor,source_sha256=source_identity,output_sha256=scene_fingerprint(a.output)['sha256'],measurement_supplied=bool(anchor)),indent=2))
print(f'Exported {a.output}: factor {factor:.6g}; status {manifest["scale_status"]}; validation required',flush=True)
