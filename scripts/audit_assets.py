"""Check simulator asset dependencies without assuming a machine-specific root."""
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pxr import UsdUtils,Sdf
import mujoco
from scene_reconstruction.provenance import scene_fingerprint
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/worldlabs_scene'));p.add_argument('--robot-usd',type=Path);p.add_argument('--report',type=Path,default=Path('reports/asset_dependency_audit.json'));a=p.parse_args();scene=a.scene.resolve();model=mujoco.MjModel.from_xml_path(str(scene/'scene.xml'));xml=ET.parse(scene/'scene.xml');files=[]
for element in xml.getroot().iter():
 if element.get('file'):files.append((scene/element.get('file')).resolve())
records=[]
for root,allow_runtime in [(scene/'scene.usda',False)]+([(a.robot_usd.resolve(),True)] if a.robot_usd else []):
 layers,assets,unresolved=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(root)));runtime=[str(x) for x in unresolved if allow_runtime and str(x)=='OmniPBR.mdl'];unexpected=[str(x) for x in unresolved if str(x) not in runtime];outside_assets=[str(x) for x in assets if Path(str(x)).is_absolute() and not Path(str(x)).is_relative_to(root.parent)];outside_layers=[l.identifier for l in layers if Path(l.identifier).is_absolute() and not Path(l.identifier).is_relative_to(root.parent)]
 records.append(dict(asset=str(root),layers=len(layers),non_layer_assets=len(assets),runtime_provided_material_modules=runtime,unresolved_non_runtime_assets=unexpected,outside_assets=outside_assets,outside_layers=outside_layers))
checks=dict(mujoco_model_loads=True,mujoco_files_local=all(x.is_relative_to(scene) for x in files),mujoco_files_exist=all(x.exists() for x in files),usd_dependencies_resolved_except_declared_runtime_modules=all(not r['unresolved_non_runtime_assets'] for r in records),usd_files_local=all(not r['outside_assets'] and not r['outside_layers'] for r in records))
r=dict(scene_fingerprint=scene_fingerprint(scene)['sha256'],checks=checks,all_passed=all(checks.values()),mujoco_meshes=model.nmesh,mujoco_external_files=len(files),usd=records,resolution_context='usd-core outside Isaac Kit; OmniPBR.mdl is supplied by the tested Isaac runtime',scope='File locality and resolver dependencies; runtime/physics validation is separate');a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(r,indent=2));print(json.dumps(dict(checks=checks,all_passed=r['all_passed']),indent=2))
if not r['all_passed']:sys.exit(1)
