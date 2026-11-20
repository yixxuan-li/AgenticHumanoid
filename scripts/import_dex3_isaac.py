"""Convert the pinned official ScaleBridge G1/Dex3 MJCF to a local Isaac USD.

Uses the installed Isaac Sim MJCF importer; retains source inertias and records
joint/link inventory. Simulator-specific validation is still required afterward.
"""
import argparse,json,os,sys,shutil,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/robots/g1_dex3'));p.add_argument('--fast-exit',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);app=launcher.app

def main():
 from isaacsim.core.utils.extensions import enable_extension
 enable_extension('isaacsim.asset.importer.mjcf')
 import omni.kit.commands
 from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
 folder=a.output.resolve();folder.mkdir(parents=True,exist_ok=True);source=Path('vendor/ScaleBFM/ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof_dex3.xml').resolve();tree=ET.parse(source);root=tree.getroot();root.find('compiler').set('meshdir','meshes');shutil.copytree(source.parent/'meshes',folder/'meshes',dirs_exist_ok=True)
 for world in root.findall('worldbody'):
  for element in list(world):
   if element.tag in ['geom','light','camera']:world.remove(element)
  for body in world.findall('body'):
   if body.find('freejoint') is not None:body.set('pos','0 0 0');body.set('quat','1 0 0 0')
 for asset in root.findall('asset'):
  for texture in list(asset):
   if texture.tag=='texture':asset.remove(texture)
  for material in asset.findall('material'):
   for key in ['texture','texrepeat','texuniform']:material.attrib.pop(key,None)
 for geom in root.iter('geom'):
  if 'type' not in geom.attrib:geom.set('type','mesh' if geom.get('mesh') else 'sphere')
 for world in list(root.findall('worldbody')):
  if not len(world):root.remove(world)
 for parent in root.iter():
  for child in list(parent):
   if child.tag=='body' and not list(child):parent.remove(child)
 input_path=folder/'g1_dex3_import.xml';tree.write(input_path,encoding='unicode')
 ok,config=omni.kit.commands.execute('MJCFCreateImportConfig')
 if not ok:raise RuntimeError('MJCF import configuration unavailable')
 config.set_fix_base(False);config.set_import_inertia_tensor(True);config.set_self_collision(True);config.set_make_default_prim(True);config.set_visualize_collision_geoms(False)
 destination=folder/'g1_dex3.usd';ok,path=omni.kit.commands.execute('MJCFCreateAsset',mjcf_path=str(input_path),import_config=config,prim_path='/G1',dest_path=str(destination))
 if not ok:raise RuntimeError('MJCF importer failed')
 for _ in range(10):app.update()
 stage=Usd.Stage.Open(str(destination));removed=[]
 for prim in list(stage.Traverse()):
  if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.GetChildren():removed.append(str(prim.GetPath()));prim.SetActive(False)
  if prim.HasAPI(UsdPhysics.CollisionAPI):
   collision=PhysxSchema.PhysxCollisionAPI.Apply(prim);collision.CreateContactOffsetAttr(.002);collision.CreateRestOffsetAttr(0.)
 stage.GetRootLayer().Save();joints=[str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)];rigid=[str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)];articulation=[str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
 report=dict(source=str(source),import_xml=str(input_path),usd=str(destination),root_prim=path,revolute_joints=len(joints),rigid_links=len(rigid),articulation_roots=articulation,removed_empty_articulation_containers=removed,joints=joints,scale_meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),root_pose_normalized=True,contact_offset_m=.002,rest_offset_m=0.,validation='Import inventory only; body control and hand contacts still require tests')
 (folder/'import_audit.json').write_text(json.dumps(report,indent=2));print('DEX3_IMPORT '+json.dumps(report),flush=True)
 if len(joints)!=43:raise RuntimeError(f'Expected 29 body + 14 hand joints, got {len(joints)}')
try:main()
except BaseException:
 import traceback
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
 if a.fast_exit:os._exit(1)
 app.close(wait_for_replicator=False);raise
else:
 if a.fast_exit:sys.stdout.flush();sys.stderr.flush();os._exit(0)
 app.close(wait_for_replicator=False)
