"""Create portable local delivery archives and verify a relocated scene."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
from pxr import Usd,UsdUtils
from scene_reconstruction.provenance import scene_fingerprint
root=Path.cwd();scene=Path('outputs/final_scene');delivery=Path('deliverables');delivery.mkdir(exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def files(folder):return sorted(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
sources=[Path('README.md'),Path('requirements.txt'),*files('scripts'),*files('scene_reconstruction'),*files('skills/reconstruct-video-scene')]
selected=['native_viewer_startup.json','RESULTS.md','final_summary.json','task_state.json','final_scene_physics.json','final_isaaclab_validation.json','final_placement_validation.json','final_explore_scalebfm_explore_mode0.json','dex3_pick_final_pick.json','isaac_dex3_pick_final_sequence.json','skill_agent_final_plan.json','skill_agent_final_pick.json','skill_agent_final_explore.json','final_appearance_validation.json','final_nav_scalebfm_navigate_mode0.json','final_squat_scalebfm_squat_mode7.json','final_native_nav_scalebfm_isaac_navigate.json','final_native_squat_scalebfm_isaac_squat.json','robot_transfer_audit.json','asset_dependency_audit.json','scale_sensitivity.json','scale095_physics.json','scale105_physics.json','dex3_pick_scale105_pick.json','skill_reuse_validation.json','skill_final_scene_settling.json','skill_native_isaac_validation.json','skill_hinge_export_smoke.json','skill_packaged_roi_validation.json','skill_fusion_smoke.json','runtime_versions.json','dense_geometry_full.json','room_plane_fit.json','checkpoint_provenance.json','checkpoint_integrity_and_cleanup.json']
for name in selected:
 if not (Path('reports')/name).exists():raise FileNotFoundError(name)
summary=json.loads(Path('reports/final_summary.json').read_text());identity=scene_fingerprint(scene)['sha256']
if summary['scene_fingerprint']!=identity or not summary['all_executed_final_checks_passed']:raise ValueError('Final evidence must pass and match the delivered scene')
provenance=dict(scene_fingerprint=identity,metric_accuracy_verified=False,input_video_sha256=sha('data/scene/scene.mp4'),source_files_sha256={str(p):sha(p) for p in sources},selected_reports_sha256={name:sha(Path('reports')/name) for name in selected},upstream_commits={'ScaleBFM':'abd6f17c02fe0baabc14709feb8d9ea4959aa621','VGGT':'a288dd0f14786c93483e45524328726ab7b1b4ce'},weights={'scalebfm_medium_sha256':'88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3','scalebfm_metadata_sha256':'3024f18136314fade2676e22d6eafb1a8b2befa388da55b0885a53fcb1368d16','vggt_sha256':'d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'},weight_verification='Full weight hashes verified in checkpoint_integrity_and_cleanup.json; archives fetch weights separately')
Path('reports/delivery_provenance.json').write_text(json.dumps(provenance,indent=2))
def archive(path,entries):
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for source,target in entries:z.write(source,target)
 with zipfile.ZipFile(path) as z:
  if z.testzip() is not None:raise ValueError('ZIP CRC failure: '+str(path))
scene_entries=[(p,Path('scene')/p.relative_to(scene)) for p in files(scene)]
archive(delivery/'scene_bundle.zip',scene_entries)
skill=Path('skills/reconstruct-video-scene');archive(delivery/'reconstruct-video-scene.zip',[(p,Path(skill.name)/p.relative_to(skill)) for p in files(skill)])
extra=[*files(scene),*files('outputs/robots/g1_dex3_v4'),Path('data/scene/scene.mp4'),Path('data/scene/calibration.example.json'),Path('reports/delivery_provenance.json')]
for name in selected:extra.append(Path('reports')/name)
for folder in ['outputs/controller/pick/final_pick','outputs/controller/isaac_pick/final_sequence','outputs/controller/final_explore']:
 extra.extend(p for p in files(folder) if p.suffix in ['.mp4','.png','.csv'])
extra.extend(Path('outputs/inspection').glob('frame_*.jpg'))
for name in ['scene_features.json','geometry_audit.json','inference_report.json']:extra.append(Path('outputs/geometry_full')/name)
project_entries=[(p,p) for p in sorted(set(sources+extra))]
relocated=Path('/tmp/agentic_scene_delivery_check')/identity[:12]
if relocated.exists():shutil.rmtree(relocated)
with zipfile.ZipFile(delivery/'scene_bundle.zip') as z:z.extractall(relocated)
relocated=relocated/'scene';model=mujoco.MjModel.from_xml_path(str(relocated/'scene.xml'));stage=Usd.Stage.Open(str(relocated/'scene.usda'));layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(relocated/'scene.usda'))
checks=dict(zip_crc=True,relocated_mujoco_load=model.nq>0,relocated_usd_load=bool(stage),relocated_usd_assets_resolved=not unresolved,relocated_scene_identity=scene_fingerprint(relocated)['sha256']==identity)
embedded=dict(scene_fingerprint=identity,checks=checks,all_passed=all(checks.values()),archives={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [delivery/'scene_bundle.zip',delivery/'reconstruct-video-scene.zip']},project_archive_sha256='Recorded outside this archive to avoid a self-referential checksum',skill_installation='Packaged only; not installed globally',metric_accuracy_verified=False)
embedded_path=Path('/tmp/agentic_scene_embedded_delivery_integrity.json');embedded_path.write_text(json.dumps(embedded,indent=2))
archive(delivery/'reconstruction_project.zip',project_entries+[(embedded_path,Path('reports/delivery_integrity.json'))])
report=dict(scene_fingerprint=identity,checks=checks,all_passed=all(checks.values()),archives={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(delivery.glob('*.zip'))},relocation_test_directory=str(relocated),skill_installation='Packaged only; not installed globally',metric_accuracy_verified=False)
Path('reports/delivery_integrity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if not report['all_passed']:raise RuntimeError('Delivery portability failed')
