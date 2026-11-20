"""Create a provenance-bound Atlas handover report and local delivery ZIP."""
import hashlib,json,sys,zipfile,re,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
from pxr import Usd,UsdUtils
from scene_reconstruction.provenance import scene_fingerprint
scene=Path('outputs/worldlabs_scene');root=Path('outputs/worldlabs');viewer=Path('outputs/worldlabs_viewer');identity=scene_fingerprint(scene)['sha256'];mf=json.loads((scene/'manifest.json').read_text())
def read(name,bound=True):
 d=json.loads((Path('reports')/name).read_text())
 if bound and d.get('scene_fingerprint')!=identity:raise ValueError(name+' belongs to another scene')
 return d
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
mj=read('worldlabs_mujoco_validation.json');native=read('worldlabs_isaac_validation.json');export=read('worldlabs_export_validation.json');dependencies=read('worldlabs_asset_dependencies.json');nav=read('worldlabs_final_explore_scalebfm_explore_mode0.json');native_nav=read('worldlabs_native_explore_scalebfm_isaac_explore.json');appearance=read('worldlabs_appearance_validation.json');splat=read('worldlabs_splat_appearance_validation.json',False);ui=read('worldlabs_viewer_validation.json',False)
checks=dict(mujoco_physics=mj['all_passed'],physx_physics=native['all_passed'],generic_export=export['all_passed'],asset_dependencies=dependencies['all_passed'],mujoco_navigation=nav['success'],physx_navigation=native_nav['success'],spark_viewer=ui['all_passed'])
if not all(checks.values()):raise RuntimeError('Final Atlas checks are incomplete or failing')
mesh_ssim=[r['gray_ssim'] for r in appearance['results'] if r['status']=='rendered'];plan=json.loads((root/'final_navigation_plan.json').read_text());curation=json.loads((root/'curation.json').read_text());constraints=json.loads((root/'mesh_constraints.json').read_text());spz=root/'splats/refinedSplats_2000000.spz';assert sha(spz)==splat['splat_sha256'];assert sha(viewer/'room.spz')==splat['splat_sha256']
summary=dict(updated_local=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),scene=str(scene),scene_fingerprint=identity,all_executed_checks_passed=True,checks=checks,native_splat_count=2000000,worldlabs_operations=mf['worldlabs_operations'],additional_mesh_trial=json.loads((root/'mesh_refined_operation.json').read_text())['id'],dimensions_provisional_m=mf['dimensions'],scale_status=mf['scale_status'],metric_accuracy_verified=False,assumed_table_height_m=.74,static_collider_boxes=json.loads((scene/'atlas_export.json').read_text())['collider_count'],movable_bodies='One added 5 cm / 80 g benchmark cube; original furniture is static',appearance=dict(native_splat_mean_gray_ssim=splat['mean_gray_ssim'],simulator_mesh_mean_gray_ssim=float(np.mean(mesh_ssim)),review='Six actual video frames excluded from reconstruction input and textures; learned-depth camera estimation for review only'),navigation=dict(route_length_m=plan['path_length_m'],start_m=mf['navigation']['start'],goal_m=nav['planned_goal_m'],mujoco_goal_error_m=nav['goal_error_m'],physx_goal_error_m=native_nav['goal_error_m'],simulated_seconds=38,state_access='Known estimated collision map and simulator proprioception',wbc='Released ScaleBFM medium, mode 0, 50 Hz references and 200 Hz physics'),remaining_limitations=['Absolute dimensions need a real measurement.','Native splats have better appearance than the extracted simulator mesh.','Glass, thin chairs and occlusion boundaries retain mesh defects.','Furniture articulation, physical masses and friction were not recovered.','No new grasp trial is claimed for this static Atlas scene.']);Path('reports/worldlabs_summary.json').write_text(json.dumps(summary,indent=2))
text=f'''# World Labs Atlas reconstruction

The native Atlas splat and simulator exports are ready for review. **Scale is provisional, and original furniture is static.**

- [Open local Spark viewer](http://127.0.0.1:8870/) or serve `outputs/worldlabs_viewer/` locally.
- [Native two-million-splat SPZ](../outputs/worldlabs_viewer/room.spz).
- [MuJoCo MJCF](../outputs/worldlabs_scene/scene.xml), [USD](../outputs/worldlabs_scene/scene.usda), [manifest](../outputs/worldlabs_scene/manifest.json).
- [Selected textured GLB](../outputs/worldlabs/constrained_mesh.glb).
- [Two-engine navigation video](../outputs/worldlabs_scene/navigation_both_engines.mp4).
- [Actual held-out video / native SPZ / final mesh comparison](../outputs/worldlabs_scene/appearance_comparison.jpg).
- [Mesh source comparisons](../outputs/worldlabs_scene/source_comparison.jpg), [novel mesh views](../outputs/worldlabs_scene/novel_views.jpg), [native RTX view](../outputs/worldlabs_scene/isaaclab_source_006.png).
- [Portable local bundle](../deliverables/worldlabs_scene_bundle.zip), [checksums and relocation check](worldlabs_delivery_integrity.json).

## Construction

Used the official [Atlas quickstart](https://docs.atlas.worldlabs.ai/quickstart.md), [atlas-splats skill](https://github.com/worldlabsai/skills/tree/main/skills/atlas-splats), and [worldlabs-cli skill](https://github.com/worldlabsai/skills/tree/main/skills/worldlabs-cli). CLI 0.6.0 and those skills are installed. The API key was read from the authorized local file into a child-process environment; it is ignored by Git and absent from viewer assets and delivery archives.

1. Selected 12 sharp, overlapping real frames across the complete 17-second recording. Uploaded them as private project assets.
2. Ran `images2PosedRGBD` jointly at 720 × 1280, with accepted pose refinement. Returned RGB, confidence, depth and cameras remain paired.
3. Generated two rounds of three novel RGBD views from camera centers supported as free space by the observed depth. Kept three generated views, rejected inconsistent chair layouts, and masked an invented computer cart. All 15 final frames carry aligned masks, with full-white masks for unaffected images.
4. Ran one successful refined `posedImages2Splats` reconstruction, returning 2,000,000 splats. The operation is `{mf['worldlabs_operations']['splats']}`. Source and novel views were rendered in Spark on the RTX 5090.
5. Extracted a textured mesh and upright collider boxes with `splats2Mesh`. A second stereo/trimmed extraction was reviewed and retained separately: it opened additional holes in walls and glass. The selected simulator scene uses the more complete base extraction plus documented visual constraints.
6. Fitted the table outline to decoded refined-splat points, rectified curated input RGB onto its assumed 0.74 m support plane, and removed unsupported gray bridge surfaces beneath it. The tabletop atlas has {constraints['table_texture_coverage_fraction']*100:.1f}% projected coverage and {constraints['inpainted_texture_fraction']*100:.1f}% inpainted occlusion coverage. Low floor vertices were constrained to the fitted plane. These are authored priors, not recovered CAD.
7. Converted the Y-up Atlas world to Z-up, applying the floor rotation, provisional scale and translation to geometry, cameras and collider boxes together. The room has {summary['static_collider_boxes']} fixed fitted boxes, an inferred continuous floor, and one added free benchmark cube. Low fitted floor patches are consolidated into the flat support slab. Photo appearance is baked/unlit.

The apparent envelope is approximately {mf['dimensions'][0]:.2f} × {mf['dimensions'][1]:.2f} × {mf['dimensions'][2]:.2f} **provisional metres**. The table-height anchor is an assumption. Floor ROI fitting retained {mf['metric_anchor']['floor_inlier_fraction']*100:.1f}% of sampled points within a 0.025-native-unit fitting tolerance. Neither Atlas nor this residual proves physical measurement accuracy.

## Executed checks

| Check | Result | Evidence |
|---|---|---|
| MuJoCo support, drop, friction, mass and penetrations | {sum(mj['checks'].values())}/{len(mj['checks'])} pass | [Report](worldlabs_mujoco_validation.json) |
| Native PhysX support, drop, mass and finite states | {sum(native['checks'].values())}/{len(native['checks'])} pass | [Report](worldlabs_isaac_validation.json) |
| Generic exporter and dependency checks | Pass | [Exporter](worldlabs_export_validation.json), [dependencies](worldlabs_asset_dependencies.json) |
| MuJoCo ScaleBFM exploration | Pass; {nav['goal_error_m']*100:.1f} cm final goal error | [Report](worldlabs_final_explore_scalebfm_explore_mode0.json) |
| PhysX ScaleBFM exploration | Pass; {native_nav['goal_error_m']*100:.1f} cm final goal error | [Report](worldlabs_native_explore_scalebfm_isaac_explore.json) |
| Local native-SPZ viewer | Load, view switch and WASD movement pass | [Report](worldlabs_viewer_validation.json) |

Both navigation runs follow the same {plan['path_length_m']:.3f} m A* route from ({mf['navigation']['start'][0]:.2f}, {mf['navigation']['start'][1]:.2f}) to ({nav['planned_goal_m'][0]:.2f}, {nav['planned_goal_m'][1]:.2f}), with 0.32 m planner clearance. Each runs for 38 simulated seconds without falling, root teleportation or reference forcing. MuJoCo also audits furniture contacts above 1 N; the native PhysX navigation trial does not provide that contact audit. This is a map/state-based agent and proprioceptive WBC, not learned visual navigation.

Six additional actual video frames, excluded from input images and reconstruction textures, provide descriptive appearance review. Camera poses are fitted only for review using SIFT and learned-depth PnP. Mean grayscale SSIM is **{splat['mean_gray_ssim']:.3f} for the native splat** and **{np.mean(mesh_ssim):.3f} for the final mesh**. These are interpolated viewpoints along the capture, not unseen backsides or a metric-depth benchmark. [Splat measurements](worldlabs_splat_appearance_validation.json), [mesh measurements](worldlabs_appearance_validation.json).

The native splat is the better visual representation. The simulator mesh has visible thin-chair, glass, blur and occlusion defects. The unseen entrance door, air conditioner and other uncaptured surfaces are model hypotheses. Furniture is fixed in this export; Atlas did not recover free joints, cabinet mechanisms, real masses, friction or relightable materials. The added 5 cm / 80 g cube is absent from the video. Tests validate its assigned simulation behavior. No grasp success is claimed for this static Atlas room.

## Reusable skill

The portable [worldlabs-scene-reconstruction skill](../skills/worldlabs-scene-reconstruction/SKILL.md) packages configurable workspaces, cached API stages, measured-anchor calibration, export and checks. [Offline portability checks](worldlabs_skill_validation.json) passed using a separate project and synthetic geometry, without paid API calls. The retired reconstruction workflow has been removed.

## Open and reproduce

```bash
# Native splat viewer; no API key needed after download.
python3 -m http.server 8870 --bind 127.0.0.1 --directory outputs/worldlabs_viewer

# Native interactive MuJoCo scene.
.venv/bin/python scripts/view_scene.py --scene outputs/worldlabs_scene/scene.xml --fast-exit

# Current Atlas job status; resumes without submitting duplicate work.
.venv/bin/python scripts/worldlabs_scene.py status --stage splats

# The actual map/state navigation test.
.venv/bin/python scripts/run_skill_agent.py --scene outputs/worldlabs_scene \\
  --task navigate --run-name atlas_explore --device cuda --video

# Same route in native PhysX, using the saved scene-bound plan.
scripts/run_isaac.sh scripts/run_scalebfm_isaac.py --task explore \\
  --scene outputs/worldlabs_scene/scene.usda --plan outputs/worldlabs/final_navigation_plan.json \\
  --seconds 38 --run-name atlas_native --headless --fast-exit --device cuda:0
```

Viewer controls: select a source/novel camera, drag to orbit, wheel to zoom, WASD to move, Q/E down/up. The browser viewer is visual-only; collision/dynamics run in MuJoCo and Isaac Lab. The viewer folder is self-contained and contains neither the API key nor signed download URLs. Node rendering tools were tested with Spark 2.4.0, Three.js 0.187.0 and Playwright; the simulator project adds `trimesh` for GLB conversion.

The selected raw requests, resumable operation records, calibration and intermediate variants are private local data under `outputs/worldlabs/`. They are not part of the public source repository. The `worldlabs_scene.py` helper uses the CLI's saved login, `WLT_API_KEY`, or the authorized local first-line credential file. Use a fresh output workspace and source-specific frame/ROI/camera configuration for another recording; this office's fitting constants are not universal. See the reusable skill's [Atlas integration reference](../skills/worldlabs-scene-reconstruction/references/atlas.md).

Earlier extraction artifacts and the blocked planning attempt are preserved. A small benchmark cube initially closed a narrow conservative clearance corridor; the selected scene reserves the routes before placing that prop. The original, stereo/trimmed and selected constrained meshes remain separate for review. Generated assets and archives remain Git-ignored, as requested.
'''
Path('reports/WORLDLABS_RESULTS.md').write_text(text)
public=['worldlabs_skill_validation.json','WORLDLABS_RESULTS.md','worldlabs_summary.json','worldlabs_mujoco_validation.json','worldlabs_isaac_validation.json','worldlabs_export_validation.json','worldlabs_asset_dependencies.json','worldlabs_appearance_validation.json','worldlabs_splat_appearance_validation.json','worldlabs_final_explore_scalebfm_explore_mode0.json','worldlabs_native_explore_scalebfm_isaac_explore.json','worldlabs_viewer_validation.json']
provenance=dict(scene_fingerprint=identity,splat_sha256=sha(spz),raw_mesh_sha256=sha(root/'mesh/mesh.glb'),constrained_mesh_sha256=sha(root/'constrained_mesh.glb'),collider_glb_sha256=sha(root/'mesh/colliderBoxes.glb'),input_video_sha256=sha('data/scene/scene.mp4'),operations=summary['worldlabs_operations'],curation=curation,visual_constraints=constraints,reports_sha256={n:sha(Path('reports')/n) for n in public},source_sha256={str(p):sha(p) for folder in ['scripts','scene_reconstruction','skills/worldlabs-scene-reconstruction'] for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'})
Path('reports/worldlabs_provenance.json').write_text(json.dumps(provenance,indent=2));public+=['worldlabs_provenance.json']
delivery=Path('deliverables/worldlabs_scene_bundle.zip');delivery.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(delivery,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for folder,name in [(scene,scene),(viewer,viewer)]:
  for p in folder.rglob('*'):
   if p.is_file():z.write(p,Path(name)/p.relative_to(folder))
 for n in public:z.write(Path('reports')/n,Path('reports')/n)
 z.write(root/'final_navigation_plan.json','outputs/worldlabs/final_navigation_plan.json')
 z.write(root/'constrained_mesh.glb','outputs/worldlabs/constrained_mesh.glb')
 z.write(root/'mesh/colliderBoxes.glb','outputs/worldlabs/mesh/colliderBoxes.glb')
 for folder in ['scripts','scene_reconstruction','skills/worldlabs-scene-reconstruction']:
  for p in Path(folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':z.write(p,p)
 for p in [Path('requirements.txt'),Path('.gitignore'),Path('data/scene/scene.mp4'),Path('data/scene/calibration.example.json')]:z.write(p,p)
 z.write('README.md','PROJECT_README.md')
 z.writestr('README.md','# World Labs scene bundle\n\nRead [the result and limitations](reports/WORLDLABS_RESULTS.md).\n\nThe environment is at `outputs/worldlabs_scene/`; the native splat viewer is at `outputs/worldlabs_viewer/`.\n\nRun `python3 -m http.server 8870 --bind 127.0.0.1 --directory outputs/worldlabs_viewer` and open http://127.0.0.1:8870/. No API key is needed for these downloaded files.\n\nThe room is static, with one added free benchmark cube. Absolute scale remains provisional. Code and source video are included; learned weights, robot sources and the Isaac runtime are separate prerequisites described in PROJECT_README.md.\n')
 z.writestr('reports/worldlabs_delivery_integrity.json',json.dumps(dict(scene_fingerprint=identity,zip_crc_passed=True,archive_self_digest='Recorded outside this archive to avoid a self-referential checksum',metric_accuracy_verified=False),indent=2))
key=Path('worldlabs').read_text().splitlines()[0].strip().encode() if Path('worldlabs').exists() else b''
with zipfile.ZipFile(delivery) as z:
 assert z.testzip() is None
 for item in z.infolist():
  data=z.read(item)
  if key and key in data:raise RuntimeError('Credential detected in archive; refusing delivery')
  if Path(item.filename).suffix in ['.json','.md','.html','.js','.mjs','.py','.txt'] and re.search(rb'https?://[^\s"<>]+[?&](?:sig|X-Amz-Signature|X-Goog-Signature)=',data,re.I):raise RuntimeError('Private signed URL detected in archive')
 relocated=Path('/tmp/worldlabs_delivery_check')/identity[:12]
 if relocated.exists():shutil.rmtree(relocated)
 z.extractall(relocated)
relocated_scene=relocated/scene;mujoco.MjModel.from_xml_path(str(relocated_scene/'scene.xml'));stage=Usd.Stage.Open(str(relocated_scene/'scene.usda'));layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(relocated_scene/'scene.usda'));assert stage and not unresolved;assert scene_fingerprint(relocated_scene)['sha256']==identity;assert sha(relocated/viewer/'room.spz')==sha(spz)
integrity=dict(scene_fingerprint=identity,archive=str(delivery),bytes=delivery.stat().st_size,sha256=sha(delivery),zip_crc_passed=True,relocated_mujoco_load=True,relocated_usd_load=True,relocated_dependencies_resolved=True,relocated_scene_identity=True,relocated_splat_identity=True,contains_api_key=False,contains_signed_download_urls=False,metric_accuracy_verified=False)
Path('reports/worldlabs_delivery_integrity.json').write_text(json.dumps(integrity,indent=2));print(json.dumps(summary,indent=2));print(json.dumps(integrity,indent=2))
