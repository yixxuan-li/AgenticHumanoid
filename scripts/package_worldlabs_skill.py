"""Sync the portable skill toolbox from maintained source and build a local ZIP."""
from pathlib import Path
import shutil,zipfile

root=Path(__file__).resolve().parents[1]
skill=root/'skills/worldlabs-scene-reconstruction'
toolbox=skill/'assets/toolbox'
if toolbox.exists():shutil.rmtree(toolbox)
(toolbox/'scripts').mkdir(parents=True)
(toolbox/'scene_reconstruction').mkdir()
scripts=['inspect_video','select_worldlabs_frames','worldlabs_scene','inspect_worldlabs_scene','calibrate_worldlabs_scene','prepare_worldlabs_mesh','export_worldlabs_scene','prepare_worldlabs_viewer','render_worldlabs_scene','validate_worldlabs_mujoco','validate_worldlabs_isaac','validate_export','audit_assets','rescale_scene','prepare_worldlabs_review_geometry','validate_heldout_views']
for name in scripts:shutil.copy2(root/'scripts'/f'{name}.py',toolbox/'scripts'/f'{name}.py')
for name in ['__init__.py','export.py','layout.py','navigation.py','provenance.py','atlas_viewer.html']:shutil.copy2(root/'scene_reconstruction'/name,toolbox/'scene_reconstruction'/name)
(toolbox/'requirements.txt').write_text('numpy>=2.0\nscipy>=1.14\nPillow>=11\nopencv-python-headless>=4.10\nmujoco==3.11.0\nusd-core>=25.5\ntrimesh>=4.0\nimageio\nimageio-ffmpeg\nmatplotlib>=3.9\n')
(toolbox/'.gitignore').write_text('/outputs/\n/deliverables/\n/worldlabs\n.env\n.env.*\n.venv/\nnode_modules/\n__pycache__/\n*.pyc\n*.spz\n*.glb\n*.npz\n*.zip\n')
delivery=root/'deliverables/worldlabs-scene-reconstruction.zip';delivery.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(delivery,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(skill.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:archive.write(path,path.relative_to(skill.parent))
print(delivery)
