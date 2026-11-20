"""Independent feature-based geometry baseline; preserves registration failures."""
import argparse
import json
from pathlib import Path
import pycolmap

p=argparse.ArgumentParser();p.add_argument('--frames',type=Path,default=Path('outputs/inspection'));p.add_argument('--output',type=Path,default=Path('outputs/colmap'));args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True);database=args.output/'database.db'
images=[p.name for p in sorted(args.frames.glob('frame_[0-9][0-9][0-9][0-9].jpg'))]
if not database.exists():
    reader=pycolmap.ImageReaderOptions();reader.camera_model='SIMPLE_RADIAL'
    extract=pycolmap.FeatureExtractionOptions();extract.num_threads=8;extract.sift.max_num_features=5000
    pycolmap.extract_features(database,args.frames,image_names=images,camera_mode=pycolmap.CameraMode.SINGLE,reader_options=reader,extraction_options=extract,device=pycolmap.Device.cpu)
    match=pycolmap.FeatureMatchingOptions();match.num_threads=8
    pycolmap.match_exhaustive(database,matching_options=match,device=pycolmap.Device.cpu)
options=pycolmap.IncrementalPipelineOptions();options.num_threads=8;options.max_runtime_seconds=180;options.random_seed=119
models=pycolmap.incremental_mapping(database,args.frames,args.output/'sparse',options=options)
report=dict(pycolmap_version=pycolmap.__version__,frames=len(images),models=[],scale_status='arbitrary SfM scale')
for i,model in models.items():
    model.write_text(args.output/'sparse'/str(i));model.export_PLY(args.output/f'cloud_{i}.ply')
    report['models'].append(dict(id=i,registered_images=model.num_reg_images(),points3d=model.num_points3D(),mean_reprojection_error_px=model.compute_mean_reprojection_error(),mean_track_length=model.compute_mean_track_length(),summary=model.summary()))
report['interpretation']='Sparse registration and low reprojection error do not certify dense geometry or metric scale.'
(args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
