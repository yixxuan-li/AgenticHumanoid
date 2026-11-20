"""Fit a physical semantic scene to the dense source geometry."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene_reconstruction.dense_layout import make_dense_layout
from scene_reconstruction.export import export_all
from scene_reconstruction.textures import write_textures
from scene_reconstruction.photo_skin import attach_photo_skin
from scene_reconstruction.planar_texture import make_table_texture

p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,default=Path('outputs/geometry_full'));p.add_argument('--dynamic-geometry',type=Path);p.add_argument('--output',type=Path,default=Path('outputs/reconstructed_scene'));p.add_argument('--hide-front-wall-visual',action='store_true',help='Retain its collision proxy but use captured surfaces for appearance');args=p.parse_args()
manifest=make_dense_layout(args.geometry/'scene_features.json')
if args.hide_front_wall_visual:
    wall=next(b for b in manifest['bodies'] if b['name']=='wall_front_observed')
    for part in wall['parts']:part['visible']=False
    wall['visual_fallback']='Captured wall surfaces; solid primitive retained for collision only'
write_textures(args.output/'textures',Path('outputs/inspection'));make_table_texture(manifest,args.geometry,args.output);export_all(manifest,args.output)
print(json.dumps(attach_photo_skin(manifest,args.geometry,args.output,args.dynamic_geometry)),flush=True)
print(f"Exported {len(manifest['bodies'])} fitted bodies to {args.output}; scale remains unmeasured")
