"""Build the editable semantic reconstruction and both simulator formats."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene_reconstruction.export import export_all
from scene_reconstruction.layout import make_layout
from scene_reconstruction.textures import write_textures


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/scene'))
    p.add_argument('--frames',type=Path,default=Path('outputs/inspection'))
    p.add_argument('--measured-desk-height',type=float,help='Measured desk height [m]; rescales all geometry')
    p.add_argument('--calibration',type=Path,help='JSON with description, reconstructed_length_m, measured_length_m')
    args=p.parse_args();manifest=make_layout()
    if args.calibration and args.measured_desk_height:p.error('Use one metric anchor at a time')
    anchor=None
    if args.calibration:
        anchor=json.loads(args.calibration.read_text())
        if min(anchor['reconstructed_length_m'],anchor['measured_length_m'])<=0:p.error('Anchor lengths must be positive meters')
        factor=anchor['measured_length_m']/anchor['reconstructed_length_m']
    elif args.measured_desk_height:
        if not .3<args.measured_desk_height<1.5:p.error('Check measured height in meters')
        factor=args.measured_desk_height/manifest['desk_height']
        anchor=dict(description='measured desk height',reconstructed_length_m=manifest['desk_height'],measured_length_m=args.measured_desk_height)
    if anchor:
        if not .25<factor<4:p.error('Implausible global scale factor; check units and anchor correspondence')
        manifest['dimensions']=[v*factor for v in manifest['dimensions']]
        manifest['desk_height']*=factor;manifest['scale_status']='single_measured_anchor'
        manifest['global_scale']=factor
        manifest['metric_anchor']=dict(**anchor,measured=True)
        manifest['navigation']['start']=[v*factor for v in manifest['navigation']['start']]
        manifest['navigation']['goals']=[[v*factor for v in pt] for pt in manifest['navigation']['goals']]
        for b in manifest['bodies']:
            b['pos']=[v*factor for v in b['pos']];b['mass']*=factor**3
            for part in b['parts']:
                part['pos']=[v*factor for v in part['pos']];part['size']=[v*factor for v in part['size']]
    write_textures(args.output/'textures',args.frames)
    export_all(manifest,args.output)
    print(f"Exported {len(manifest['bodies'])} bodies: {args.output}; scale={manifest['scale_status']}")


if __name__=='__main__':main()
