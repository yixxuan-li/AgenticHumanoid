"""Compose native MuJoCo figures using the shared editable layout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from figure_layout import Teaser, system, INK, MUTED, BLUE, TEAL
ROOT=Path(__file__).resolve().parents[1]

def teaser(out):
    a=out/'assets'; c=Teaser()
    crop=(0,180,2200,1308)
    c.image(a/'hero_sequence.png',(30,30,2670,1370),region=crop)
    # Labels use exact fixed-camera projection of the four root positions.
    anchors=json.loads((out/'scene/hero_anchors.json').read_text())
    for word,(px,py) in zip(['Pick','Turn','Carry','Place'],anchors):
        x=30+px*2670/2200; y=30+(py-crop[1])*1370/(crop[3]-crop[1])+12
        c.label(x-100,min(y,1325),word,200,dot=True)
    c.box(55,55,390,265,color='none',edge='white',radius=15,lw=2)
    c.image(a/'onboard_pick.png',(66,66,368,243),layer=10)
    c.label(78,255,'Onboard',180)
    c.box(2243,1108,425,292,color='none',edge='white',radius=15,lw=2)
    c.image(a/'onboard_place.png',(2254,1119,403,270),layer=10)
    c.label(2266,1328,'Onboard',180)
    c.box(30,1416,2670,60,color='#F1F3F4',radius=8)
    c.text(1365,1429,'Move the cube from the table into the drawer.',13,INK,ha='center',style='italic')
    names=['posture','reach','push','open_drawer','place','close_drawer']
    labels=['Posture','Reach','Push','Open drawer','Place','Close drawer']
    for i,(name,label) in enumerate(zip(names,labels)):
        row,col=divmod(i,2); x=2740+col*715; y=30+row*385
        c.image(a/f'{name}.png',(x,y,695,370))
        c.label(x+12,y+306,label,245 if 'drawer' in label else 170)
    c.text(2745,1211,'AgenticHumanoid',14,INK,'bold')
    c.box(2740,1280,665,90,color='#EEF5FC',radius=10)
    c.text(3072,1307,'Skill programs',15,BLUE,'bold',ha='center')
    c.box(3470,1280,685,90,color='#FBF3E7',radius=10)
    c.text(3812,1307,'Symbolic knowledge',14,'#AB7429','bold',ha='center')
    c.text(3438,1301,'↔',24,TEAL,ha='center')
    c.text(2745,1424,'Explore → validate → reuse',12,TEAL)
    c.text(4155,1430,'MuJoCo · authored poses',10,MUTED,ha='right')
    c.save(out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/figures_v5')
    args=parser.parse_args();out=args.output.resolve();a=out/'assets'
    # The system diagram shows the actual captured input and native room render.
    capture=ROOT/'outputs/figures_v3/assets/captured_13s.png'
    if not (a/'captured_13s.png').exists():shutil.copyfile(capture,a/'captured_13s.png')
    shutil.copyfile(a/'room_probe.png',a/'reconstructed_room.png')
    teaser(out);system(out)
    report=json.loads((out/'render_provenance.json').read_text())
    report.update(figure_version=5, language='English',
        design_reference={'url':'https://arxiv.org/pdf/2611.08039','figure':1,
            'borrowed_design_principles':['Dominant multi-exposure view','Onboard insets','Compact skill gallery','Short action labels'],
            'paper_artwork_or_results_in_final':False},
        figure_assets='MuJoCo images plus an actual captured input frame in the system diagram',
        diagram_elements='Editable vector labels, arrows, boxes and architecture',
        source_scripts=['scripts/render_research_teaser_v5.py','scripts/draw_research_figures_v5.py'])
    report['captured_frame_sha256']=hashlib.sha256((a/'captured_13s.png').read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(out/'teaser_v5.png'); print(out/'system_v5.png')


if __name__=='__main__':main()
