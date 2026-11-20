"""Source-pixel textures and reproducible procedural surface detail."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw


def write_textures(output: Path, frames: Path):
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(119)
    # Color/structure follows visible oak laminate; procedural detail is not scan data.
    h,w=1536,2048
    floor=np.zeros((h,w,3),dtype=np.float32)
    base=np.array([153,126,84],dtype=float)
    for row in range(32):
        y0,y1=row*h//32,(row+1)*h//32
        tone=base+rng.normal(0,7,(1,1,1))
        grain=rng.normal(0,1,(y1-y0,w))
        grain=cv2.GaussianBlur(grain,(101,1),0)*18
        band=rng.normal(0,1,(y1-y0,1))*1.8
        floor[y0:y1]=tone+grain[:,:,None]+band[:,:,None]
        floor[y0:y0+2]-=22
        offset=int(rng.integers(0,w//4))
        for xx in range(offset,w,w//4):floor[y0:y1,max(0,xx-1):xx+1]-=20
    Image.fromarray(np.uint8(np.clip(floor,0,255))).save(output/'floor.png')
    desktop=cv2.resize(np.uint8(np.clip(floor[:128],0,255)),(1024,512))
    desktop=np.clip(desktop.astype(float)*[1.22,1.10,.92],0,255).astype(np.uint8)
    Image.fromarray(desktop).save(output/'desktop.png')

    def crop_warp(frame,quad,name,size):
        im=cv2.imread(str(frames/f'frame_{frame:04d}.jpg'))
        dst=np.float32([[0,0],[size[0]-1,0],[size[0]-1,size[1]-1],[0,size[1]-1]])
        H=cv2.getPerspectiveTransform(np.float32(quad),dst)
        result=cv2.warpPerspective(im,H,size,borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(output/f'{name}.png'),result)
        return dict(frame=frame,quad_px=quad,output_size=list(size),method='planar homography from source pixels; resolution is not new detail')
    provenance={}
    # Use a visible subsection of the glass board; the recording clips the full extent.
    provenance['whiteboard']=crop_warp(33,[[43,94],[313,120],[313,321],[76,431]],'whiteboard',(768,480))
    provenance['screen']=crop_warp(16,[[194,230],[232,233],[232,261],[194,258]],'screen',(384,240))
    provenance['box_label']=crop_warp(24,[[258,422],[284,424],[280,434],[254,432]],'box_label',(256,128))
    # A clean fiducial design approximates the repeated mat in the video. It is labeled
    # procedural because recovering its complete original artwork is impossible here.
    mat=Image.new('RGB',(1800,800),(221,220,211));d=ImageDraw.Draw(mat)
    for cx in [430,1370]:
        d.ellipse((cx-300,120,cx+300,720),fill=(211,205,196))
        d.ellipse((cx-240,180,cx+240,660),fill=(224,222,212))
        d.ellipse((cx-160,260,cx+160,580),fill=(204,207,200))
        d.ellipse((cx-110,310,cx+110,530),fill=(229,226,215))
        d.polygon([(cx-330,40),(cx+330,40),(cx,130)],fill=(144,152,177))
        d.polygon([(cx-330,760),(cx+330,760),(cx,670)],fill=(144,152,177))
    for row in [0,1,2]:
        for col in range(7):
            x=45+col*270;y=42+row*310
            d.rectangle((x-10,y-10,x+85,y+85),fill=(241,240,234))
            d.rectangle((x,y,x+75,y+75),fill=(18,20,18))
            for a in range(5):
                for b in range(5):
                    if rng.random()>.5:d.rectangle((x+10+a*11,y+10+b*11,x+19+a*11,y+19+b*11),fill=(230,230,221))
    mat.save(output/'fiducial.png')
    provenance['fiducial']=dict(method='procedural approximation of visible printed tracking mats; markers are decorative, not calibrated AprilTags')
    provenance['floor']=dict(method='procedural oak laminate appearance following video; not a measured reflectance map')
    provenance['desktop']=dict(method='procedural wood appearance following video')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return provenance
