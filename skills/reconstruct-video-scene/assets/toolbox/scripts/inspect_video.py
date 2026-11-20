"""Decode without leaking phone metadata, and quantify feature/parallax evidence."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/inspection'))
    p.add_argument('--fps', type=float, default=2)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-y', '-i', str(args.video), '-map_metadata', '-1',
                    '-vf', f'fps={args.fps}', str(args.output/'frame_%04d.jpg')],
                   check=True, capture_output=True)
    files = sorted(args.output.glob('frame_[0-9][0-9][0-9][0-9].jpg'))
    sift = cv2.SIFT_create(nfeatures=5000)
    features = [sift.detectAndCompute(cv2.imread(str(f), 0), None) for f in files]
    matches = []
    for i in range(len(files)-1):
        k1, d1 = features[i]; k2, d2 = features[i+1]
        raw = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        good = [a for a,b in raw if a.distance < .72*b.distance]
        a = np.float32([k1[m.queryIdx].pt for m in good])
        b = np.float32([k2[m.trainIdx].pt for m in good])
        h, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 2.0)
        err = np.linalg.norm(cv2.perspectiveTransform(a[None], h)[0]-b, axis=1)
        matches.append(dict(frame_a=i+1, frame_b=i+2, matches=len(good),
                            homography_inliers=int(mask.sum()),
                            homography_inlier_fraction=float(mask.mean()),
                            median_homography_residual_px=float(np.median(err))))
    w,h = Image.open(files[0]).size
    report = dict(frame_count=len(files), sample_fps=args.fps, upright_resolution=[w,h],
                  pairs=matches, metric_scale='unobserved: needs measured reference',
                  interpretation='High homography support is consistent with rotational camera motion or planar surfaces; it does not prove recoverable depth.')
    (args.output/'video_analysis.json').write_text(json.dumps(report, indent=2))
    sheet = Image.new('RGB', (9*160, ((len(files)+8)//9)*310), '#111111')
    for j,f in enumerate(files):
        im=Image.open(f); im.thumbnail((160,284)); x=j%9*160;y=j//9*310
        sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x+4,y+285),f'{j/args.fps:.1f}s #{j+1}',fill='white')
    sheet.save(args.output/'contact_sheet.jpg')
    print(json.dumps({k:v for k,v in report.items() if k != 'pairs'},indent=2))


if __name__ == '__main__':
    main()
