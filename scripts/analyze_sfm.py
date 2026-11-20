"""Estimate relative scale from source-observed floor/table planes, with diagnostics."""
import json
from pathlib import Path
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fit_plane(points,threshold,seed=119,normal_hint=None):
    if len(points)<8:raise ValueError('Insufficient points to support a plane')
    rng=np.random.default_rng(seed);best=np.zeros(len(points),bool)
    for _ in range(3000):
        a,b,c=points[rng.choice(len(points),3,replace=False)];n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
        if norm<1e-8:continue
        n/=norm;mask=np.abs((points-a)@n)<threshold
        if normal_hint is not None and abs(n@normal_hint)<np.cos(np.radians(25)):continue
        if mask.sum()>best.sum():best=mask
    center=np.median(points[best],axis=0);_,_,v=np.linalg.svd(points[best]-center);normal=v[-1]
    return normal,-normal@center,best


def main():
    rec=pycolmap.Reconstruction('outputs/colmap/sparse/0');points=[];colors=[];floor=[];table=[]
    for pid,point in rec.points3D.items():
        rgb=point.color.astype(float);floor_vote=False;table_vote=False
        for element in point.track.elements:
            image=rec.images[element.image_id];xy=image.points2D[element.point2D_idx].xy
            frame=int(image.name.split('_')[-1].split('.')[0])
            if xy[1]>430 and rgb[0]>rgb[1]*1.07 and rgb[1]>rgb[2]*1.08 and rgb.min()>35:floor_vote=True
            if 15<=frame<=22 and 340<xy[1]<420 and rgb.mean()>135 and rgb.max()-rgb.min()<55:table_vote=True
        points.append(point.xyz);colors.append(rgb);floor.append(floor_vote);table.append(table_vote)
    points=np.array(points);colors=np.array(colors)/255;floor=np.array(floor);table=np.array(table)
    report=dict(floor_candidates=int(floor.sum()),table_candidates=int(table.sum()),source='COLMAP sparse points and source-frame region/color masks',metric_anchor='assumed 0.74 m desk height, not measured')
    try:
        centers=np.array([im.projection_center() for im in rec.images.values()])
        up=np.mean([-im.cam_from_world().rotation.matrix().T[:,1] for im in rec.images.values()],axis=0);up/=np.linalg.norm(up)
        spread=np.linalg.norm(np.percentile(points,95,axis=0)-np.percentile(points,5,axis=0))
        n,d,mask=fit_plane(points[floor],spread*.004,normal_hint=up)
        if np.median(centers@n+d)<0:n=-n;d=-d
        floor_dist=points@n+d
        # Select a tabletop cluster parallel to the observed ground plane.
        table_heights=floor_dist[table];bins=np.linspace(.05,np.percentile(floor_dist,95),100)
        counts,edges=np.histogram(table_heights,bins);idx=np.argmax(counts);h=np.median(table_heights[(table_heights>=edges[idx])&(table_heights<=edges[idx+1])])
        scale=.74/h
        rot,_=Rotation.align_vectors([[0,0,1]],[n]);aligned=rot.apply(points)*scale;aligned[:,2]+=d*scale
        residual=np.abs(points[floor][mask]@n+d)*scale
        report.update(floor_normal=n.tolist(),floor_offset=float(d),floor_inliers=int(mask.sum()),floor_residual_median_m=float(np.median(residual)),
            tabletop_height_model_units=float(h),provisional_m_per_model_unit=float(scale),
            observed_bounds_m=np.percentile(aligned,[2,98],axis=0).tolist(),
            caution='Plane/color masks can include other surfaces; scale is only a diagnostic hypothesis.')
        report['quality_checks']=dict(floor_majority_inliers=bool(mask.mean()>.5),floor_residual_under_3cm=bool(np.median(residual)<.03),few_points_below_floor=bool((aligned[:,2]<-.10).mean()<.03))
        report['accepted_for_scale']=all(report['quality_checks'].values())
        np.savez_compressed('outputs/colmap/aligned_cloud.npz',points=aligned,colors=colors,rotation=rot.as_matrix(),floor_offset=d,provisional_scale=scale)
        fig=plt.figure(figsize=(12,8));ax=fig.add_subplot(projection='3d');ax.scatter(*aligned.T,c=colors,s=3);ax.set(xlabel='X [provisional m]',ylabel='Y [provisional m]',zlabel='Z [provisional m]',title='Feature-based reconstruction: measured reprojections, assumed metric anchor');ax.set_box_aspect(np.ptp(aligned,axis=0));fig.tight_layout();fig.savefig('outputs/colmap/sparse_geometry.png',dpi=150);plt.close(fig)
    except Exception as exc:report['status']='plane estimation failed';report['error']=str(exc)
    Path('reports/sfm_geometry.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
