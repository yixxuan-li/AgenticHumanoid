"""Conservative 2-D occupancy from actual MuJoCo collision shapes, with A*."""
import heapq
import itertools
import math
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def collision_grid(model,data,dimensions,resolution=.04,radius=.28,height=1.4):
    nx,ny=np.ceil(np.array(dimensions[:2])/resolution).astype(int)+1
    blocked=np.zeros((ny,nx),np.uint8)
    supported=np.zeros_like(blocked)
    for i in range(model.ngeom):
        if not (model.geom_contype[i] or model.geom_conaffinity[i]):continue
        shape=int(model.geom_type[i]);size=model.geom_size[i];r=data.geom_xmat[i].reshape(3,3);pos=data.geom_xpos[i]
        if shape==6:half=size # box
        elif shape==5:half=np.array([size[0],size[0],size[1]]) # cylinder
        elif shape==2:half=np.repeat(size[0],3)
        elif shape==3:half=np.array([size[0],size[0],size[0]+size[1]])
        else:continue
        pts=np.array(list(itertools.product([-1,1],repeat=3)))*half
        pts=pts@r.T+pos
        uv=np.int32(np.round(pts[:,:2]/resolution));hull=cv2.convexHull(uv)
        if pts[:,2].max()<.03:
            if -.03<=pts[:,2].max():cv2.fillConvexPoly(supported,hull,1)
            continue
        if pts[:,2].min()>height:continue
        cv2.fillConvexPoly(blocked,hull,1)
    blocked[supported==0]=1
    blocked[0,:]=1;blocked[-1,:]=1;blocked[:,0]=1;blocked[:,-1]=1
    clearance=distance_transform_edt(1-blocked)*resolution
    return dict(blocked=blocked,clearance=clearance,occupied=clearance<radius,resolution=resolution,radius=radius)


def plan(grid,start,goal):
    resolution=grid['resolution'];occupied=grid['occupied'];ny,nx=occupied.shape
    def cell(xy):return tuple(np.round(np.array(xy)/resolution).astype(int))
    a,b=cell(start),cell(goal)
    for name,(x,y) in [('start',a),('goal',b)]:
        if not (0<=x<nx and 0<=y<ny) or occupied[y,x]:raise ValueError(f'{name} is obstructed for radius {grid["radius"]} m: {(round(float(x*resolution),3),round(float(y*resolution),3))}')
    cost={a:0.};previous={};queue=[(math.dist(a,b),a)]
    moves=[(dx,dy) for dx in [-1,0,1] for dy in [-1,0,1] if dx or dy]
    while queue:
        _,u=heapq.heappop(queue)
        if u==b:
            nodes=[b]
            while nodes[-1]!=a:nodes.append(previous[nodes[-1]])
            return np.array(nodes[::-1])*resolution
        for dx,dy in moves:
            v=(u[0]+dx,u[1]+dy);x,y=v
            if not (0<=x<nx and 0<=y<ny) or occupied[y,x]:continue
            if dx and dy and (occupied[u[1],x] or occupied[y,u[0]]):continue
            c=cost[u]+math.hypot(dx,dy)
            if c<cost.get(v,float('inf')):
                cost[v]=c;previous[v]=u;heapq.heappush(queue,(c+math.dist(v,b),v))
    raise ValueError('No collision-free path at this body radius')
