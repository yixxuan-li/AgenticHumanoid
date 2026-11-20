"""Plot measured final-scene trajectories; figures do not certify metric scale."""
import os,sys,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/agentic-humanoid-matplotlib')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,mujoco
from scene_reconstruction.navigation import collision_grid
from scene_reconstruction.provenance import scene_fingerprint
scene=Path('outputs/final_scene');manifest=json.loads((scene/'manifest.json').read_text());identity=scene_fingerprint(scene)['sha256']
reports=[json.loads(Path(p).read_text()) for p in ['reports/dex3_pick_final_pick.json','reports/isaac_dex3_pick_final_sequence.json','reports/final_explore_scalebfm_explore_mode0.json']]
if any(r['scene_fingerprint']!=identity for r in reports):raise ValueError('Refusing to plot trials from another scene version')
a=np.loadtxt('outputs/controller/pick/final_pick/trajectory.csv',delimiter=',',skiprows=1);b=np.loadtxt('outputs/controller/isaac_pick/final_sequence/trajectory.csv',delimiter=',',skiprows=1);explore=np.loadtxt('outputs/controller/final_explore/explore_mode0_trajectory.csv',delimiter=',',skiprows=1)
m=mujoco.MjModel.from_xml_path(str(scene/'scene.xml'));d=mujoco.MjData(m)
for _ in range(2000):mujoco.mj_step(m,d)
grid=collision_grid(m,d,manifest['dimensions'],radius=.32);fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained');colors=['#0b867d','#ca6c26']
ax=axes[0,0];ny,nx=grid['blocked'].shape;image=np.ones((ny,nx,3))*.95;image[grid['occupied']]=[.82,.81,.76];image[grid['blocked'].astype(bool)]=[.32,.35,.37];r=grid['resolution'];ax.imshow(image,origin='lower',extent=[-r/2,(nx-.5)*r,-r/2,(ny-.5)*r]);ax.plot(explore[:,1],explore[:,2],color='#526ac5',label='MuJoCo exploration');stop=reports[0]['pick_start_seconds'];ax.plot(a[a[:,0]<stop,1],a[a[:,0]<stop,2],color=colors[0],lw=2,label='MuJoCo pick approach');ax.plot(b[b[:,0]<stop,1],b[b[:,0]<stop,2],color=colors[1],ls='--',label='PhysX pick approach');ax.set(xlim=[0,manifest['dimensions'][0]],ylim=[0,manifest['dimensions'][1]],xlabel='X [provisional m]',ylabel='Y [provisional m]',title='Actual root trajectories and collision map');ax.set_aspect('equal');ax.legend(fontsize=8,loc='upper left')
ax=axes[0,1]
for data,color,label in [(a,colors[0],'MuJoCo'),(b,colors[1],'PhysX')]:ax.plot(data[:,0],data[:,12],color=color,label=label)
ax.axvline(stop,color='.5',lw=1,ls=':',label='Manipulation phase');ax.axhline(manifest['desk_height'],color='.6',ls='--',lw=1,label='Tabletop');ax.set(xlabel='Simulation time [s]',ylabel='Block center height [m]',title='Contact-only lift after navigation',ylim=[.70,1.02]);ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axes[1,0]
for data,color,label in [(a,colors[0],'MuJoCo'),(b,colors[1],'PhysX')]:ax.plot(data[:,0],data[:,3],color=color,label=label)
ax.set(xlabel='Simulation time [s]',ylabel='Robot root height [m]',title='Physical root motion; no pose forcing');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axes[1,1]
for data,color,label in [(a,colors[0],'MuJoCo'),(b,colors[1],'PhysX')]:
 mask=data[:,0]>stop+2.+.02;ax.plot(data[mask,0]-stop,data[mask,13],color=color,label=label)
ax.set(xlabel='Time since manipulation begins [s]',ylabel='Desired wrist position error [m]',title='Wrist-controlled phase (after initial 2 s hold)',ylim=[0,.16]);ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Released ScaleBFM + Dex3: measured simulation results\nRoom scale assumes 0.74 m table height; added benchmark block; map/simulator-state agent',fontsize=12)
fig.savefig(scene/'benchmark_results.png',dpi=180);fig.savefig(scene/'benchmark_results.pdf');plt.close(fig)
print('Saved '+str(scene/'benchmark_results.png'))
