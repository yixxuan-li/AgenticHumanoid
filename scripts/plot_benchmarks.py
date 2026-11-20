"""Plot saved physical trajectories from both engines, without rerunning tests."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/agentichumanoid-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('outputs/controller')
load=lambda path:np.loadtxt(root/path,delimiter=',',skiprows=1)
ms=load('squat_mode7_trajectory.csv');ps=load('isaac_squat_trajectory.csv')
mn=load('navigate_mode0_trajectory.csv');pn=load('isaac_navigate_trajectory.csv')
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
fig.suptitle('Released ScaleBFM medium policy: actual MuJoCo and Isaac Lab dynamics',fontweight='bold')
for ax,data1,data2,column,title,ylabel in [(axes[0,0],ms,ps,3,'Whole-body squat tracking','Pelvis height [m]'),(axes[0,1],mn,pn,1,'Pelvis-mode navigation','Pelvis X [m]')]:
    target_col=column+3
    ax.plot(data1[:,0],data1[:,target_col],'--',color='#3e4650',label='Reference')
    ax.plot(data1[:,0],data1[:,column],color='#008b83',label='MuJoCo')
    ax.plot(data2[:,0],data2[:,column],color='#cd6e35',label='Isaac Lab / PhysX')
    ax.set(title=title,xlabel='Simulation time [s]',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(frameon=False)
for data,label,color in [(ms,'MuJoCo','#008b83'),(ps,'Isaac Lab / PhysX','#cd6e35')]:axes[1,0].plot(data[:,0],data[:,-1],label=label,color=color)
axes[1,0].set(title='Squat reference-position error',xlabel='Simulation time [s]',ylabel='Root error [m]');axes[1,0].grid(alpha=.2)
for data,label,color in [(mn,'MuJoCo','#008b83'),(pn,'Isaac Lab / PhysX','#cd6e35')]:axes[1,1].plot(data[:,0],data[:,-1],label=label,color=color)
axes[1,1].set(title='Navigation reference-position error',xlabel='Simulation time [s]',ylabel='Root error [m]');axes[1,1].grid(alpha=.2)
fig.supxlabel('No root teleporting or reference forcing during execution. Scene scale and mass/friction remain provisional.',fontsize=10)
fig.savefig(root/'benchmark_comparison.png',dpi=180);fig.savefig(root/'benchmark_comparison.pdf');plt.close(fig)
summary={}
for task,one,two in [('squat',ms,ps),('navigate',mn,pn)]:
    a=one[:,1:4];b=np.column_stack([np.interp(one[:,0],two[:,0],two[:,j]) for j in range(1,4)])
    summary[task]=dict(mean_cross_engine_root_difference_m=float(np.linalg.norm(a-b,axis=1).mean()),mujoco_reference_rmse_m=float(np.sqrt(np.mean(one[:,-1]**2))),isaac_reference_rmse_m=float(np.sqrt(np.mean(two[:,-1]**2))))
Path('reports/cross_engine_comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
