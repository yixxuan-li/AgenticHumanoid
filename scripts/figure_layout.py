"""Shared editable typography, viewports, and system layout for native figures."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/agentic-humanoid-figures')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image
INK='#243440'; MUTED='#65717B'; LINE='#DCE5EC'
BLUE='#337ABD'; ORANGE='#E88835'; TEAL='#188F85'; AMBER='#B67825'; PURPLE='#7660A4'
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})

class Canvas:
    def __init__(self,w,h):
        self.w,self.h=w,h
        self.fig=plt.figure(figsize=(w/200,h/200),dpi=200,facecolor='white')
        self.ax=self.fig.add_axes([0,0,1,1],zorder=10)
        self.ax.set(xlim=(0,w),ylim=(h,0));self.ax.axis('off');self.ax.patch.set_alpha(0)
        self.texts=[]
    def text(self,x,y,s,size=12,color=INK,weight='normal',**kwargs):
        artist=self.ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,va='top',**kwargs)
        self.texts.append(artist);return artist
    def box(self,x,y,w,h,fill='white',edge=LINE,dash=False,r=14,lw=1):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                         facecolor=fill,edgecolor=edge,linewidth=lw,linestyle=(0,(4,3)) if dash else '-',zorder=2))
    def arrow(self,x1,y1,x2,y2,color=MUTED,both=False,dash=False,curve=0,lw=1.5):
        self.ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='<->' if both else '-|>',
                         mutation_scale=14,linewidth=lw,color=color,linestyle=(0,(4,3)) if dash else '-',
                         connectionstyle=f'arc3,rad={curve}',zorder=4))
    def route(self,points,color=MUTED,dash=False,lw=1.5):
        for a,b in zip(points[:-2],points[1:-1]):
            self.ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,linestyle=(0,(4,3)) if dash else '-',zorder=4)
        self.arrow(*points[-2],*points[-1],color=color,dash=dash,lw=lw)
    def image(self,path,x,y,w,h,foreground=False):
        im=Image.open(path)
        ratio=min(w/im.width,h/im.height);nw,nh=im.width*ratio,im.height*ratio
        x+=(w-nw)/2;y+=(h-nh)/2
        ax=self.fig.add_axes([x/self.w,1-(y+nh)/self.h,nw/self.w,nh/self.h],zorder=20 if foreground else 3)
        ax.imshow(im);ax.axis('off')
        return x,y,nw,nh
    def tag(self,x,y,w,label,color=TEAL,fill='white',size=9):
        self.box(x,y,w,47,fill=fill,edge=fill,r=8)
        self.text(x+w/2,y+11,label,size,color,'bold',ha='center')
    def section(self,x,y,number,title,color=TEAL):
        self.ax.add_patch(Circle((x+22,y+22),22,facecolor=color,zorder=3))
        self.text(x+22,y+5,number,11,'white','bold',ha='center')
        self.text(x+60,y+1,title,14,color=INK,weight='bold')
    def save(self,output,stem):
        self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer()
        for text in self.texts:
            extent=text.get_window_extent(renderer)
            if extent.x0 < 0 or extent.y0 < 0 or extent.x1 > self.w+1 or extent.y1 > self.h+1:
                raise ValueError('Text outside figure: '+text.get_text())
        for ext in ('png','pdf','svg'):
            self.fig.savefig(output/f'{stem}.{ext}',dpi=200,facecolor='white')
        plt.close(self.fig)


class Teaser:
    def __init__(self):
        self.w,self.h=4200,1500
        self.fig=plt.figure(figsize=(21,7.5),dpi=200,facecolor='white')
        self.ax=self.fig.add_axes([0,0,1,1],zorder=20)
        self.ax.set(xlim=(0,self.w),ylim=(self.h,0));self.ax.axis('off');self.ax.patch.set_alpha(0)
        self.texts=[]
    def text(self,x,y,text,size=12,color=INK,weight='normal',**kwargs):
        item=self.ax.text(x,y,text,fontsize=size,color=color,fontweight=weight,va='top',**kwargs)
        self.texts.append(item)
    def box(self,x,y,w,h,color='white',edge='none',radius=10,lw=1):
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                                        facecolor=color,edgecolor=edge,linewidth=lw,zorder=3))
    def image(self,path,box,region=None,layer=3):
        """Native viewport framing; do not rewrite or repaint the source asset."""
        image=Image.open(path)
        x,y,w,h=box
        region=region or (0,0,image.width,image.height)
        x0,y0,x1,y1=region
        # Match viewport aspect by trimming redundant peripheral image content.
        rw,rh=x1-x0,y1-y0
        if rw/rh>w/h:
            trim=(rw-rh*w/h)/2;x0+=trim;x1-=trim
        else:
            trim=(rh-rw*h/w)/2;y0+=trim;y1-=trim
        axis=self.fig.add_axes([x/self.w,1-(y+h)/self.h,w/self.w,h/self.h],zorder=layer)
        axis.imshow(image,extent=(0,image.width,image.height,0))
        axis.set(xlim=(x0,x1),ylim=(y1,y0));axis.axis('off')
    def label(self,x,y,text,w=240,color=INK,dot=False):
        self.box(x,y,w,52,color='white',radius=9)
        if dot:
            self.ax.plot(x+23,y+26,'o',color=ORANGE,markersize=5,zorder=5)
            self.text(x+w/2+9,y+11,text,12,color,ha='center')
        else:self.text(x+18,y+11,text,11,color)
    def arrow(self,start,end,color=TEAL):
        self.ax.add_patch(FancyArrowPatch(start,end,arrowstyle='<->',mutation_scale=14,color=color,linewidth=1.5,zorder=4))
    def save(self,out):
        self.fig.canvas.draw();renderer=self.fig.canvas.get_renderer()
        for item in self.texts:
            r=item.get_window_extent(renderer)
            if r.x0<0 or r.y0<0 or r.x1>self.w or r.y1>self.h:
                raise ValueError('Text extends outside the canvas: '+item.get_text())
        for ext in ['png','pdf','svg']:self.fig.savefig(out/f'teaser_v5.{ext}',dpi=200,facecolor='white')
        plt.close(self.fig)


def system(out, version=5):
    a=out/'assets';c=Canvas(4200,2310)
    c.text(65,42,'AgenticHumanoid — System Overview',31,weight='bold')
    c.text(68,148,'Local feedback executes skills; episode-level reasoning updates programs and symbolic knowledge.',15,MUTED)
    c.section(65,260,'01','SCENE + ROBOT GROUNDING',BLUE)
    c.section(1000,260,'02','ONLINE EXECUTION',TEAL)
    c.section(2890,260,'03','PROGRAM–KNOWLEDGE EVOLUTION',AMBER)
    # Grounded inputs, retained alongside editable research components.
    c.box(65,350,805,1670,fill='#FAFCFD')
    c.image(a/'captured_13s.png',105,405,240,420,foreground=True)
    c.text(380,433,'Real-world video',14,weight='bold')
    c.text(380,506,'Captured office',11,MUTED)
    c.text(380,565,'Images + camera poses',11,MUTED)
    c.text(380,624,'Metric calibration',11,MUTED)
    c.text(380,683,'Known scale assumptions',10.4,MUTED)
    c.arrow(468,850,468,930,BLUE)
    c.box(105,946,725,130,fill='#EAF3FB',edge='#BDD5EA')
    c.text(467,971,'World Labs Atlas',15,BLUE,'bold',ha='center')
    c.text(467,1027,'Reconstruction · scene export',11,ha='center')
    c.image(a/'reconstructed_room.png',95,1110,745,570,foreground=True)
    c.tag(145,1685,645,'3DGS · MJCF / USD · G1 URDF' if version == 6 else 'MJCF / USD · G1 URDF · ASSET HASHES',BLUE,fill='#EDF4FA',size=10)
    c.text(110,1803,'Scene contracts',13,BLUE,'bold')
    c.text(110,1863,'World frame · scene ID · official G1 URDF',10.7)
    c.text(110,1923,'Static furniture has no inferred articulation.',10,MUTED)
    # Fast local loop.
    c.box(1000,350,1770,1335,fill='#FBFDFC',edge='#C9DDD6')
    c.tag(1050,375,1670,'LOCAL LOOP · NO GPT CALLS IN REAL-TIME CONTROL',TEAL,fill='#EAF6F1',size=10)
    c.box(1060,510,495,280,fill='#EDF6FC',edge='#C2DAEA')
    c.text(1090,542,'Onboard observation',13,BLUE,'bold')
    c.text(1090,606,'RGB / optional depth',11)
    c.text(1090,663,'Proprioception + calibration',10.5)
    c.text(1090,720,'Source + timestamp',10.5,MUTED)
    c.arrow(1564,650,1620,650,BLUE)
    c.box(1635,510,520,280,fill='white',edge='#C2DAEA')
    c.text(1665,542,'Belief & predicates',13,BLUE,'bold')
    c.text(1665,606,'Poses: frame / time / confidence',10.2)
    c.text(1665,663,'True · false · unknown',11)
    c.text(1665,720,'Geometric feasibility checks',10.2,MUTED)
    c.arrow(2165,650,2220,650,BLUE)
    c.box(2235,510,475,280,fill='#F8F3FC',edge='#D5C6E4',dash=True)
    c.text(2265,542,'Symbolic planner',12.7,PURPLE,'bold')
    c.text(2265,606,'Bind objects',11)
    c.text(2265,663,'Check all conditions',10.6)
    c.text(2265,720,'Select versioned skill',10.4)
    c.route([(2472,800),(2472,855),(1307,855),(1307,915)],PURPLE,dash=True)
    c.box(1060,930,495,280,fill='#F1F7FD',edge='#BFD5EC')
    c.text(1090,962,'Skill stage runtime',13,BLUE,'bold')
    c.text(1090,1026,'Trusted primitives',11)
    c.text(1090,1083,'Guards + bounded retries',10.5)
    c.text(1090,1140,'No arbitrary generated code',10,MUTED)
    c.arrow(1564,1070,1620,1070,TEAL)
    c.box(1635,930,520,280,fill='white',edge='#C5DAD2')
    c.text(1665,962,'Online references',13,TEAL,'bold')
    c.text(1665,1026,'Future body / hand targets',10.7)
    c.text(1665,1083,'Frame conversion + interpolation',10.0)
    c.text(1665,1140,'Metadata order + history',10.3,MUTED)
    c.arrow(2165,1070,2220,1070,TEAL)
    c.box(2235,930,475,280,fill='#EAF6F0',edge='#B5D3C4')
    c.text(2265,962,'Frozen ScaleBFM',13,TEAL,'bold')
    c.text(2265,1026,'ScaleBridge agent loop',10.5)
    c.text(2265,1083,'Whole-body control',10.9)
    c.text(2265,1140,'50 Hz references',10.3,MUTED)
    c.arrow(2472,1220,2472,1285,TEAL)
    c.box(1060,1300,1650,285,fill='#EEF5F1',edge='#BBD6C8')
    c.text(1100,1334,'MuJoCo / Isaac Sim environment' if version == 6 else 'MuJoCo environment & physical interaction',15,TEAL,'bold')
    c.text(1100,1413,'Gaussian appearance · separate collision geometry · official G1 URDF' if version == 6 else 'Reconstructed office · G1 imported from URDF · authored task props',11.3)
    c.text(1100,1480,'Robot dynamics · contact events · state trajectories · onboard images',11,MUTED)
    c.route([(1060,1465),(975,1465),(975,650),(1047,650)],TEAL)
    c.text(1910,1611,'Observe → plan → execute → observe',11,TEAL,'bold',ha='center')
    c.route([(870,1490),(925,1490),(925,1540),(1048,1540)],BLUE)
    # Explicit evidence lane below the controller.
    c.arrow(1885,1697,1885,1760,TEAL)
    c.box(1000,1775,1770,332,fill='#F4F7FA',edge='#CEDCE6')
    c.text(1040,1808,'Audit trail + fixed independent evaluator',15,weight='bold')
    c.text(1040,1888,'Images · references · robot states · contacts · versioned programs and knowledge',11.5)
    c.text(1040,1950,'Goal reached?  Fall?  Timeout?  Forbidden contact?  Result persists?',11.5)
    c.text(1040,2015,'Evaluation truth is separated from agent perception; scoring is outside GPT edits.',10.5,TEAL)
    # Episode-level agent and two evolving memories.
    c.box(2890,350,1240,350,fill='#F6F1FB',edge='#D3C5E6')
    c.text(2930,386,'GPT proposer / failure diagnosis',17,PURPLE,'bold')
    c.text(2930,470,'Task + current programs / rules + development records',11.5)
    c.text(2930,535,'Hypothesis · program revision · rule revision · next test',11.5)
    c.text(2930,600,'Between episodes; finite API / token / rollout budgets',10.8,MUTED)
    c.arrow(3190,712,3190,777,BLUE)
    c.arrow(3830,712,3830,777,AMBER)
    c.box(2890,792,595,350,fill='#F0F6FD',edge='#BFD5EC')
    c.text(2925,828,'Skill library',15,BLUE,'bold')
    c.text(2925,908,'Editable stage programs',11.5)
    c.text(2925,972,'Targets · order · branches',11)
    c.text(2925,1036,'Exact version + content hash',10.5)
    c.box(3535,792,595,350,fill='#FCF5E9',edge='#E7D1AC')
    c.text(3570,828,'Symbolic knowledge',14,AMBER,'bold')
    c.text(3570,908,'Conditions · add / delete effects',10.6)
    c.text(3570,972,'Dependencies + episode evidence',10.4)
    c.text(3570,1036,'Candidate ≠ validated',11,AMBER)
    c.arrow(3190,1154,3190,1205,BLUE)
    c.arrow(3830,1154,3830,1205,AMBER)
    c.box(2890,1220,1240,170,fill='white',edge='#CBD8E1')
    c.text(2930,1250,'Program & rule validation gate',14,weight='bold')
    c.text(2930,1321,'Schema · bounds · termination · evidence · validated scope',11)
    c.arrow(3510,1402,3510,1452,TEAL)
    c.box(2890,1468,1240,300,fill='#EDF7F2',edge='#B7D6C6')
    c.text(2930,1503,'Execute → compare → promote / retain',16,TEAL,'bold')
    c.text(2930,1585,'Development: proposal feedback   |   Validation: paired selection',11)
    c.text(2930,1650,'Accept program + knowledge together; keep failures and rollback.',10.8)
    c.text(2930,1711,'Test: frozen evaluation; no tuning feedback',10.8,TEAL,'bold')
    c.route([(2770,1910),(2825,1910),(2825,430),(2878,430)],PURPLE)
    c.text(2821,1190,'Episode records',9,PURPLE,rotation=90,ha='center')
    c.route([(4136,1590),(4172,1590),(4172,968),(4140,968)],TEAL)
    c.route([(2890,1040),(2802,1040),(2802,900),(1307,900),(1307,918)],BLUE)
    c.text(1895,862,'Selected skill version',9,BLUE,ha='center')
    c.box(2890,1855,1240,252,fill='#FAF7FD',edge='#CDBDE2',dash=True)
    c.text(2930,1893,'Research extensions',14,PURPLE,'bold')
    c.text(2930,1960,'Contrastive rule experiments · frozen unseen compositions',11)
    c.text(2930,2024,'Drawer / grasp validation · calibrated real-G1 deployment',10.8)
    c.ax.plot([65,4130],[2185,2185],color=LINE,lw=1)
    c.text(65,2222,'Solid: implemented components. Dashed: not end-to-end validated. Simulator figure poses are authored; contact skills remain unvalidated in the captured room.',9.8,MUTED)
    c.save(out,f'system_v{version}')
