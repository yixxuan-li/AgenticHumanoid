"""Editable semantic reconstruction. Dimensions are hypotheses, not survey data.

World: +X along window wall, +Y toward windows, +Z up. All lengths [m].
Part coordinates are body-local; body yaw and part rotation are radians.
"""
import math

COLORS = {
    'wall': [.84,.84,.81,1], 'white': [.85,.86,.83,1],
    'steel': [.48,.50,.47,1], 'dark': [.035,.041,.045,1],
    'black': [.016,.020,.022,1], 'wood': [.53,.32,.13,1],
    'green': [.33,.69,.065,1], 'blue': [.27,.40,.54,1],
    'glass': [.09,.13,.14,1], 'cardboard': [.47,.35,.21,1],
    'red': [.60,.025,.035,1], 'silver': [.58,.59,.56,1],
    'lamp': [.95,.95,.90,1],
    'olive': [.48,.54,.28,1],
}


def box(name,pos,size,material='white',collision=True,texture=None,rotation=None):
    return dict(name=name,shape='box',pos=list(pos),size=list(size),material=material,
                collision=collision,texture=texture,rotation=rotation or [0,0,0])


def cylinder(name,pos,radius,height,material='steel',collision=True,rotation=None):
    return dict(name=name,shape='cylinder',pos=list(pos),size=[radius,height],material=material,
                collision=collision,texture=None,rotation=rotation or [0,0,0])


def make_layout(desk_height=.74):
    bodies=[]
    def body(name,pos,parts,yaw=0,mass=0,kind='fixed',evidence='',**kwargs):
        b=dict(name=name,pos=list(pos),yaw=yaw,parts=parts,mass=mass,kind=kind,evidence=evidence,**kwargs)
        bodies.append(b);return b
    # Room dimensions come from desk-count/layout interpretation, pending metric anchor.
    width,depth,height=6.4,4.8,2.85
    body('floor',[width/2,depth/2,-.06],[box('slab',[0,0,0],[width,depth,.12],'wood',texture='floor')],evidence='Visible wood plank flooring throughout video')
    body('wall_left',[-.06,depth/2,height/2],[box('wall',[0,0,0],[.12,depth,height],'wall')])
    body('wall_right',[width+.06,depth/2,height/2],[box('wall',[0,0,0],[.12,depth,height],'wall')])
    body('wall_window',[width/2,depth+.06,height/2],[box('wall',[0,0,0],[width,.12,height],'wall')])
    # Unseen wall is deliberately marked inferred and has a traversable doorway.
    body('wall_entry_left',[.65,-.06,height/2],[box('wall',[0,0,0],[1.3,.12,height],'wall')],evidence='Unseen closure hypothesis')
    body('wall_entry_right',[4.4,-.06,height/2],[box('wall',[0,0,0],[4.0,.12,height],'wall')],evidence='Unseen closure hypothesis')
    body('entry_lintel',[1.85,-.06,2.525],[box('lintel',[0,0,0],[1.1,.12,.65],'wall')],evidence='Unseen doorway hypothesis')
    for x in [.018,width-.018]:
        body(f'skirting_{x}',[x,depth/2,.045],[box('strip',[0,0,0],[.025,depth,.09],'wood')])
    body('window_skirting',[width/2,depth-.012,.045],[box('strip',[0,0,0],[width,.025,.09],'wood')])
    body('ceiling_visual',[width/2,depth/2,height-.008],[box('surface',[0,0,0],[width,depth,.016],'wall',False)],evidence='White ceiling visible at top of pan; hidden in overhead review renders')
    body('ceiling_window_beam',[width/2,4.03,height-.10],[box('beam',[0,0,0],[width,.16,.20],'wall')],evidence='Ceiling beam visible above window wall')
    for i,(x,y) in enumerate([(1.6,1.4),(4.8,1.4),(1.6,3.4),(4.8,3.4)]):
        body(f'ceiling_strip_{i}',[x,y,height-.035],[box('fixture',[0,0,0],[1.15,.075,.025],'lamp',False)],evidence='Fluorescent strips inferred from ceiling and window reflections')
    # Windows are opaque night glass, matching the recorded nighttime appearance.
    for i,x in enumerate([1.10,3.18,5.26]):
        parts=[box('night_glass',[0,0,0],[1.84,.035,1.35],'glass',False)]
        for dx in [-.95,0,.95]:parts.append(box(f'mullion_{dx}',[dx,-.028,0],[.035,.065,1.44],'silver'))
        for z in [-.70,.70]:parts.append(box(f'rail_{z}',[0,-.028,z],[1.94,.065,.04],'silver'))
        parts += [box('sill',[0,-.10,-.72],[2.0,.24,.06]),box('blind',[0,-.065,.53],[1.94,.035,.42],'blue',False)]
        for j in range(20):parts.append(box(f'blind_slat_{j}',[0,-.086,.33+j*.020],[1.94,.008,.004],'steel',False))
        parts.append(box('blind_header',[0,-.055,.78],[1.98,.07,.07],'blue',False))
        body(f'window_{i}',[x,depth-.06,1.78],parts,evidence='Frames 8–24; three window bays, metal frames and blue blinds')

    def desk(name,pos,w=1.35,d=.65,material='wood',yaw=0):
        parts=[box('top',[0,0,desk_height-.018],[w,d,.036],material,texture='desktop' if material=='wood' else None)]
        for x in [-w/2+.06,w/2-.06]:
            for y in [-d/2+.055,d/2-.055]:parts.append(box(f'leg_{x}_{y}',[x,y,(desk_height-.036)/2],[.035,.035,desk_height-.036],'dark' if material=='wood' else 'white'))
        parts.append(box('rear_rail',[0,d/2-.06,.57],[w-.10,.025,.075],'dark' if material=='wood' else 'white'))
        return body(name,pos,parts,yaw,evidence='Desk dimensions estimated from conventional 0.74 m height')

    def pedestal(name,pos):
        parts=[box('case',[0,0,.32],[.39,.51,.64])]
        for j in range(3):
            parts += [box(f'drawer_front_{j}',[0,-.261,.14+j*.19],[.37,.022,.18]),box(f'handle_{j}',[0,-.279,.19+j*.19],[.14,.013,.009],'steel',False)]
        body(name,pos,parts,evidence='White drawer units under window desks')

    def monitor(name,pos,yaw=0,on=False):
        parts=[box('base',[0,0,.012],[.21,.16,.024],'dark'),box('stem',[0,.025,.12],[.035,.035,.22],'black'),
               box('bezel',[0,.035,.30],[.49,.038,.29],'black'),box('display',[0,.013,.30],[.461,.005,.259],'dark',False,texture='screen' if on else None)]
        body(name,pos,parts,yaw,evidence='Visible flat-panel monitors; screen imagery sampled from video')

    def keyboard(name,pos,yaw=0):
        parts=[box('base',[0,0,.012],[.33,.12,.024],'dark',False)]
        for j in range(5):
            for k in range(14):parts.append(box(f'key_{j}_{k}',[-.151+k*.023,-.047+j*.023,.026],[.018,.017,.007],'black',False))
        body(name,pos,parts,yaw)

    def green_chair(name,pos,yaw=0):
        parts=[box('seat',[0,0,.455],[.43,.40,.035],'green'),box('back',[0,.19,.66],[.43,.027,.35],'green',rotation=[-.10,0,0])]
        for part in parts:part['visual_style']='rounded'
        for x in [-.18,.18]:
            for y in [-.16,.16]:parts.append(cylinder(f'leg_{x}_{y}',[x,y,.22],.012,.44))
        body(name,pos,parts,yaw,4.0,'free',evidence='Lime green stacking chairs, frames 9–30')

    def office_chair(name,pos,yaw=0):
        parts=[box('seat',[0,0,.46],[.47,.44,.075],'dark'),box('mesh_back_collision',[0,.19,.76],[.47,.045,.49],'dark'),cylinder('post',[0,0,.27],.028,.33,'steel')]
        parts[0]['visual_style']='rounded';parts[1]['visible']=False
        for j,x in enumerate(np_range(-.22,.22,27)):
            parts.append(box(f'mesh_vertical_{j}',[x,.19,.76],[.002,.002,.46],'black',False))
        for j,z in enumerate(np_range(.53,.99,29)):
            parts.append(box(f'mesh_horizontal_{j}',[0,.19,z],[.44,.002,.002],'black',False))
        for x in [-.235,.235]:parts.append(box(f'back_frame_{x}',[x,.19,.76],[.025,.035,.49],'black'))
        for z in [.515,1.005]:parts.append(box(f'back_rail_{z}',[0,.19,z],[.47,.035,.025],'black'))
        for x in [-.29,.29]:
            parts += [box(f'arm_{x}',[x,0,.66],[.055,.30,.045],'black'),box(f'arm_post_{x}',[x,.1,.56],[.025,.025,.18],'black')]
        for j in range(5):
            a=j*2*math.pi/5; dx,dy=.18*math.cos(a),.18*math.sin(a)
            parts += [box(f'spoke_{j}',[dx,dy,.10],[.36,.035,.035],'black',rotation=[0,0,a]),cylinder(f'caster_{j}',[1.75*dx,1.75*dy,.035],.035,.045,'black',rotation=[math.pi/2,0,a])]
        body(name,pos,parts,yaw,12,'free',evidence='Black five-caster office chairs')

    for i,x in enumerate([1.12,2.49,3.86,5.23]):
        desk(f'window_desk_{i}',[x,4.40,0])
        pedestal(f'window_drawers_{i}',[x-.40,4.40,0])
        monitor(f'window_monitor_{i}',[x+.12,4.50,desk_height],on=i in [0,2])
        keyboard(f'window_keyboard_{i}',[x+.10,4.23,desk_height])
        body(f'pc_tower_{i}',[x+.46,4.46,.27],[box('case',[0,0,0],[.20,.43,.54],'dark')])
        if i%2:green_chair(f'window_chair_{i}',[x+.08,3.67,0],math.pi)
    office_chair('window_office_chair',[1.25,3.65,0],math.pi+.25)

    # Central row: printed fiducial work mats, black folding legs.
    for i,x in enumerate([1.36,3.18,5.00]):
        w,d=1.80,.80
        parts=[box('top',[0,0,desk_height-.02],[w,d,.04],'white'),box('mat',[0,0,desk_height+.001],[w-.014,d-.014,.002],'white',False,texture='fiducial')]
        for dx in [-.70,.70]:
            for dy in [-.30,.30]:parts.append(box(f'leg_{dx}_{dy}',[dx,dy,.35],[.027,.027,.70],'dark',rotation=[0,math.copysign(.09,dx),0]))
        parts.append(box('rail',[0,0,.55],[1.45,.025,.035],'dark'))
        body(f'central_table_{i}',[x,2.37,0],parts,evidence='Three folding tables and fiducial mats, frames 13–25')
        green_chair(f'near_green_chair_{i}',[x,1.58,0])
        if i!=2:green_chair(f'far_green_chair_{i}',[x,3.10,0],math.pi)
    monitor('central_monitor',[2.95,2.43,desk_height],yaw=-.3)
    office_chair('central_office_chair',[4.05,.55,0],1.15)

    # Right-wall white desks and overhead yellow storage, seen late in the pan.
    for i,y in enumerate([1.36,2.74]):
        desk(f'right_desk_{i}',[6.03,y,0],w=1.32,d=.60,material='white',yaw=-math.pi/2)
        monitor(f'right_monitor_{i}',[6.10,y+.08,desk_height],yaw=-math.pi/2)
        pedestal(f'right_drawers_{i}',[6.04,y-.38,0])
    green_chair('right_green_chair',[5.55,.65,0],math.pi/2)
    office_chair('right_office_chair',[4.75,3.27,0],1.3)
    body('yellow_wall_cabinet',[6.19,3.88,2.15],[box('case',[0,0,0],[.40,1.20,.72]),box('yellow_front',[-.209,0,0],[.018,1.18,.70],'cardboard')],evidence='Yellow wall cupboard, frames 22–27')
    body('black_shelf',[6.04,4.15,.94],[box('back',[.10,0,0],[.025,.55,1.86],'dark')]+[box(f'shelf_{j}',[0,0,-.85+j*.34],[.40,.57,.035],'black') for j in range(6)]+[box(f'upright_{j}',[0,j*.54-.27,0],[.045,.035,1.86],'black') for j in range(2)],evidence='Equipment rack at corner')

    # Cabinet hinge is explicit and has a physical door panel and handle.
    body('tall_cabinet',[.38,1.45,0],[box('back',[-.23,0,.93],[.035,.72,1.86],'steel'),box('left',[0,-.36,.93],[.48,.025,1.86],'steel'),box('right',[0,.36,.93],[.48,.025,1.86],'steel'),box('top',[0,0,1.85],[.48,.74,.035],'steel'),box('base',[0,0,.04],[.48,.74,.08],'steel')],evidence='Tall gray cabinet, beginning of pan')
    body('cabinet_door',[.64,1.08,.06],[box('panel',[0,.36,.90],[.025,.72,1.80],'steel'),box('handle',[.032,.63,.91],[.025,.025,.17],'dark')],mass=6,kind='hinge',joint=dict(axis=[0,0,1],range=[-1.6,0],damping=1.2,friction=.2),evidence='Articulation is inferred from a visible cabinet door')
    desk('left_workbench',[.43,2.85,0],w=1.55,d=.72,yaw=math.pi/2)
    monitor('left_monitor',[.32,2.85,desk_height],yaw=math.pi/2,on=True)
    body('red_bag',[.38,1.45,1.98],[box('bag',[0,0,0],[.24,.23,.25],'red')],evidence='Red item on cabinet; simplified visual proxy')
    # Whiteboard texture is derived only from source pixels, never generated content.
    body('whiteboard',[6.365,.65,1.60],[box('glass',[0,0,0],[.014,1.60,1.00],'white',False,texture='whiteboard')],evidence='Board in frames 30–34; approximate full extent')
    body('cardboard_box',[6.08,.44,.18],[box('bottom',[0,0,-.17],[.42,.48,.02],'cardboard'),box('side_a',[-.20,0,0],[.02,.48,.34],'cardboard'),box('side_b',[.20,0,0],[.02,.48,.34],'cardboard'),box('side_c',[0,-.23,0],[.42,.02,.34],'cardboard'),box('side_d',[0,.23,0],[.42,.02,.34],'cardboard')],mass=.5,kind='free',evidence='Open cardboard box beneath board')
    body('pick_box',[4.95,2.22,desk_height+.043],[box('tissue_box',[0,0,0],[.13,.085,.08],'white',texture='box_label')],mass=.12,kind='free',evidence='Small rectangular object on table in frame 24; contents/mass unknown',affordances=['pick','place','push'])
    body('bottle',[1.0,4.18,desk_height+.105],[cylinder('bottle',[0,0,0],.032,.20,'silver'),cylinder('cap',[0,0,.109],.025,.018,'dark')],mass=.30,kind='free',evidence='Bottle-like clutter on desk; dimensions/mass estimated',affordances=['pick','place'])
    body('test_block',[4.93,2.50,desk_height+.026],[box('block',[0,0,0],[.05,.05,.05],'red')],mass=.08,kind='free',evidence='ADDED benchmark prop, not reconstructed from video',affordances=['pick','place'])
    return dict(schema_version=1,global_scale=1.,units='m, kg, s, rad',axes='right-handed Z-up; X along window wall; Y toward windows',
                dimensions=[width,depth,height],desk_height=desk_height,scale_status='provisional',
                metric_anchor=dict(type='assumed conventional desk height',meters=desk_height,measured=False),
                uncertainty=dict(room_dimensions_fraction=.25,furniture_dimensions_fraction=.15,mass_and_friction='assumed, not recoverable from RGB video'),
                navigation=dict(start=[1.85,.65],goals=[[.95,3.15],[4.90,3.45],[5.0,.65]],radius=.28),
                bodies=bodies)


def np_range(start,stop,count):
    return [start+(stop-start)*i/(count-1) for i in range(count)]
