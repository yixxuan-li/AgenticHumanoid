"""Source-grounded collision layout fitted to the recovered office geometry.

All dimensions remain provisional until a measured scale anchor is supplied.
The early hand-authored scene is used only as a library of object primitives.
"""
import copy
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .layout import box,cylinder,make_layout


def make_dense_layout(feature_path):
    features=json.loads(Path(feature_path).read_text());library={b['name']:b for b in make_layout()['bodies']}
    shift=np.array([.20,.60,0.]);bodies=[]
    W,D,H=features.get('room_envelope',{}).get('dimensions_m',[6.25,5.90,2.85])
    def body(name,pos,parts,yaw=0,mass=0,kind='fixed',evidence='',**kwargs):
        b=dict(name=name,pos=list(pos),yaw=float(yaw),parts=parts,mass=float(mass),kind=kind,evidence=evidence,**kwargs);bodies.append(b);return b
    def clone(name,template,pos,yaw=None,scales=None,evidence='Source-image and dense-geometry fit'):
        b=copy.deepcopy(library[template]);b.update(name=name,pos=list(pos),evidence=evidence)
        if yaw is not None:b['yaw']=float(yaw)
        if scales is not None:
            scales=np.array(scales,float)
            for p in b['parts']:
                p['pos']=(np.array(p['pos'])*scales).tolist()
                if p['shape']=='box':p['size']=(np.array(p['size'])*scales).tolist()
                else:p['size']=[p['size'][0]*float(np.mean(scales[:2])),p['size'][1]*scales[2]]
        bodies.append(b);return b
    # Main measured-view footprint plus an inferred entry landing. Unknown outside
    # floor is excluded from navigation by collision_grid's support mask.
    body('floor_main',[W/2,(.6+D)/2,-.06],[box('slab',[0,0,0],[W,D-.6,.12],'wood',texture='floor')],evidence='Observed office floor; plane fitted from video')
    body('floor_entry',[(4.6+W)/2,.30,-.06],[box('slab',[0,0,0],[W-4.6,.60,.12],'wood',texture='floor')],evidence='Inferred landing at partly visible entry portal')
    body('wall_left',[-.06,(.6+D)/2,H/2],[box('wall',[0,0,0],[.12,D-.6,H],'wall')],evidence='Left-wall computer desks in late video frames')
    body('wall_window',[W/2,D+.06,H/2],[box('wall',[0,0,0],[W,.12,H],'wall')],evidence='Window wall; backing proxy behind photographed surfaces')
    body('wall_right',[W+.06,(.9+D)/2,H/2],[box('wall',[0,0,0],[.12,D-.9,H],'wall')],evidence='Computer/workbench wall in early frames; entry opening retained')
    body('wall_front_observed',[2.30,.54,H/2],[box('wall',[0,0,0],[4.60,.12,H],'wall')],evidence='Front whiteboard wall seen at end of pan')
    body('wall_entry_left',[4.54,.30,H/2],[box('wall',[0,0,0],[.12,.60,H],'wall')],evidence='Inferred entry closure; outside room not reconstructed')
    body('wall_entry_front',[(4.6+W)/2,-.06,H/2],[box('wall',[0,0,0],[W-4.6,.12,H],'wall')],evidence='Inferred entry closure')
    body('entry_lintel',[W+.06,.45,(2.15+H)/2],[box('lintel',[0,0,0],[.12,.9,H-2.15],'wall')],evidence='Partly visible doorframe; opening height estimated')
    body('ceiling_visual',[W/2,D/2,H-.008],[box('surface',[0,0,0],[W,D,.016],'wall',False)],evidence='Observed ceiling, with unobserved entry extension inferred')
    body('ceiling_window_beam',[W/2,D-.75,H-.10],[box('beam',[0,0,0],[W,.16,.20],'wall')])
    for i,(x,y) in enumerate([(1.6,2.0),(4.8,2.0),(1.6,4.0),(4.8,4.0)]):body(f'ceiling_strip_{i}',[x,y,H-.035],[box('fixture',[0,0,0],[1.15,.075,.025],'lamp',False)],evidence='Strip-light arrangement inferred from video/reflections')
    for i,x in enumerate([1.10,3.18,5.20]):
        window=clone(f'window_{i}',f'window_{i}',[x,5.90-.06,1.78])
        for part in window['parts']:part['collision']=False
        window['evidence']='Captured window appearance with fallback visuals; sealed-window collision is the fitted wall plane. Recess/frame details are simplified.'
    # One central folding table, rather than repeated detections across pan frames.
    table=features['white_tabletop_components'][1]
    # Choose the central horizontal component by its observed location, not count.
    candidates=[c for c in features['white_tabletop_components'] if 1.5<c['center'][0]<4.0 and 1.4<c['center'][1]<3.2 and c['extent'][0]>1.]
    if candidates:table=max(candidates,key=lambda c:c['points'])
    table_xy=np.array(table['center'][:2])+shift[:2];tw=1.76;td=.86
    rect=table.get('oriented_rectangle',{});table_xy=np.array(rect.get('center',table['center'][:2]))+shift[:2]
    tw,td=rect.get('size',[tw,td]);table_yaw=rect.get('yaw_rad',0.)
    central=clone('central_table_0','central_table_0',[*table_xy,0],yaw=table_yaw,scales=[tw/1.8,td/.8,1.],evidence='Single central folding table; source mat and recovered horizontal surface')
    central['recovered_surface_bounds']=table['bounds'];central['dimension_uncertainty_m']=[.10,.08,.025]
    for part in central['parts']:
        if part['name']=='mat':part['texture']='captured_table'
    # Workstations around three walls. Photographic surfaces preserve the actual
    # desk seams and clutter; these are their load-bearing collision proxies.
    for i,y in enumerate([.83,2.48]):
        clone(f'left_desk_{i}','right_desk_0',[.46,y+.60,0],yaw=math.pi/2,scales=[1.65/1.32,.68/.60,1.],evidence='Two left-wall white workstations, frames 25–30')
        clone(f'left_drawers_{i}','right_drawers_0',[.40,y+.60-.43,0],yaw=math.pi/2)
        clone(f'left_monitor_{i}','right_monitor_0',[.35,y+.60,.74],yaw=math.pi/2)
    for i,y in enumerate([2.55,3.91]):
        clone(f'right_desk_{i}','right_desk_0',[5.66,y+.60,0],yaw=-math.pi/2,scales=[1.32/1.32,.68/.60,1.],evidence='Right-wall white workstations visible before window pan')
        clone(f'right_drawers_{i}','right_drawers_0',[5.75,y+.60-.38,0],yaw=-math.pi/2)
        clone(f'right_monitor_{i}','right_monitor_0',[5.80,y+.60,.74],yaw=-math.pi/2)
    clone('right_workbench','left_workbench',[5.75,1.35+.60,0],yaw=-math.pi/2,scales=[1.30/1.55,.68/.72,1.],evidence='Wood workbench beside tall cabinet at start of pan')
    clone('right_workbench_monitor','left_monitor',[5.75,1.35+.60,.74],yaw=-math.pi/2)
    clone('window_counter','window_desk_0',[3.15,4.65+.60,0],yaw=0,scales=[4.95/1.35,.62/.65,1.],evidence='Continuous row of adjacent window workstations; seams remain photographic')
    for i,x in enumerate([1.15,2.45,3.75,5.05]):
        clone(f'window_drawers_{i}','window_drawers_0',[x,4.65+.60,0])
        clone(f'window_monitor_{i}','window_monitor_0',[x+.18,4.73+.60,.74],yaw=0)
    fitted_monitors=features.get('monitor_faces',{})
    if len(fitted_monitors)==8:
        bodies[:]=[b for b in bodies if not (b['name'].startswith(('left_monitor_','right_monitor_','window_monitor_')) or b['name']=='right_workbench_monitor')]
        for name,face in fitted_monitors.items():
            center=np.array(face['center'])+shift;bottom=center[2]-face['height_m']/2
            stand_height=max(.035,bottom-.74)
            parts=[box('panel',[0,0,0],[face['width_m'],.035,face['height_m']],'dark',rotation=[face['tilt_rad'],0,0]),
                cylinder('stand',[0,0,(.74+bottom)/2-center[2]],.018,stand_height,'dark'),box('base',[0,0,.752-center[2]],[.22,.15,.025],'dark')]
            for part in parts:part['visible']=False
            item=body(name,center,parts,yaw=face['yaw_rad'],evidence=f"Monitor face fitted from source frame {face['source_frame']}, pixel ROI {face['source_bbox']}; hidden depth/stand estimated")
            item['source_face_fit']=face
    clone('central_monitor','central_monitor',[3.55+.20,2.40+.60,.74],yaw=math.pi,evidence='Central monitor position from source and recovered photo surface')
    # Four green chairs; two disconnected color patches of the same left chair
    # are merged, and a small green desktop accessory is excluded.
    green=features['green_chairs']
    far=green[0];far_xy=np.array(far.get('back_surface_center',far['center'])[:2])+[0,-.19]+shift[:2]
    near=green[1];near_xy=np.array(near.get('seat_xy',near['center'][:2]))+shift[:2]
    right=min([g for g in green if g['center'][0]>4. and g['extent'][2]>.2],key=lambda g:abs(g['center'][1]-3.1))
    right_xy=np.array(right.get('seat_xy',right['center'][:2]))+shift[:2]
    clone('far_green_chair','far_green_chair_0',[*far_xy,0],yaw=0,evidence='Green chair beyond central table; seat inferred from observed back panel')
    clone('near_green_chair','near_green_chair_0',[*near_xy,0],yaw=near.get('suggested_yaw_rad',math.pi),evidence='Near central green chair, tracked across video frames')
    clone('right_green_chair','near_green_chair_0',[*right_xy,0],yaw=right.get('suggested_yaw_rad',math.pi/2),evidence='Right workstation green chair')
    clone('left_green_chair','near_green_chair_0',[.85+.20,.50+.60,0],yaw=-math.pi/2,evidence='Left workstation green chair; two recovered patches belong to one body')
    for name,feature in [('far_green_chair',far),('near_green_chair',near),('right_green_chair',right),('left_green_chair',None)]:
        chair=next(b for b in bodies if b['name']==name)
        back=next(p for p in chair['parts'] if p['name']=='back')
        back['size'][2]=.22;back['pos'][2]=.67
        for p in chair['parts']:
            if p['name']=='seat':p['pos'][2]=.4025
            if p['name'].startswith('leg_'):p['pos'][2]=.20;p['size'][1]=.40
        if feature is not None:
            xy=np.array(feature.get('back_surface_center',feature.get('back_xy',feature['center']))[:2])+shift[:2]
            delta=np.r_[xy,0]-chair['pos'];local=delta@Rotation.from_euler('z',chair['yaw']).as_matrix()
            back['pos'][:2]=local[:2].tolist()
        chair['back_panel_fit']='Observed green panel approximately 0.22 m high; centroid fitted to recovered back surface'
    for name,feature in [('office_window','office_window'),('office_table_end','office_table_end')]:
        item=features[feature];pose=item.get('focused_pose',{});xy=np.array(pose.get('seat_xy',item['center'][:2]))+shift[:2]
        clone(name,'window_office_chair',[*xy,0],yaw=pose.get('yaw_rad',math.pi/2),evidence='Black office chair from source seat/back surface fit')
    gray=features['gray_chair'];pose=gray.get('focused_pose',{});xy=np.array(pose.get('seat_xy',gray['center'][:2]))+shift[:2]
    graybody=clone('gray_chair','near_green_chair_0',[*xy,0],yaw=pose.get('yaw_rad',-math.pi/2),evidence='Gray stacking chair at left workstation')
    for part in graybody['parts']:
        if part['material']=='green':part['material']='white'
    # Green/white object under the window is a storage bin, not another chair.
    binpos=np.array(features['source_landmarks']['storage_bin']['center'][:2])+shift[:2]
    body('storage_bin',[*binpos,.18],[box('case',[0,0,0],[.44,.36,.36],'olive'),box('lid',[0,0,.19],[.47,.39,.035],'white')],mass=2.,kind='free',evidence='White-lidded storage bin near window corner; hidden interior approximated')
    # The visible broad cabinet face establishes its normal; the unseen depth and
    # door hinge are explicitly hypotheses.
    cabinet=features['source_landmarks']['cabinet_front'];normal=np.array(cabinet['plane_normal'][:2]);normal/=np.linalg.norm(normal)
    if normal[0]>0:normal=-normal
    face=np.array(cabinet['center'][:2]);cabxy=face-.26*normal+shift[:2]
    yaw=math.atan2(normal[1],normal[0])
    clone('tall_cabinet','tall_cabinet',[*cabxy,0],yaw=yaw,evidence='Cabinet observed face fitted; hidden cuboid depth estimated')
    local=np.array([.26,-.37,.06]);R=Rotation.from_euler('z',yaw).as_matrix();doorpos=np.r_[cabxy,0]+R@local
    clone('cabinet_door','cabinet_door',doorpos,yaw=yaw,evidence='Cabinet front/hinge inferred from handle and body; opening is not observed')
    clone('black_shelf','black_shelf',[.55,3.80+.60,.94],yaw=math.pi,evidence='Black vertical storage stack and folded furniture near left/window corner; hidden subdivision approximated')
    yellow=features.get('yellow_wall_cabinet_components',[])
    if yellow:
        face=max(yellow,key=lambda c:c['points']);center=np.array(face['center'])+shift
        cabinet=clone('yellow_wall_cabinet','yellow_wall_cabinet',center,yaw=math.pi,scales=[.32/.40,1.40/1.20,.74/.72],evidence='Yellow wall cupboard from recovered source-color component; closed box collision, hidden interior unknown')
        for part in cabinet['parts']:part['visible']=False
    board=features['source_landmarks']['whiteboard']['center']
    body('whiteboard',[board[0]+.20,.62,1.45],[box('panel',[0,0,0],[2.0,.018,1.10],'white',False,texture='whiteboard')],evidence='Glass writing board on front wall; photographed surface retained')
    clone('cardboard_box','cardboard_box',[2.18,.85,.18],yaw=0,evidence='Open cardboard box at front wall in final video frames')
    observed=features['source_landmarks']['observed_small_box'];op=np.array(observed['center'][:2])+shift[:2]
    body('pick_box',[*op,.755],[box('case',[0,0,0],[.11,.070,.030],'dark',texture='box_label')],mass=.10,kind='free',evidence='Small black rectangular table object, source frame 21; identity and mass unknown',affordances=['pick','place','push'])
    bottle=features['source_landmarks']['window_bottle'];bp=np.array(bottle['center'][:2])+shift[:2]
    body('bottle',[*bp,.90],[cylinder('bottle',[0,0,0],.034,.32,'silver'),cylinder('cap',[0,0,.17],.026,.020,'dark')],mass=.40,kind='free',evidence='Bottle on window counter; transparent geometry approximate',affordances=['pick','place'])
    tx,ty=table_xy
    corner=np.r_[table_xy,0]+Rotation.from_euler('z',table_yaw).apply([tw/2-.035,-td/2+.04,0])
    block_pose=[float(corner[0]),float(corner[1]),.766]
    approach=[float(corner[0]-.15),float(corner[1]-.38),.74]
    body('test_block',block_pose,[box('block',[0,0,0],[.05,.05,.05],'red')],mass=.08,kind='free',evidence='ADDED benchmark prop, absent from source video; declared pose near table front-right corner',affordances=['pick','place'])
    # Desk seams/supports are occluded. Keep support proxies against the wall
    # rather than through the observed gray chair; record these assumptions.
    for b in bodies:
        if b['name'].startswith('left_desk_'):
            for p in b['parts']:
                if p['name'].startswith('leg_'):p['pos'][1]=abs(p['pos'][1])
            b['hidden_support_assumption']='Rear supports against wall; front leg positions unobserved'
        if b['name']=='left_drawers_1':b['pos'][1]=3.80
        if b['name']=='window_monitor_1' and len(fitted_monitors)!=8:b['pos'][0]-=.32
    if 'room_envelope' in features:
        for name,offset in [('wall_right',[6.25-W,0,0]),('wall_window',[0,5.90-D,0])]:
            wall=next(b for b in bodies if b['name']==name)
            visual=copy.deepcopy(wall['parts'][0]);visual['name']='fallback_visual';visual['pos']=(np.array(visual['pos'])+offset).tolist();visual['collision']=False
            wall['parts'][0]['visible']=False;wall['parts'].append(visual)
            wall['visual_fallback']='Kept behind observed photo surfaces; fitted plane supplies collision'
    # Hide only inferred visual proxies which would duplicate captured desk legs
    # or electronics. Their collision shapes remain inspectable in simulator tools.
    for b in bodies:
        if b['kind']=='fixed' and any(s in b['name'] for s in ['desk','counter','monitor','drawers','workbench','shelf']):
            for p in b['parts']:
                if p['name']!='top':p['visible']=False
    return dict(schema_version=1,global_scale=1.,units='m, kg, s, rad',axes='right-handed Z-up; X along window wall; Y toward windows',
        dimensions=[W,D,H],main_room_y_range=[.6,D],desk_height=.74,scale_status='provisional_dense_fit',
        metric_anchor=dict(type='assumed desk/table height',meters=.74,measured=False),
        source_geometry=str(feature_path),source_geometry_shift_m=shift.tolist(),
        room_envelope=features.get('room_envelope',{}),
        uncertainty=dict(metric_anchor_fraction=.10,room_geometry_m=.15,furniture_fit_m=.08,mass_and_friction='assumed from object class; not inferred from RGB'),
        navigation=dict(start=[2.80,1.35],goals=[[4.55,4.00],[1.35,3.80],[4.55,1.60]],radius=.28),
        cameras=[('overview',[3.15,-2.,5.8],[3.15,3.,.7]),('entry',[2.80,1.35,1.55],[3.2,4.8,1.0]),('window_view',[2.85,1.40,1.60],[3.25,5.2,1.4]),('right_view',[2.8,1.35,1.55],[5.8,2.4,1.1])],
        benchmarks=dict(pick=dict(target='test_block',approach=approach,block=block_pose,yaw_rad=math.pi/2,
            approach_evidence='Fitted tabletop front-right corner, with robot-hand offsets from simulated contact-grasp trials',
            controller_profile=dict(robot_root_height_m=.74,settled_root_height_m=.78,wrist_forward_offset_m=.145,wrist_height_offset_m=.07,wrist_lateral_offset_m=.015,pose_servo=True,slow_grasp=True,grasp_seconds=20.))),bodies=bodies)
