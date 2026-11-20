"""Generic semantic-shape helpers. Supplied dimensions need explicit provenance.

World: choose a consistent floor frame with +Z up. All lengths [m].
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
