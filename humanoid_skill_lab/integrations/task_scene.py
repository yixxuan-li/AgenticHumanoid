"""Declared movable props in read-only reconstructed scenes or open workcells."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def validate_task(task):
    if not isinstance(task, dict) or task.get("support") not in {"added_table", "existing_table"}:
        raise ValueError("Task needs a declared added_table or existing_table support")
    for name, size in [("robot_position", 3), ("object_xy", 2), ("target_xy", 2)]:
        value = task.get(name)
        if not isinstance(value, list) or len(value) != size or not np.isfinite(value).all():
            raise ValueError("Invalid task binding: "+name)
    if not np.isfinite(task.get("yaw", np.nan)) or not .4 <= task.get("support_height", 0) <= .9:
        raise ValueError("Invalid task heading or support height")
    if not .6 <= task["robot_position"][2] <= .85:
        raise ValueError("Invalid initial robot height")
    for name in ["object_xy", "target_xy"]:
        x, y = task[name]
        if not .39 <= x <= .85 or not -.34 <= y <= .34:
            raise ValueError("Task object or target outside declared workcell")
    return task


def compose(source, task, output):
    """Author only initial task bodies; the execution loop never sets their state."""
    import mujoco
    task = validate_task(task)
    if source:
        source = Path(source).resolve()
        scene = mujoco.MjSpec.from_file(str(source))
        for mesh in scene.meshes:
            if mesh.file:
                mesh.file = str(source.parent/mesh.file)
        for texture in scene.textures:
            if texture.file:
                texture.file = str(source.parent/texture.file)
        scene.meshdir = scene.texturedir = ""
    else:
        if task["support"] == "existing_table":
            raise ValueError("An existing support requires a source scene")
        scene = mujoco.MjSpec()
        scene.visual.headlight.ambient = [.5]*3
        scene.visual.headlight.diffuse = [.5]*3
        scene.worldbody.add_light(pos=[0, -1, 3], dir=[0, 0, -1], diffuse=[.8]*3)
        scene.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                  size=[10, 10, .05], rgba=[.22, .25, .28, 1], friction=[1, .005, .0001])
    rotation = Rotation.from_euler("z", task["yaw"])
    origin = np.r_[task["robot_position"][:2], 0.]
    quat = rotation.as_quat(scalar_first=True)
    height = task["support_height"]
    def body(name, local, movable=False):
        result = scene.worldbody.add_body(name=name, pos=origin+rotation.apply(local), quat=quat)
        if movable:
            result.add_freejoint(name=name+"_free")
        return result
    def box(parent, name, pos, halfsize, rgba, **kwargs):
        return parent.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos,
                               size=halfsize, rgba=rgba, **kwargs)
    if task["support"] == "added_table":
        table = body("table", [.65, 0, height-.025])
        box(table, "table_top", [0, 0, 0], [.30, .43, .025], [.55, .44, .30, 1], friction=[.6, .005, .0001])
        for index, (x, y) in enumerate([[-.24, -.35], [.24, .35]]):
            box(table, f"table_leg{index}", [x, y, -height/2], [.025, .025, (height-.05)/2], [.3, .3, .3, 1])
    cube = body("cube", [*task["object_xy"], height+.045], movable=True)
    box(cube, "cube_geom", [0, 0, 0], [.04]*3, [.9, .035, .025, 1], mass=.12, friction=[.4, .005, .0001])
    target = body("target", [*task["target_xy"], height+.003])
    box(target, "target_visual", [0, 0, 0], [.065, .065, .001], [.02, .85, .06, .65], contype=0, conaffinity=0)
    for index, (xy, color) in enumerate([([.74, .19], [.04, .15, .9, 1]), ([.47, .25], [.9, .75, .03, 1])]):
        distractor = body(f"distractor_{index}", [*xy, height+.035], movable=True)
        box(distractor, f"distractor_geom_{index}", [0, 0, 0], [.03]*3, color, mass=.08, friction=[.4, .005, .0001])
    scene.visual.global_.offwidth = 640
    scene.visual.global_.offheight = 480
    scene.compile()
    Path(output).write_text(scene.to_xml())
    return Path(output)


def perception_workspace(task):
    """Static workcell calibration, not ground-truth object/target positions."""
    return dict(origin=[*task["robot_position"][:2], 0.], yaw=task["yaw"],
                lower=[.30, -.42, task["support_height"]-.015],
                upper=[.98, .42, task["support_height"]+.18])
