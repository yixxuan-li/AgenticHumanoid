"""Retain BaseEnv buffers, observation transforms and BFMAgent.get_action.

The portable eager backend uses the upstream export wrapper and released weights.
It is explicitly distinct from running an official precompiled TensorRT artifact.
"""
import os
import sys
import ast
import types
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from humanoid_skill_lab.provenance import ROOT, verify_upstream

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
SOURCE = ROOT / "vendor/ScaleBFM"
verify_upstream()
sys.path.insert(0, str(SOURCE / "ScaleBridge"))
from scalebridge.env.base_env import BaseEnv
def load_bfm_agent():
    # Upstream imports torch_tensorrt unconditionally just to register serialized
    # TensorRT operators. Eager weights need none. Remove only that import from
    # the parsed module; preserve the reviewed upstream class and methods.
    path = SOURCE / "ScaleBridge/scalebridge/agent/bfm_agent.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Import)
                 and [a.name for a in node.names] == ["torch_tensorrt"])]
    module = types.ModuleType("skill_lab_upstream_bfm_agent")
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module.BFMAgent


BFMAgent = load_bfm_agent()
from scalebridge.simulator.base_simulator import BaseSimulator
from scalebridge.simulator.mujoco_simulator import MujocoSimulator
from scalebridge.observation import components
import mujoco
from scene_reconstruction.scalebfm import ScaleBFM, RobotState


class EagerBFMAgent(BFMAgent):
    """Inherit the official get_action argument order; only replace model loading."""
    def __init__(self, output, device="cuda", mode=2):
        self.device = device
        self.control_mode = torch.tensor([mode], dtype=torch.long, device=device)
        self.loader = ScaleBFM(SOURCE, ROOT / "checkpoints/scalebfm_m.pt",
                               ROOT / "checkpoints/scalebfm_m_metadata.json",
                               SOURCE / "ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof.xml",
                               Path(output), device=device)
        self.meta_data_dict = self.loader.metadata.copy()
        self.policy = self.loader.wrapper


class HeadlessSimulator(MujocoSimulator):
    """Upstream stepping with named joint addresses and actuator torque limits.

    Replaces GUI rendering, initialization and index assumptions. No object or
    robot state is written after calibrate. Contacts are sampled at physics rate.
    """
    def _setup_backbone(self):
        BaseSimulator._setup_backbone(self)
        self.mujoco_model = mujoco.MjModel.from_xml_path(self.cfg.asset.xml_path)
        self.mujoco_model.opt.timestep = self.low_dt
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.state = RobotState(self.mujoco_model, self.metadata_dict)
        self.contacts = []
        self.max_torque = 0.0
        self.minimum_root_z = float("inf")

    def calibrate(self, init_state_dict=None):
        m, d, s = self.mujoco_model, self.mujoco_data, self.state
        mujoco.mj_resetData(m, d)
        d.qpos[s.root_qadr:s.root_qadr+7] = [*self.cfg.get("initial_root_position", [0, 0, .76]), 1, 0, 0, 0]
        d.qpos[s.qadr] = self.metadata_dict["default_dof_pos"]
        if init_state_dict:
            d.qpos[s.root_qadr:s.root_qadr+3] = init_state_dict["root_pos"]
            d.qpos[s.root_qadr+3:s.root_qadr+7] = init_state_dict["root_quat"]
            d.qpos[s.qadr] = init_state_dict["dof_pos"]
            d.qvel[s.vadr] = init_state_dict["dof_vel"]
            d.qvel[s.root_vadr:s.root_vadr+3] = init_state_dict["root_lin_vel"]
            d.qvel[s.root_vadr+3:s.root_vadr+6] = init_state_dict["root_ang_vel"]
        mujoco.mj_forward(m, d)
        return d.qpos[:3].copy(), d.qpos[3:7].copy()

    def refresh_sim(self):
        d, s = self.mujoco_data, self.state
        return {k: torch.as_tensor(v.copy(), dtype=torch.float32) for k, v in {
            "root_pos": d.qpos[s.root_qadr:s.root_qadr+3],
            "root_quat_wxyz": d.qpos[s.root_qadr+3:s.root_qadr+7],
            "base_ang_vel": d.qvel[s.root_vadr+3:s.root_vadr+6],
            "dof_pos": d.qpos[s.qadr], "dof_vel": d.qvel[s.vadr]}.items()}

    def apply_action(self, tgt_dof_pos):
        m, d, s = self.mujoco_model, self.mujoco_data, self.state
        if not np.isfinite(tgt_dof_pos).all():
            raise RuntimeError("Nonfinite policy output")
        force = np.zeros(6)
        self.contacts = []
        for _ in range(self.decimation):
            s.apply_pd(d, np.asarray(tgt_dof_pos).reshape(-1))
            self.max_torque = max(self.max_torque, float(np.abs(d.ctrl).max()))
            mujoco.mj_step(m, d)
            self.minimum_root_z = min(self.minimum_root_z, float(d.qpos[s.root_qadr+2]))
            for i, c in enumerate(d.contact):
                mujoco.mj_contactForce(m, d, i, force)
                if force[0] < .1:
                    continue
                self.contacts.append({"timestamp": float(d.time), "geom1": m.geom(c.geom1).name,
                                      "geom2": m.geom(c.geom2).name,
                                      "body1": m.body(int(m.geom_bodyid[c.geom1])).name,
                                      "body2": m.body(int(m.geom_bodyid[c.geom2])).name,
                                      "normal_force": float(force[0])})
        # mj_step integrates qpos after computing transforms. Refresh only the
        # derived kinematics/camera pose, without advancing time or changing
        # velocities/torques, so observation timestamps describe current poses.
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)

    def _render(self):
        pass


class OnlineEnv(BaseEnv):
    """Official buffer management plus a synchronous local reference provider."""
    def __init__(self, metadata, scene, provider, device="cuda", initial_root_position=None):
        self.provider = provider
        self.scene = str(scene)
        self.initial_root_position = initial_root_position or [0,0,.76]
        self.provider.frame_origin = np.array([*self.initial_root_position[:2], 0.], dtype=float)
        names = ["root_quat_buffer", "base_ang_vel_buffer", "dof_pos_buffer", "dof_vel_buffer",
                 "actions_buffer", "body_pos_w_future", "body_quat_w_future",
                 "target_body_pos_future_to_robot_base", "target_body_rot_future_to_robot_base", "future_time_offsets"]
        cfg = OmegaConf.create({"observation": {n: {"func": {"_target_": f"scalebridge.observation.components.{n}",
                                                         "_partial_": True}} for n in names}})
        super().__init__(cfg, metadata, device)

    def _setup_simulator(self):
        cfg = OmegaConf.create({"asset": {"xml_path": self.scene}, "low_dt": .005,
                                "decimation": 4, "joystick": False, "initial_root_position": self.initial_root_position})
        self.simulator = HeadlessSimulator(cfg, self.metadata_dict)
        self.dt = self.simulator.high_dt

    def _setup_observation_manager(self):
        # Fixed code-owned functions; no model-controlled Hydra import targets.
        self.obs_func_names = list(self.cfg.observation)
        self.obs_func_list = [getattr(components, name) for name in self.obs_func_names]

    def _setup_state_manager(self):
        super()._setup_state_manager()
        self._gather_reference()

    def _update_state_manager(self):
        super()._update_state_manager()
        self._gather_reference()

    def _gather_reference(self):
        p, q = self.provider.sample(float(self.simulator.mujoco_data.time), self.metadata_dict["future_idx"])
        expected = (len(self.metadata_dict["future_idx"]), len(self.metadata_dict["selected_body_names"]))
        if p.shape != (*expected, 3) or q.shape != (*expected, 4) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("Reference shape or finite-value contract violated")
        if not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-4):
            raise ValueError("Reference rotations must be unit wxyz quaternions")
        self.state_buffer.update({"body_pos_w_future": torch.tensor(p[None], dtype=torch.float32, device=self.device),
                                 "body_quat_w_wxyz_future": torch.tensor(q[None], dtype=torch.float32, device=self.device),
                                 "future_frame_offset": torch.tensor(self.metadata_dict["future_idx"], device=self.device)[None, :, None]})

    def reset(self):
        obs = super().reset()
        s, d = self.simulator.state, self.simulator.mujoco_data
        self.provider.reset(d.xpos[s.selected_bodies].copy(), d.xquat[s.selected_bodies].copy())
        self._gather_reference()
        return self._update_observation_manager()

    def refresh_reference(self):
        self._gather_reference()
        return self._update_observation_manager()
