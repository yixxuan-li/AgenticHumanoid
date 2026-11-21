"""Actual ScaleBridge/MuJoCo episodes and auditable artifacts."""
import json
import time
import math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .provenance import ROOT, UPSTREAM_COMMIT, canonical_hash, revision, source_hashes, digest
from .skills.schema import check
from .skills.runtime import SkillRuntime
from .perception.state_estimator import StateEstimator


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_scene(output, config):
    import mujoco
    from scene_reconstruction.scalebfm import build_robot_scene
    from .integrations.scalebridge.backend import SOURCE
    robot = SOURCE / "ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof.xml"
    path = output / "scene.xml"
    imported = config.get("scene_directory")
    if imported:
        from .integrations.reconstructed_scene import load_scene
        source_scene = Path(load_scene(imported)["xml"])
    else:
        source_scene = ROOT / "humanoid_skill_lab/scenes/tabletop.xml"
    build_robot_scene(robot, source_scene, path)
    spec = mujoco.MjSpec.from_file(str(path))
    if not imported:
        spec.body("env_cube").pos[:2] = config.get("object_xy", [.45, -.18])
        spec.body("env_target").pos[:2] = config.get("target_xy", [.60, -.18])
    # Calibrated simulation camera: optical forward +x/down relative to torso.
    camera = spec.body("torso_link").add_camera(name="onboard", pos=[.10, 0, .24], fovy=75)
    angle = math.radians(50)
    rotation = np.array([[0, math.sin(angle), -math.cos(angle)], [-1, 0, 0], [0, math.cos(angle), math.sin(angle)]])
    camera.quat = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    spec.compile()
    path.write_text(spec.to_xml())
    return path


def run_episode(program, config, output, device="cuda", video=True, baseline=None, probe=None):
    """One deterministic configuration; no network calls or API credentials."""
    from .integrations.scalebridge.backend import EagerBFMAgent, OnlineEnv, SOURCE
    from .integrations.scalebridge.reference_provider import ReferenceProvider, ReplayProvider
    from .perception.onboard_camera import OnboardCamera
    from .evaluation.oracle import Evaluator
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image
    scene_context = None
    if config.get("scene_directory"):
        from .integrations.reconstructed_scene import load_scene
        scene_context = load_scene(config["scene_directory"])
        if probe is None:
            raise ValueError("Imported scenes support controller probes and the existing navigation workflow. Push exploration needs declared task bindings and an independent scene evaluator; static reconstructed furniture is not a validated manipulation target.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    program = check(program)
    write_json(output / "skill.json", program)
    write_json(output / "knowledge.json", config.get("knowledge", {"status": "candidate", "rules": []}))
    scene = make_scene(output, config)
    mode = 7 if baseline == "replay" else config.get("mode", 2)
    agent = EagerBFMAgent(output, device, mode)
    provider = (ReplayProvider(agent.get_meta_data(), SOURCE / "ScaleBridge/scalebridge/data/motion/g1_29dof/squat.npz")
                if baseline == "replay" else ReferenceProvider(agent.get_meta_data()))
    env = OnlineEnv(agent.get_meta_data(), scene, provider, device,
                    scene_context["robot_start"] if scene_context else None)
    obs_dict = env.reset()
    simulator = env.simulator
    m, d, s = simulator.mujoco_model, simulator.mujoco_data, simulator.state
    if baseline == "replay":
        motion = provider.motion
        q = motion["body_quat_w"][0, 0]
        init = {"root_pos": motion["body_pos_w"][0, 0], "root_quat": q,
                "dof_pos": motion["joint_pos"][0], "dof_vel": motion["joint_vel"][0],
                "root_lin_vel": motion["body_lin_vel_w"][0,0],
                "root_ang_vel": Rotation.from_quat(q, scalar_first=True).inv().apply(motion["body_ang_vel_w"][0,0])}
        simulator.calibrate(init)
        for key, value in simulator.refresh_sim().items():
            env.state_buffer[key+"_buffer"][:] = value.to(device)
        obs_dict = env.refresh_reference()
    runtime = SkillRuntime(program, provider)
    estimator = StateEstimator(config.get("observation_mode", "oracle_debug"), enabled_objects=scene_context is None)
    if scene_context:
        from .evaluation.control import ControlEvaluator
        evaluator = ControlEvaluator(m,d)
    else:
        evaluator = Evaluator(m, d, minimum_height=.20 if baseline == "replay" else .4)
    manifest = {"schema_version": 1, "implementation": "ScaleBridge BaseEnv + BFMAgent.get_action + upstream eager export wrapper",
                "upstream_commit": UPSTREAM_COMMIT, "project_commit": revision(), "source_hashes": source_hashes(),
                "checkpoint_sha256": agent.loader.checkpoint_sha256, "metadata_sha256": agent.loader.metadata_sha256,
                "scene_sha256": digest(scene), "skill_sha256": canonical_hash(program), "config": config,
                "policy_backend": "upstream_eager_pytorch", "tensorrt_artifact_executed": False,
                "device": device, "control_mode": mode, "mode_name": agent.loader.mode_names[mode],
                "physics_hz": 200, "policy_hz": 50, "observation_hz": 10,
                "kinematics_timestamp": "current simulation time; transforms refreshed after integration",
                "camera": {"body": "torso_link", "position_m": [.10, 0, .24], "down_angle_deg": 50, "fovy_deg": 75,
                           "calibration_status": "simulation_only", "depth": True},
                "torch_version": torch.__version__, "mujoco_version": mujoco.__version__, "baseline": baseline,
                "probe": probe,
                "source_scene": scene_context,
                "program_frame_origin_world_m": provider.frame_origin.tolist(),
                "result_status": "running"}
    write_json(output / "manifest.json", manifest)
    refs, local_refs, ref_quat, states, actions, times, errors, events, body_positions, local_quats, policy_actions = ([] for _ in range(11))
    camera, onboard_writer, review_writer, renderer = None, None, None, None
    deadline = config.get("seconds", 18.)
    finished_at = None
    started = time.monotonic()
    observations = (output / "observations.jsonl").open("w")
    (output / "frames").mkdir()
    try:
        camera = OnboardCamera(m, object_bodies={} if scene_context else None)
        if video:
            onboard_writer = imageio.get_writer(output / "onboard.mp4", fps=10)
            review_writer = imageio.get_writer(output / "overview.mp4", fps=10)
            renderer = mujoco.Renderer(m, height=480, width=640)
        belief = None
        for step in range(math.ceil(deadline / env.dt)):
            now = float(d.time)
            if step % 5 == 0:
                observation = camera.observe(d, config.get("observation_mode", "oracle_debug"))
                belief = estimator.update(observation)
                frame_path = f"frames/{step:06d}.jpg"
                Image.fromarray(observation.rgb).save(output / frame_path)
                depth_path = f"frames/{step:06d}_depth.npz"
                np.savez_compressed(output / depth_path, depth_m=observation.depth)
                observations.write(json.dumps({"timestamp": now, "source": observation.source, "rgb": frame_path,
                                               "depth": depth_path, "camera_frame": "optical_right_down_forward",
                                               "intrinsics": observation.intrinsics.tolist(), "world_from_camera": observation.world_from_camera.tolist(),
                                               "proprioception": observation.proprioception,
                                               "poses": {k: vars(v) for k,v in belief.poses.items()},
                                               "predicates": belief.predicates, "confidence": belief.confidence})+"\n")
                if onboard_writer:
                    onboard_writer.append_data(observation.rgb)
                    cam = mujoco.MjvCamera()
                    cam.lookat[:] = provider.frame_origin + [.35, -.02, .70]
                    cam.distance, cam.azimuth, cam.elevation = 2.2, 135, -20
                    renderer.update_scene(d, camera=cam)
                    review_writer.append_data(renderer.render())
            if not baseline:
                runtime.update(now, belief)
            obs_dict = env.refresh_reference()
            wp = obs_dict["body_pos_w_future"][0].cpu().numpy()
            wq = obs_dict["body_quat_w_future"][0].cpu().numpy()
            refs.append(wp.copy())
            ref_quat.append(wq.copy())
            local_refs.append(obs_dict["target_body_pos_future_to_robot_base"][0].cpu().numpy().copy())
            local_quats.append(obs_dict["target_body_rot_future_to_robot_base"][0].cpu().numpy().copy())
            times.append(now)
            states.append(np.r_[d.qpos.copy(), d.qvel.copy()])
            body_positions.append(d.xpos[s.selected_bodies].copy())
            errors.append(np.linalg.norm(wp[0]-d.xpos[s.selected_bodies], axis=1))
            agent.before_step(obs_dict)
            with torch.inference_mode():
                target, action = agent.get_action(obs_dict)
            actions.append(target[0].cpu().numpy())
            policy_actions.append(action[0].cpu().numpy())
            obs_dict = env.step(target, action)
            agent.after_step(obs_dict, action)
            events.extend(simulator.contacts)
            evaluator.update(d, simulator.contacts)
            if evaluator.fallen:
                runtime.outcome, runtime.reason = "failure", "robot_fell"
                break
            if baseline:
                continue
            if runtime.outcome != "running":
                if finished_at is None:
                    finished_at = float(d.time)
                if d.time-finished_at >= 1.1:
                    break
        if baseline and not evaluator.fallen:
            runtime.outcome = "completed"
        metrics = evaluator.report(runtime, float(d.time), deadline)
        errors = np.asarray(errors)
        metrics.update({"wall_seconds": time.monotonic()-started, "tracking_mean_m": errors.mean(axis=0).tolist(),
                        "tracking_max_m": errors.max(axis=0).tolist(), "body_names": agent.get_meta_data()["selected_body_names"],
                        "root_tracking_mean_m": float(errors[:,0].mean()), "right_hand_tracking_mean_m": float(errors[:,-1].mean()),
                        "max_torque_nm": simulator.max_torque, "minimum_root_height_m": simulator.minimum_root_z,
                        "baseline": baseline, "agent_pose_sources": sorted({p.source for p in belief.poses.values()})})
        if baseline:
            metrics["manipulation_success"] = None
            metrics["success"] = bool(not metrics["fallen"] and metrics["root_tracking_mean_m"] < .15)
        if probe:
            if probe not in {"stand", "reach", "height"}:
                raise ValueError("Unknown controller probe")
            actual = np.asarray(body_positions)
            commanded = np.asarray(refs)[:,0]
            diagnostics = {"final_hand_error_m": float(np.linalg.norm(actual[-1,-1]-commanded[-1,-1])),
                           "hand_displacement_m": float(np.linalg.norm(actual[-1,-1]-actual[0,-1])),
                           "root_height_range_m": float(np.ptp(actual[:,0,2])),
                           "root_reference_height_range_m": float(np.ptp(commanded[:,0,2])),
                           "final_root_error_m": float(np.linalg.norm(actual[-1,0]-commanded[-1,0]))}
            criterion = {"stand": diagnostics["final_root_error_m"] < .10,
                         "reach": diagnostics["final_hand_error_m"] < .06 and diagnostics["hand_displacement_m"] > .1,
                         "height": diagnostics["root_height_range_m"] > .025 and diagnostics["final_root_error_m"] < .10}[probe]
            metrics["manipulation_success"] = None
            metrics["probe"] = probe
            metrics["probe_diagnostics"] = diagnostics
            metrics["success"] = bool(criterion and not metrics["fallen"] and not metrics["forbidden_contacts"] and not metrics["timeout"])
        if scene_context:
            from .integrations.reconstructed_scene import load_scene
            if load_scene(scene_context["directory"])["identity"] != scene_context["identity"]:
                raise RuntimeError("Reconstructed scene changed during execution; evidence is invalid")
        write_json(output / "metrics.json", metrics)
        write_json(output / "evaluation_trace.json", evaluator.rows)
        manifest["result_status"] = "executed"
        return metrics
    except Exception as error:
        manifest["result_status"] = "error"
        # This process has no credentials, and these errors concern physics only.
        write_json(output / "error.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        observations.close()
        for resource in (camera, onboard_writer, review_writer, renderer):
            if resource:
                resource.close()
        np.savez_compressed(output / "references.npz", timestamp=times, world_positions=refs, world_quaternions_wxyz=ref_quat,
                            root_positions=local_refs, root_quaternions_wxyz=local_quats)
        np.savez_compressed(output / "robot_states.npz", timestamp=times, qpos_qvel=states, joint_targets=actions,
                            tracking_error_m=errors, body_positions=body_positions, policy_actions=policy_actions, nq=m.nq, nv=m.nv)
        with (output / "events.jsonl").open("w") as stream:
            for event in sorted(runtime.events+events, key=lambda event: event["timestamp"]):
                stream.write(json.dumps(event)+"\n")
        write_json(output / "manifest.json", manifest)
