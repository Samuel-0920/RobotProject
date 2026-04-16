#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import dill
import hydra
import mani_skill.envs  # noqa: F401 - register ManiSkill envs
import numpy as np
import torch
import gymnasium as gym

from diffusion_policy.env_runner.dp_runner import DPRunner
from diffusion_policy.workspace.robotworkspace import RobotWorkspace


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _policy_uses_language(policy) -> bool:
    try:
        return "language_emb" in policy.normalizer.params_dict
    except Exception:
        return False


def _build_language_emb(instruction: str) -> np.ndarray:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from motor_adapter import encode_text_emb_for_film

    emb = encode_text_emb_for_film(instruction, device="cpu").detach().cpu().numpy()
    return emb[0].astype(np.float32)


def _load_policy(ckpt_path: Path, device: str):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy


def _as_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _robot_state_vector(env) -> np.ndarray:
    """Match Zarr `state` from replay (ManiSkill `articulations/panda` / robot.get_state), not qpos+qvel+tcp concat."""
    st = _as_np(env.unwrapped.agent.robot.get_state())
    if st.ndim > 1:
        st = st[0]
    return st.astype(np.float32)


def _build_obs_for_policy(
    raw_obs: Dict[str, Any],
    env,
    image_hw: tuple[int, int] = (240, 320),
    language_emb: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    # base_camera rgb: (1, H, W, 3) uint8
    rgb = _as_np(raw_obs["sensor_data"]["base_camera"]["rgb"])[0]
    target_h, target_w = image_hw
    if rgb.shape[0] != target_h or rgb.shape[1] != target_w:
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    head_cam = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0  # CHW

    agent_pos = _robot_state_vector(env)
    assert agent_pos.shape[0] == 31, f"robot state dim should be 31, got {agent_pos.shape[0]}"

    obs = {
        "head_cam": head_cam,
        "agent_pos": agent_pos,
    }
    if language_emb is not None:
        obs["language_emb"] = language_emb
    return obs


def _extract_success(info: Dict[str, Any]) -> bool:
    if "success" not in info:
        return False
    s = info["success"]
    if isinstance(s, torch.Tensor):
        return bool(s.item()) if s.numel() == 1 else bool(torch.any(s).item())
    if isinstance(s, np.ndarray):
        return bool(s.item()) if s.size == 1 else bool(np.any(s))
    return bool(s)


def _normalize_frame(frame: Any) -> np.ndarray:
    arr = _as_np(frame)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr.astype(np.uint8)
    raise ValueError(f"Unexpected render frame shape: {arr.shape}")


def _scalar_bool(x: Any) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.item()) if x.numel() == 1 else bool(torch.any(x).item())
    if isinstance(x, np.ndarray):
        return bool(x.item()) if x.size == 1 else bool(np.any(x))
    return bool(x)


def _step_trace(raw_obs: Dict[str, Any], env, action: np.ndarray, info: Dict[str, Any], t: int) -> Dict[str, Any]:
    extra = raw_obs.get("extra", {})
    tcp_pose = _as_np(extra.get("tcp_pose", np.zeros((1, 7), dtype=np.float32)))
    goal_pos = _as_np(extra.get("goal_pos", np.zeros((1, 3), dtype=np.float32)))
    is_grasped = _as_np(extra.get("is_grasped", np.zeros((1,), dtype=np.float32)))

    tcp_pose = tcp_pose[0] if tcp_pose.ndim > 1 else tcp_pose
    goal_pos = goal_pos[0] if goal_pos.ndim > 1 else goal_pos
    is_grasped = is_grasped.reshape(-1)

    cube_p = _as_np(env.unwrapped.cube.pose.p)
    cube_p = cube_p[0] if cube_p.ndim > 1 else cube_p
    d_tcp_cube = float(np.linalg.norm(tcp_pose[:3] - cube_p))
    dz_tcp_cube = float(abs(float(tcp_pose[2]) - float(cube_p[2])))

    return {
        "step": int(t),
        "tcp_pose": [float(v) for v in tcp_pose.tolist()],
        "cube_pos": [float(v) for v in cube_p.tolist()],
        "goal_pos": [float(v) for v in goal_pos.tolist()],
        "is_grasped": float(is_grasped[0]) if is_grasped.size > 0 else 0.0,
        "success": bool(_extract_success(info)),
        "action": [float(v) for v in action.tolist()],
        "action_gripper": float(action[-1]),
        "d_tcp_cube": d_tcp_cube,
        "dz_tcp_cube": dz_tcp_cube,
    }


def main():
    parser = argparse.ArgumentParser(description="Closed-loop ManiSkill PickCube-v1 evaluation with trained checkpoint.")
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("/home/haotian/RobotProject/ADM_DP/policy/Diffusion-Policy/checkpoints/PickCube_v1_lang_bridge/150.ckpt"),
    )
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=150,
        help="Max env steps per episode. Also passed as gym.make(max_episode_steps=...) so values > 50 override PickCube-v1's default TimeLimit.",
    )
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--save_video", type=Path, default=None, help="Optional mp4 path when render_mode=rgb_array")
    parser.add_argument("--instruction", type=str, default="Pick up the green cube")
    parser.add_argument(
        "--gripper_warmup_steps",
        type=int,
        default=0,
        help="Force gripper action to open value for first N env steps (debug early-close issue).",
    )
    parser.add_argument(
        "--gripper_open_value",
        type=float,
        default=1.0,
        help="Action value for gripper channel during warmup/gating open state (default +1.0).",
    )
    parser.add_argument(
        "--print_step_pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print step-by-step tcp/cube/goal pose and gripper action.",
    )
    parser.add_argument(
        "--step_trace_path",
        type=Path,
        default=None,
        help="Optional JSONL path for per-step traces.",
    )
    parser.add_argument(
        "--gripper_proximity_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only allow closing gripper when tcp is close to cube and height is aligned.",
    )
    parser.add_argument(
        "--gripper_gate_dist",
        type=float,
        default=0.025,
        help="Distance threshold (meters) for tcp-cube proximity gate.",
    )
    parser.add_argument(
        "--gripper_gate_dz",
        type=float,
        default=0.012,
        help="Absolute z-difference threshold (meters) for proximity gate.",
    )
    parser.add_argument(
        "--arm_action_scale",
        type=float,
        default=1.0,
        help="Scale factor for arm action dims (all except last gripper dim). <1.0 makes motion gentler.",
    )
    parser.add_argument(
        "--action_ema_alpha",
        type=float,
        default=0.0,
        help="EMA smoothing factor in [0,1). 0 disables smoothing; higher means smoother/slower action changes.",
    )
    parser.add_argument(
        "--max_action_delta",
        type=float,
        default=0.0,
        help="Per-step max absolute delta for arm dims. 0 disables clipping.",
    )
    args = parser.parse_args()

    policy = _load_policy(args.ckpt_path, args.device)
    runner = DPRunner(output_dir=None, n_obs_steps=3, n_action_steps=8)
    use_lang = _policy_uses_language(policy)
    language_emb = _build_language_emb(args.instruction) if use_lang else None

    env = gym.make(
        args.env_id,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        sensor_configs={"base_camera": {"width": 320, "height": 240}},
        render_mode=args.render_mode,
        max_episode_steps=args.max_steps,
    )
    raw_obs, info = env.reset(seed=args.seed)
    runner.reset_obs()
    runner.update_obs(_build_obs_for_policy(raw_obs, env, language_emb=language_emb))

    frames: List[np.ndarray] = []
    success = False
    step_count = 0
    step_traces: List[Dict[str, Any]] = []
    prev_action: np.ndarray | None = None

    for t in range(args.max_steps):
        action_seq = runner.get_action(policy)
        action = action_seq[0].astype(np.float32)  # receding horizon: execute first action only
        # Apply optional motion shaping to reduce aggressive "hit-and-push" behavior.
        if args.arm_action_scale != 1.0:
            action[:-1] *= np.float32(args.arm_action_scale)
        if prev_action is not None and args.max_action_delta > 0:
            delta = action[:-1] - prev_action[:-1]
            delta = np.clip(delta, -args.max_action_delta, args.max_action_delta)
            action[:-1] = prev_action[:-1] + delta
        if prev_action is not None and args.action_ema_alpha > 0:
            a = np.float32(args.action_ema_alpha)
            action[:-1] = a * prev_action[:-1] + (1.0 - a) * action[:-1]
        eligible_to_close = True
        if t < args.gripper_warmup_steps:
            action[-1] = np.float32(args.gripper_open_value)
        if args.gripper_proximity_gate:
            extra = raw_obs.get("extra", {})
            tcp_pose = _as_np(extra.get("tcp_pose", np.zeros((1, 7), dtype=np.float32)))
            tcp_pose = tcp_pose[0] if tcp_pose.ndim > 1 else tcp_pose
            cube_p = _as_np(env.unwrapped.cube.pose.p)
            cube_p = cube_p[0] if cube_p.ndim > 1 else cube_p
            d_tcp_cube = float(np.linalg.norm(tcp_pose[:3] - cube_p))
            dz = float(abs(float(tcp_pose[2]) - float(cube_p[2])))
            # If not close/aligned enough, force gripper open to avoid early sweeping pushes.
            eligible_to_close = bool(d_tcp_cube <= args.gripper_gate_dist and dz <= args.gripper_gate_dz)
            if not eligible_to_close:
                action[-1] = np.float32(args.gripper_open_value)
        prev_action = action.copy()
        raw_obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        runner.update_obs(_build_obs_for_policy(raw_obs, env, language_emb=language_emb))
        trace = _step_trace(raw_obs=raw_obs, env=env, action=action, info=info, t=t)
        trace["eligible_to_close"] = bool(eligible_to_close)
        step_traces.append(trace)
        if args.print_step_pose:
            print(
                "[STEP {step:03d}] success={success} grasp={is_grasped:.0f} "
                "a7={action_gripper:+.3f} d={d_tcp_cube:.4f} dz={dz_tcp_cube:.4f} "
                "eligible_to_close={eligible_to_close} tcp={tcp_pose} cube={cube_pos} goal={goal_pos}".format(
                    **trace
                )
            )

        if args.render_mode == "human":
            env.render()
        else:
            frame = env.render()
            if frame is not None:
                frames.append(_normalize_frame(frame))

        success = _extract_success(info)
        if success or _scalar_bool(terminated) or _scalar_bool(truncated):
            break

    env.close()

    if args.save_video is not None and len(frames) > 0:
        import imageio.v2 as imageio

        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(args.save_video), frames, fps=20)

    if args.step_trace_path is not None:
        args.step_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with args.step_trace_path.open("w", encoding="utf-8") as f:
            for row in step_traces:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== Real Sim Eval Finished ===")
    print(f"env: {args.env_id}")
    print(f"checkpoint: {args.ckpt_path}")
    print(f"steps: {step_count}")
    print(f"success: {success}")
    print(f"language_conditioning: {use_lang}")
    if args.save_video is not None:
        print(f"video: {args.save_video}")
    if args.step_trace_path is not None:
        print(f"step_trace: {args.step_trace_path}")


if __name__ == "__main__":
    main()
