#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    policy = _load_policy(args.ckpt_path, args.device)
    runner = DPRunner(output_dir=None, n_obs_steps=3, n_action_steps=8)
    use_lang = _policy_uses_language(policy)
    language_emb = _build_language_emb(args.instruction) if use_lang else None

    env = gym.make(
        args.env_id,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        max_episode_steps=args.max_steps,
    )
    raw_obs, info = env.reset(seed=args.seed)
    runner.reset_obs()
    runner.update_obs(_build_obs_for_policy(raw_obs, env, language_emb=language_emb))

    frames: List[np.ndarray] = []
    success = False
    step_count = 0

    for t in range(args.max_steps):
        action_seq = runner.get_action(policy)
        action = action_seq[0].astype(np.float32)  # receding horizon: execute first action only
        raw_obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        runner.update_obs(_build_obs_for_policy(raw_obs, env, language_emb=language_emb))

        if args.render_mode == "human":
            env.render()
        else:
            frame = env.render()
            if frame is not None:
                frames.append(_normalize_frame(frame))

        success = _extract_success(info)
        if success or terminated or truncated:
            break

    env.close()

    if args.save_video is not None and len(frames) > 0:
        import imageio.v2 as imageio

        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(args.save_video), frames, fps=20)

    print("=== Real Sim Eval Finished ===")
    print(f"env: {args.env_id}")
    print(f"checkpoint: {args.ckpt_path}")
    print(f"steps: {step_count}")
    print(f"success: {success}")
    print(f"language_conditioning: {use_lang}")
    if args.save_video is not None:
        print(f"video: {args.save_video}")


if __name__ == "__main__":
    main()
