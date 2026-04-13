#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill
import numpy as np
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env_runner.dp_runner import DPRunner
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
import hydra


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


def load_policy(ckpt_path: Path, device: str):
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


def main():
    parser = argparse.ArgumentParser(description="Smoke-evaluate a DP checkpoint without planner/env.")
    parser.add_argument(
        "--zarr_path",
        type=Path,
        default=Path("/home/haotian/RobotProject/ADM_DP/data/zarr_data/PickCube_v1_lang_bridge.zarr"),
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("/home/haotian/RobotProject/ADM_DP/policy/Diffusion-Policy/checkpoints/PickCube_v1_lang_bridge/150.ckpt"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_obs_steps", type=int, default=3)
    parser.add_argument("--instruction", type=str, default="Pick up the green cube")
    args = parser.parse_args()

    keys = ["head_camera", "state", "action", "language_emb"]
    rb = ReplayBuffer.copy_from_path(str(args.zarr_path), keys=keys)
    n = rb["action"].shape[0]
    if n < args.n_obs_steps:
        raise RuntimeError(f"Not enough frames in zarr ({n}) for n_obs_steps={args.n_obs_steps}")

    policy = load_policy(args.ckpt_path, args.device)
    runner = DPRunner(output_dir=None, n_obs_steps=args.n_obs_steps, n_action_steps=8)
    use_lang = _policy_uses_language(policy)
    language_emb = _build_language_emb(args.instruction) if use_lang else None

    # Prime obs deque with first n_obs_steps frames
    for i in range(args.n_obs_steps):
        obs = {
            "head_cam": rb["head_camera"][i].astype(np.float32) / 255.0,  # CHW
            "agent_pos": rb["state"][i].astype(np.float32),
        }
        if use_lang:
            if "language_emb" in rb.keys():
                obs["language_emb"] = rb["language_emb"][i].astype(np.float32)
            else:
                obs["language_emb"] = language_emb
        runner.update_obs(obs)

    pred = runner.get_action(policy)
    gt = rb["action"][args.n_obs_steps : args.n_obs_steps + pred.shape[0]]

    print("=== Smoke Eval OK ===")
    print(f"checkpoint: {args.ckpt_path}")
    print(f"dataset: {args.zarr_path}")
    print(f"pred action shape: {pred.shape}")
    print(f"gt action shape:   {gt.shape}")
    print(f"language_conditioning: {use_lang}")
    if gt.shape == pred.shape:
        l2 = np.linalg.norm(pred - gt, axis=1).mean()
        print(f"mean L2(pred,gt): {l2:.6f}")
    print(f"pred sample[0]: {np.array2string(pred[0], precision=4)}")


if __name__ == "__main__":
    main()
