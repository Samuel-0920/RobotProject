#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import yaml
import zarr


def _tuple_shape(x) -> Tuple[int, ...]:
    return tuple(int(v) for v in x)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate zarr data contract against task shape_meta (keys/shapes/dtypes)."
    )
    parser.add_argument(
        "--zarr_path",
        type=Path,
        required=True,
        help="Path to replay zarr dataset.",
    )
    parser.add_argument(
        "--task_config",
        type=Path,
        default=Path(
            "/home/haotian/RobotProject/ADM_DP/policy/Diffusion-Policy/diffusion_policy/config/task/pickcube_lang_minimal.yaml"
        ),
        help="Task yaml containing shape_meta.",
    )
    args = parser.parse_args()

    if not args.zarr_path.exists():
        raise FileNotFoundError(f"zarr not found: {args.zarr_path}")
    if not args.task_config.exists():
        raise FileNotFoundError(f"task config not found: {args.task_config}")

    cfg = yaml.safe_load(args.task_config.read_text())
    obs_meta: Dict = cfg["shape_meta"]["obs"]
    action_meta: Dict = cfg["shape_meta"]["action"]

    root = zarr.open(str(args.zarr_path), mode="r")
    if "data" not in root:
        raise RuntimeError("Invalid zarr: missing group 'data'")
    data = root["data"]

    # Mapping between zarr keys and model/input keys.
    mapping = {
        "head_camera": "head_cam",
        "state": "agent_pos",
        "language_emb": "language_emb",
    }

    ok = True
    print("=== Data Contract Check ===")
    print(f"zarr: {args.zarr_path}")
    print(f"task: {args.task_config}")
    print()

    for z_key, m_key in mapping.items():
        if z_key not in data:
            print(f"[FAIL] missing zarr key: data/{z_key}")
            ok = False
            continue
        if m_key not in obs_meta:
            print(f"[FAIL] missing shape_meta obs key: {m_key}")
            ok = False
            continue
        z_shape = _tuple_shape(data[z_key].shape[1:])
        m_shape = _tuple_shape(obs_meta[m_key]["shape"])
        z_dtype = str(data[z_key].dtype)
        match = z_shape == m_shape
        ok = ok and match
        print(
            f"[{'OK' if match else 'FAIL'}] data/{z_key:12s} -> obs/{m_key:12s} "
            f"shape zarr={z_shape} cfg={m_shape} dtype={z_dtype}"
        )

    # Action check
    if "action" not in data:
        print("[FAIL] missing zarr key: data/action")
        ok = False
    else:
        z_shape = _tuple_shape(data["action"].shape[1:])
        m_shape = _tuple_shape(action_meta["shape"])
        z_dtype = str(data["action"].dtype)
        match = z_shape == m_shape
        ok = ok and match
        print(
            f"[{'OK' if match else 'FAIL'}] data/action        -> action         "
            f"shape zarr={z_shape} cfg={m_shape} dtype={z_dtype}"
        )

    # Meta check
    if "meta" not in root or "episode_ends" not in root["meta"]:
        print("[FAIL] missing meta/episode_ends")
        ok = False
    else:
        eps = root["meta"]["episode_ends"]
        print(f"[OK]  meta/episode_ends shape={tuple(eps.shape)} dtype={eps.dtype}")

    print()
    if ok:
        print("Result: PASS")
        return 0
    print("Result: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

