#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np
import zarr


def _add_repo_root_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _iter_datasets(group: h5py.Group, prefix: str = "") -> Iterable[Tuple[str, h5py.Dataset]]:
    for key, value in group.items():
        cur = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Dataset):
            yield cur, value
        elif isinstance(value, h5py.Group):
            yield from _iter_datasets(value, cur)


def _score_image_path(path: str, ds: h5py.Dataset) -> int:
    score = 0
    p = path.lower()
    if "sensor_data" in p:
        score += 28
    if "base_camera" in p:
        score += 18
    if "rgb" in p:
        score += 30
    if "image" in p or "camera" in p:
        score += 15
    if "obs/" in p:
        score += 10
    if ds.dtype == np.uint8:
        score += 10
    if ds.ndim >= 3 and ds.shape[-1] in (3, 4):
        score += 20
    if ds.ndim == 4:
        score += 10
    return score


def _find_image_dataset(traj_group: h5py.Group) -> Optional[str]:
    candidates: List[Tuple[int, str]] = []
    for path, ds in _iter_datasets(traj_group):
        if ds.ndim >= 3 and ds.shape[-1] in (3, 4):
            candidates.append((_score_image_path(path, ds), path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _score_state_path(path: str, ds: h5py.Dataset) -> int:
    """Prefer full robot+task state (31-D panda in PickCube) over obs/agent/qpos (9-D)."""
    score = 0
    p = path.lower()
    if ds.ndim == 2:
        score += 10
        if ds.shape[1] == 31:
            score += 120
        elif ds.shape[1] == 9:
            score += 15
    if "env_states" in p and "articulations" in p:
        score += 50
    if "panda" in p:
        score += 35
    if "qpos" in p and "agent" in p:
        score += 12
    if "joint" in p:
        score += 8
    if "tcp" in p:
        score += 8
    if "state" in p:
        score += 5
    return score


def _find_state_dataset(traj_group: h5py.Group, action_len: int) -> Optional[str]:
    candidates: List[Tuple[int, str]] = []
    for path, ds in _iter_datasets(traj_group):
        if ds.ndim != 2:
            continue
        if ds.shape[0] not in (action_len, action_len + 1):
            continue
        score = _score_state_path(path, ds)
        if score > 0:
            candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _normalize_image_seq(img_seq: np.ndarray, frame_count: int, fallback_hw: Tuple[int, int]) -> np.ndarray:
    if img_seq.ndim == 3 and img_seq.shape[-1] in (3, 4):
        img_seq = img_seq[None, ...]
    if img_seq.ndim != 4:
        h, w = fallback_hw
        return np.zeros((frame_count, h, w, 3), dtype=np.uint8)

    # support NCHW -> NHWC
    if img_seq.shape[-1] not in (3, 4) and img_seq.shape[1] in (3, 4):
        img_seq = np.transpose(img_seq, (0, 2, 3, 1))

    if img_seq.shape[-1] == 4:
        img_seq = img_seq[..., :3]

    if img_seq.shape[0] == frame_count + 1:
        img_seq = img_seq[:frame_count]
    elif img_seq.shape[0] != frame_count:
        h, w = fallback_hw
        return np.zeros((frame_count, h, w, 3), dtype=np.uint8)

    if img_seq.dtype != np.uint8:
        img_seq = np.clip(img_seq, 0, 255).astype(np.uint8)
    return img_seq


def _resize_rgb_sequence(images_nhwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize (N,H,W,3) uint8 to (N,out_h,out_w,3) for DP training resolution."""
    import cv2

    n, h, w, c = images_nhwc.shape
    if h == out_h and w == out_w:
        return images_nhwc
    out = np.empty((n, out_h, out_w, 3), dtype=np.uint8)
    for i in range(n):
        out[i] = cv2.resize(images_nhwc[i], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ManiSkill trajectory.h5 to DP zarr with language embeddings.")
    parser.add_argument(
        "--h5_path",
        type=Path,
        default=Path("~/.maniskill/demos/PickCube-v1/teleop/trajectory.h5").expanduser(),
        help="Input ManiSkill trajectory h5 path.",
    )
    parser.add_argument(
        "--out_zarr",
        type=Path,
        default=Path("ADM_DP/data/zarr_data/PickCube_v1_lang_bridge.zarr"),
        help="Output zarr path.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Pick up the green cube",
        help="Semantic instruction used to generate language embedding.",
    )
    parser.add_argument(
        "--clip_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for CLIP text embedding.",
    )
    parser.add_argument(
        "--fallback_h",
        type=int,
        default=240,
        help="Fallback image height when rgb observations are absent.",
    )
    parser.add_argument(
        "--fallback_w",
        type=int,
        default=320,
        help="Fallback image width when rgb observations are absent.",
    )
    parser.add_argument(
        "--no_resize_rgb",
        action="store_true",
        help="Keep native camera resolution from H5; default is resize RGB to fallback_h x fallback_w (matches pickcube_lang_minimal 240x320).",
    )
    args = parser.parse_args()

    repo_root = _add_repo_root_to_syspath()
    from motor_adapter import encode_text_emb_for_film

    h5_path = args.h5_path.expanduser().resolve()
    out_zarr = (repo_root / args.out_zarr).resolve() if not args.out_zarr.is_absolute() else args.out_zarr.resolve()
    out_zarr.parent.mkdir(parents=True, exist_ok=True)

    phrase = args.instruction.strip()
    if not phrase:
        raise ValueError("Instruction must be non-empty.")
    lang_emb = encode_text_emb_for_film(phrase, device=args.clip_device).detach().cpu().numpy()
    if lang_emb.ndim != 2 or lang_emb.shape[1] != 512:
        raise ValueError(f"Unexpected language embedding shape: {lang_emb.shape}, expected (1, 512)")

    action_list: List[np.ndarray] = []
    image_list: List[np.ndarray] = []
    state_list: List[np.ndarray] = []
    lang_list: List[np.ndarray] = []
    episode_ends: List[int] = []
    total_frames = 0

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_")[1]))
        if not traj_keys:
            raise RuntimeError(f"No traj_* groups found in {h5_path}")

        img_path_used: Optional[str] = None
        state_path_used: Optional[str] = None
        used_fallback_images = False

        for traj_key in traj_keys:
            traj = f[traj_key]
            if "actions" not in traj:
                raise RuntimeError(f"Missing actions in {traj_key}")
            actions = np.asarray(traj["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 8:
                raise RuntimeError(f"{traj_key}/actions shape must be (N, 8), got {actions.shape}")
            n = actions.shape[0]

            img_path = _find_image_dataset(traj)
            if img_path is None:
                used_fallback_images = True
                images = np.zeros((n, args.fallback_h, args.fallback_w, 3), dtype=np.uint8)
            else:
                img_path_used = img_path_used or img_path
                images_raw = np.asarray(traj[img_path])
                images = _normalize_image_seq(
                    images_raw,
                    frame_count=n,
                    fallback_hw=(args.fallback_h, args.fallback_w),
                )
                if np.all(images == 0) and images_raw.size > 0 and images_raw.ndim < 3:
                    used_fallback_images = True
                if (
                    not used_fallback_images
                    and not args.no_resize_rgb
                    and (images.shape[1] != args.fallback_h or images.shape[2] != args.fallback_w)
                ):
                    images = _resize_rgb_sequence(images, args.fallback_h, args.fallback_w)

            state_path = _find_state_dataset(traj, action_len=n)
            if state_path is None:
                raise RuntimeError(f"Cannot find state dataset in {traj_key} with first dim N or N+1.")
            state_path_used = state_path_used or state_path
            states = np.asarray(traj[state_path], dtype=np.float32)
            if states.shape[0] == n + 1:
                states = states[:n]
            if states.shape[0] != n:
                raise RuntimeError(f"{traj_key}/{state_path} cannot align with actions: {states.shape} vs {actions.shape}")

            language = np.repeat(lang_emb, repeats=n, axis=0).astype(np.float32)

            action_list.append(actions)
            image_list.append(images)
            state_list.append(states)
            lang_list.append(language)

            total_frames += n
            episode_ends.append(total_frames)

    actions_all = np.concatenate(action_list, axis=0).astype(np.float32)
    images_all = np.concatenate(image_list, axis=0).astype(np.uint8)
    states_all = np.concatenate(state_list, axis=0).astype(np.float32)
    lang_all = np.concatenate(lang_list, axis=0).astype(np.float32)
    episode_ends_arr = np.asarray(episode_ends, dtype=np.int64)

    sd = states_all.shape[1]
    if sd != 31:
        print(
            f"\nWARNING: state dimension is {sd}, but pickcube_lang_minimal / eval_real_sim expect 31. "
            f"If you used a replayed RGB H5, ensure env_states/articulations/panda exists; "
            f"re-run this script (it prefers panda 31-D over obs/agent/qpos).\n"
        )

    if out_zarr.exists():
        import shutil

        shutil.rmtree(out_zarr)

    root = zarr.open_group(str(out_zarr), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    data.create_dataset(
        "action",
        data=actions_all,
        chunks=(min(1024, actions_all.shape[0]), actions_all.shape[1]),
        dtype="float32",
        compressor=compressor,
    )
    data.create_dataset(
        "obs_image",
        data=images_all,
        chunks=(min(128, images_all.shape[0]), images_all.shape[1], images_all.shape[2], images_all.shape[3]),
        dtype="uint8",
        compressor=compressor,
    )
    data.create_dataset(
        "obs_state",
        data=states_all,
        chunks=(min(1024, states_all.shape[0]), states_all.shape[1]),
        dtype="float32",
        compressor=compressor,
    )
    data.create_dataset(
        "language_emb",
        data=lang_all,
        chunks=(min(1024, lang_all.shape[0]), lang_all.shape[1]),
        dtype="float32",
        compressor=compressor,
    )
    # Keep compatibility with current ADM_DP training loader which expects:
    # data/head_camera (N, 3, H, W) and data/state.
    head_camera_nchw = np.transpose(images_all, (0, 3, 1, 2))
    data.create_dataset(
        "head_camera",
        data=head_camera_nchw,
        chunks=(min(128, head_camera_nchw.shape[0]), head_camera_nchw.shape[1], head_camera_nchw.shape[2], head_camera_nchw.shape[3]),
        dtype="uint8",
        compressor=compressor,
    )
    data.create_dataset(
        "state",
        data=states_all,
        chunks=(min(1024, states_all.shape[0]), states_all.shape[1]),
        dtype="float32",
        compressor=compressor,
    )
    meta.create_dataset("episode_ends", data=episode_ends_arr, dtype="int64", compressor=compressor)

    print("=== Conversion Complete ===")
    print(f"Input H5: {h5_path}")
    print(f"Output zarr: {out_zarr}")
    print(f"Image dataset path used: {img_path_used if img_path_used else 'NONE (fallback black images)'}")
    print(f"State dataset path used: {state_path_used}")
    print(f"Fallback images used: {used_fallback_images}")
    print("=== Zarr Shapes ===")
    print(f"data/action: {actions_all.shape}")
    print(f"data/obs_image: {images_all.shape}")
    print(f"data/obs_state: {states_all.shape}")
    print(f"data/language_emb: {lang_all.shape}")
    print(f"data/head_camera (dp_compat): {head_camera_nchw.shape}")
    print(f"data/state (dp_compat): {states_all.shape}")
    print(f"meta/episode_ends: {episode_ends_arr.shape}")
    if used_fallback_images:
        print(
            "\nTip: 当前 H5 里没有可用的 RGB 序列（常见于采集时 env_kwargs 里 obs_mode=none）。"
            "请先用 ManiSkill 官方回放生成带相机的轨迹，再重新运行本脚本：\n"
            "  python -m mani_skill.trajectory.replay_trajectory \\\n"
            "    --traj-path /path/to/trajectory.h5 --save-traj --obs-mode rgb \\\n"
            "    --use-env-states -c pd_joint_pos\n"
            "会在同目录得到 trajectory.rgb.pd_joint_pos.<backend>.h5，把 --h5_path 指向该文件即可。\n"
        )


if __name__ == "__main__":
    main()
