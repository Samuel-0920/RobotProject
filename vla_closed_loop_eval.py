#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dill
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_paths() -> None:
    root = _repo_root()
    dp_root = root / "ADM_DP" / "policy" / "Diffusion-Policy"
    for p in (root, dp_root):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


_ensure_paths()

from cognitive_brain import CognitiveBrain, class_for_id
from diffusion_policy.env_runner.dp_runner import DPRunner
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from motor_adapter import build_semantic_phrase_for_film, encode_text_emb_for_film
from perception_engine import ManiSkillPerceptionWrapper, format_reference_library_for_vlm


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_policy(ckpt_path: Path, device: str):
    import hydra

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


def _robot_state_for_policy(env) -> np.ndarray:
    """Align with Zarr `state` / replay `env_states/.../panda` (robot.get_state), not qpos+qvel+tcp concat."""
    st = _to_np(env.unwrapped.agent.robot.get_state())
    if st.ndim > 1:
        st = st[0]
    return st.astype(np.float32)


def _obs_to_policy(raw_obs: Dict[str, Any], env, image_hw: Tuple[int, int] = (240, 320)) -> Dict[str, np.ndarray]:
    import cv2

    rgb = _to_np(raw_obs["sensor_data"]["base_camera"]["rgb"])[0]
    h, w = image_hw
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    head_cam = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0

    state = _robot_state_for_policy(env)
    return {"head_cam": head_cam, "agent_pos": state}


def _extract_success(info: Dict[str, Any]) -> bool:
    if "success" not in info:
        return False
    s = info["success"]
    if isinstance(s, torch.Tensor):
        return bool(s.item()) if s.numel() == 1 else bool(torch.any(s).item())
    if isinstance(s, np.ndarray):
        return bool(s.item()) if s.size == 1 else bool(np.any(s))
    return bool(s)


def _scalar_bool(x: Any) -> bool:
    """ManiSkill / Gymnasium may return terminated, truncated as torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return bool(x.item()) if x.numel() == 1 else bool(torch.any(x).item())
    if isinstance(x, np.ndarray):
        return bool(x.item()) if x.size == 1 else bool(np.any(x))
    if isinstance(x, np.bool_):
        return bool(x)
    return bool(x)


_human_render_warned = False


def _fetch_vllm_served_model_ids(base_url: str, api_key: str) -> List[str]:
    """GET {base}/v1/models -> OpenAI-style model ids (must match chat `model` field)."""
    url = base_url.rstrip("/") + "/v1/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        return [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return []


def _resolve_vlm_model_id(base_url: str, api_key: str, requested: str) -> str:
    """
    Align --vlm_model with vLLM --served-model-name.
    If only one model is served and `requested` is wrong, auto-pick it (fixes 404).
    """
    ids = _fetch_vllm_served_model_ids(base_url, api_key)
    if not ids:
        return requested
    if requested in ids:
        return requested
    if len(ids) == 1:
        print(
            f"[VLM] --vlm_model={requested!r} 未在 /v1/models 中；"
            f"当前仅注册 {ids[0]!r}，已自动改用（请与 vLLM --served-model-name 一致）。",
            flush=True,
        )
        return ids[0]
    print(
        f"[VLM] --vlm_model={requested!r} 不存在。已注册模型: {ids}。"
        f"请任选其一作为 --vlm_model，或与启动时的 --served-model-name 一致。",
        flush=True,
    )
    return requested


def _safe_env_render(*, env, render_mode: str, frames: List[np.ndarray]) -> None:
    """human 模式在无 DISPLAY / viewer 未创建时会抛错；捕获后跳过以免评测中断。"""
    global _human_render_warned
    if render_mode == "human":
        try:
            env.render()
        except Exception as e:
            if not _human_render_warned:
                _human_render_warned = True
                print(
                    f"[VLA] render_mode=human 但窗口渲染失败（常见于 SSH/无 DISPLAY），已跳过 human render: {e}",
                    flush=True,
                )
        return
    try:
        frame = env.render()
        if frame is not None:
            arr = _to_np(frame)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            frames.append(arr.astype(np.uint8))
    except Exception:
        pass


def _rgb_depth_for_perception(raw_obs: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    rgb = _to_np(raw_obs["sensor_data"]["base_camera"]["rgb"])[0]
    depth = None
    depth_raw = raw_obs["sensor_data"]["base_camera"].get("depth")
    if depth_raw is not None:
        # ManiSkill depth is int16 in millimeters
        depth_np = _to_np(depth_raw)[0]
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
        depth = depth_np.astype(np.uint16, copy=False)
    return rgb, depth


def _waypoint_from_obs(raw_obs: Dict[str, Any]) -> Dict[str, float]:
    tcp_pose = _to_np(raw_obs["extra"]["tcp_pose"])[0]  # [x,y,z,qw,qx,qy,qz]
    return {
        "world_pos": [float(tcp_pose[0]), float(tcp_pose[1]), float(tcp_pose[2])],
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }


# COCO/YOLO 在桌面仿真里常见误检；不参与 manip 任务，避免 VLM 选错 target_id。
YOLO_SPURIOUS_CLASS_NAMES = frozenset(
    {
        "traffic light",
        "stop sign",
        "fire hydrant",
        "parking meter",
        "bench",
        "baseball bat",
        "tennis racket",
        "skateboard",
        "surfboard",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "kite",
        "bird",
        "airplane",
        "boat",
        "toothbrush",
    }
)


def _scene_json_from_sim_pickcube(env) -> Dict[str, Any]:
    """
    Build scene JSON from simulator ground truth (no YOLO/SAM2).
    Works with ManiSkill PickCube-* envs that expose `.cube` and `.goal_site`.
    """
    u = env.unwrapped
    if not (hasattr(u, "cube") and hasattr(u, "goal_site")):
        return format_reference_library_for_vlm({})
    try:
        cp = u.cube.pose.p
        gp = u.goal_site.pose.p
        if isinstance(cp, torch.Tensor):
            cp = cp.detach().cpu().numpy()
        if isinstance(gp, torch.Tensor):
            gp = gp.detach().cpu().numpy()
        if cp.ndim > 1:
            cp = cp[0]
        if gp.ndim > 1:
            gp = gp[0]
        cx, cy, cz = float(cp[0]), float(cp[1]), float(cp[2])
        gx, gy, gz = float(gp[0]), float(gp[1]), float(gp[2])
        objects: List[Dict[str, Any]] = [
            {
                "id": "green_cube",
                "class_name": "cube",
                "centroid_3d_m": [cx, cy, cz],
            },
            {
                "id": "goal_target",
                "class_name": "goal",
                "centroid_3d_m": [gx, gy, gz],
            },
        ]
        return {"objects": objects, "count": len(objects)}
    except Exception:
        return format_reference_library_for_vlm({})


def _filter_spurious_scene_objects(scene_json: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    objs = list(scene_json.get("objects") or [])
    kept: List[Dict[str, Any]] = []
    removed_classes: List[str] = []
    for o in objs:
        cn = str(o.get("class_name", "")).strip().lower()
        if cn in YOLO_SPURIOUS_CLASS_NAMES:
            removed_classes.append(cn)
            continue
        kept.append(o)
    return {"objects": kept, "count": len(kept)}, removed_classes


def _sanitize_reflect_for_policy(
    reflect: Dict[str, Any],
    scene_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Map free-form VLM actions to DP/CLIP-friendly verbs; drop invalid target_id.
    """
    out = {k: str(v) if v is not None else "" for k, v in reflect.items()}
    valid_ids = {str(o.get("id", "")) for o in (scene_json.get("objects") or [])}

    act = str(out.get("action", "grasp")).strip().lower().replace(" ", "").replace("_", "")
    if act in ("tryagain", "retry", "again", "regrasp", "pickup", "pick", "take"):
        out["action"] = "grasp"
    elif act in ("place", "putdown", "put", "release", "drop"):
        out["action"] = "place"
    elif act == "grasp":
        out["action"] = "grasp"
    else:
        out["action"] = "grasp"

    macro = str(out.get("macro_action", "")).strip()
    macro_l = macro.lower()
    _allowed_macro = frozenset(("", "reset", "relocalize", "abort"))
    if len(macro) > 36 or macro.count(" ") >= 4:
        out["macro_action"] = ""
    elif macro_l not in _allowed_macro:
        out["macro_action"] = ""
    tid = str(out.get("target_id", "")).strip()
    if tid and tid not in valid_ids:
        out["target_id"] = ""
        tid = ""
    if tid:
        cn = class_for_id(scene_json, tid).strip().lower()
        if cn in YOLO_SPURIOUS_CLASS_NAMES:
            out["target_id"] = ""
    return out


def _local_vlm_reflect_with_image(
    *,
    image_rgb: np.ndarray,
    scene_objects: Dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Call local vLLM with image+scene prompt. Returns (parsed dict or None, error string or None)."""
    try:
        import cv2
        from openai import OpenAI
    except Exception as e:
        return None, f"import_failed: {e}"

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return None, "jpeg_encode_failed"

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    scene_str = json.dumps(scene_objects, ensure_ascii=False)
    user_text = (
        "I tried to pick up the green cube but failed. "
        "Based on the image and object list, what should I do?\n\n"
        f"Scene objects JSON:\n{scene_str}\n\n"
        "Return JSON only with keys: action, target_id, macro_action."
    )
    try:
        client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/v1")
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a robot failure recovery planner. Output JSON only."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        start, end = txt.find("{"), txt.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(txt[start : end + 1])
            return {
                "action": str(data.get("action", "grasp")),
                "target_id": str(data.get("target_id", "")),
                "macro_action": str(data.get("macro_action", "")),
            }, None
        return None, "no_json_in_model_reply"
    except Exception as e:
        return None, str(e)


def _policy_supports_language(policy) -> bool:
    try:
        return "language_emb" in policy.normalizer.params_dict
    except Exception:
        return False


def _inject_language_in_obs(obs: Dict[str, np.ndarray], lang_emb: Optional[np.ndarray], enabled: bool) -> Dict[str, np.ndarray]:
    if enabled and lang_emb is not None:
        obs = dict(obs)
        obs["language_emb"] = lang_emb.astype(np.float32)
    return obs


def _run_phase(
    *,
    env,
    policy,
    runner: DPRunner,
    start_obs: Dict[str, Any],
    steps: int,
    language_emb: Optional[np.ndarray],
    lang_enabled: bool,
    render_mode: str,
    frames: List[np.ndarray],
) -> Tuple[Dict[str, Any], Dict[str, Any], bool, int, bool, bool]:
    """Returns (raw_obs, info, success, executed, terminated, truncated) from last step."""
    raw_obs, info = start_obs, {}
    success = False
    executed = 0
    terminated = False
    truncated = False
    for _ in range(steps):
        action_seq = runner.get_action(policy)
        action = action_seq[0].astype(np.float32)
        raw_obs, reward, terminated, truncated, info = env.step(action)
        terminated, truncated = _scalar_bool(terminated), _scalar_bool(truncated)
        obs = _obs_to_policy(raw_obs, env)
        runner.update_obs(_inject_language_in_obs(obs, language_emb, lang_enabled))
        _safe_env_render(env=env, render_mode=render_mode, frames=frames)
        success = _extract_success(info)
        executed += 1
        if success or terminated or truncated:
            break
    return raw_obs, info, success, executed, terminated, truncated


def main():
    parser = argparse.ArgumentParser(description="3+1 architecture closed-loop evaluation with failure reflection.")
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=Path("ADM_DP/policy/Diffusion-Policy/checkpoints/PickCube_v1_lang_bridge/150.ckpt"),
    )
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--phase_steps", type=int, default=100)
    parser.add_argument("--render_mode", type=str, default="rgb_array", choices=["rgb_array", "human"])
    parser.add_argument(
        "--video_path",
        type=Path,
        default=Path("ADM_DP/policy/Diffusion-Policy/eval_video/vla_closed_loop_eval.mp4"),
    )
    parser.add_argument(
        "--reflect_log_path",
        type=Path,
        default=Path("ADM_DP/policy/Diffusion-Policy/eval_video/vla_reflection_log.jsonl"),
    )
    parser.add_argument("--vlm_base_url", type=str, default="http://localhost:11435")
    parser.add_argument(
        "--vlm_model",
        type=str,
        default="Qwen2-VL-7B-Instruct-AWQ",
        help="Must match vLLM --served-model-name (e.g. Qwen2-VL-7B-Instruct-AWQ).",
    )
    parser.add_argument("--instruction", type=str, default="Pick up the green cube")
    parser.add_argument(
        "--scene_source",
        type=str,
        default="perception",
        choices=["perception", "sim_gt", "hybrid"],
        help=(
            "perception: 反思阶段用 YOLO+SAM2（EnhancedDetectionPipeline）生成 scene_json。"
            " sim_gt: 不加载检测模型，PickCube 用仿真 cube/goal 真值（调试用）。"
            " hybrid: 先跑 YOLO+SAM2；若检测为空或失败则回退 sim_gt（PickCube）。"
        ),
    )
    parser.add_argument(
        "--perception_visualization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="为 True 时 perception.perceive(..., generate_visualization=True)，在管线 output_dir 保存检测可视化。",
    )
    parser.add_argument(
        "--reset_before_retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After Phase A failure, call env.reset() before Phase D so the retry gets a full episode budget (PickCube max_episode_steps is often 50).",
    )
    parser.add_argument(
        "--verbose_reflect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print reflection stages (scene_json summary, VLM/brain outputs, retry) to stdout.",
    )
    args = parser.parse_args()

    root = _repo_root()
    ckpt_path = args.ckpt_path if args.ckpt_path.is_absolute() else (root / args.ckpt_path)
    video_path = args.video_path if args.video_path.is_absolute() else (root / args.video_path)
    log_path = args.reflect_log_path if args.reflect_log_path.is_absolute() else (root / args.reflect_log_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Route OpenAI SDK used by CognitiveBrain to local vLLM endpoint
    os.environ["OPENAI_BASE_URL"] = args.vlm_base_url.rstrip("/") + "/v1"
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "EMPTY"
    _api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    args.vlm_model = _resolve_vlm_model_id(args.vlm_base_url, _api_key, args.vlm_model)

    policy = _load_policy(ckpt_path, args.device)
    brain = CognitiveBrain(model=args.vlm_model, api_key=_api_key)
    perception = None
    perception_init_error = None
    if args.scene_source == "sim_gt":
        if args.verbose_reflect:
            print(
                "[VLA] scene_source=sim_gt：跳过 YOLO/SAM2，反思阶段使用仿真真值 cube/goal 构造 scene_json。",
                flush=True,
            )
    else:
        # perception / hybrid：加载 YOLO + SAM2（graduate_pro vision_ai）
        try:
            perception = ManiSkillPerceptionWrapper()
            if args.verbose_reflect:
                print("[VLA] YOLO+SAM2 感知管线已加载（EnhancedDetectionPipeline）。", flush=True)
        except Exception as e:
            perception_init_error = str(e)
            if args.scene_source == "hybrid" and args.verbose_reflect:
                print(
                    f"[VLA] hybrid：感知初始化失败，将仅用 sim_gt 回退：{perception_init_error}",
                    flush=True,
                )
            elif args.verbose_reflect:
                print(f"[VLA] 感知初始化失败（反思阶段 scene_json 可能为空）：{perception_init_error}", flush=True)
    runner = DPRunner(output_dir=None, n_obs_steps=3, n_action_steps=8)
    lang_enabled = _policy_supports_language(policy)

    # Initial intent embedding (System2 -> System1 bridge)
    current_phrase = args.instruction
    current_emb = encode_text_emb_for_film(current_phrase, device="cpu").detach().cpu().numpy()[0]

    env = gym.make(
        args.env_id,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        max_episode_steps=args.phase_steps,
    )
    raw_obs, info = env.reset(seed=args.seed)
    runner.reset_obs()
    runner.update_obs(_inject_language_in_obs(_obs_to_policy(raw_obs, env), current_emb, lang_enabled))

    frames: List[np.ndarray] = []
    total_steps = 0

    # Phase A: execute
    raw_obs, info, success, used_steps, ph_a_term, ph_a_trunc = _run_phase(
        env=env,
        policy=policy,
        runner=runner,
        start_obs=raw_obs,
        steps=args.phase_steps,
        language_emb=current_emb,
        lang_enabled=lang_enabled,
        render_mode=args.render_mode,
        frames=frames,
    )
    total_steps += used_steps

    reflection_record: Dict[str, Any] = {
        "phase_a_success": success,
        "phase_a_steps": used_steps,
        "phase_a_terminated": ph_a_term,
        "phase_a_truncated": ph_a_trunc,
        "lang_injection_enabled": lang_enabled,
    }
    if perception_init_error:
        reflection_record["perception_init_error"] = perception_init_error

    # Phase B/C/D: watchdog -> reflection -> re-conditioning -> retry
    if not success:
        if args.verbose_reflect:
            print("\n========== [VLA] Phase A 未成功，进入反思 / 重试 ==========", flush=True)
            print(
                f"  phase_a_steps={used_steps} terminated={ph_a_term} truncated={ph_a_trunc}",
                flush=True,
            )
            if ph_a_trunc:
                print(
                    "  提示: 若 truncated=True，说明本环境单回合步数已用尽；"
                    "默认会在 Phase D 前 reset，否则重试往往只能跑 1 步。",
                    flush=True,
                )

        rgb, depth = _rgb_depth_for_perception(raw_obs)
        waypoint = _waypoint_from_obs(raw_obs)
        use_viz = bool(args.perception_visualization)
        if args.scene_source == "sim_gt":
            scene_json = _scene_json_from_sim_pickcube(env)
            reflection_record["scene_source"] = "sim_gt"
        elif args.scene_source == "hybrid":
            scene_json = format_reference_library_for_vlm({})
            reflection_record["scene_source"] = "hybrid"
            if perception is not None:
                try:
                    p_out = perception.perceive(rgb, depth, waypoint, generate_visualization=use_viz)
                    ok = bool(p_out.get("success", False)) and bool(
                        getattr(perception.pipeline, "reference_library", None)
                    )
                    cand = perception._format_for_vlm() if ok else format_reference_library_for_vlm({})
                    n_cand = int(cand.get("count") or 0)
                    if n_cand > 0:
                        scene_json = cand
                        reflection_record["hybrid_branch"] = "perception"
                    else:
                        scene_json = _scene_json_from_sim_pickcube(env)
                        reflection_record["hybrid_branch"] = "sim_gt_fallback_empty"
                except Exception as e:
                    scene_json = _scene_json_from_sim_pickcube(env)
                    reflection_record["perception_error"] = str(e)
                    reflection_record["hybrid_branch"] = "sim_gt_fallback_error"
            else:
                scene_json = _scene_json_from_sim_pickcube(env)
                reflection_record["hybrid_branch"] = "sim_gt_fallback_no_pipeline"
        elif perception is not None:
            try:
                p_out = perception.perceive(rgb, depth, waypoint, generate_visualization=use_viz)
                ok = bool(p_out.get("success", False)) and bool(getattr(perception.pipeline, "reference_library", None))
                scene_json = perception._format_for_vlm() if ok else format_reference_library_for_vlm({})
            except Exception as e:
                scene_json = format_reference_library_for_vlm({})
                reflection_record["perception_error"] = str(e)
            reflection_record["scene_source"] = "perception"
        else:
            scene_json = format_reference_library_for_vlm({})
            reflection_record["scene_source"] = "none"

        scene_json, sp_removed = _filter_spurious_scene_objects(scene_json)
        if sp_removed:
            reflection_record["scene_spurious_removed"] = sorted(set(sp_removed))
            if args.verbose_reflect:
                print(
                    f"  已从 scene_json 剔除疑似 COCO 误检类别: {reflection_record['scene_spurious_removed']}",
                    flush=True,
                )

        nobj = int(scene_json.get("count") or 0)
        if args.verbose_reflect:
            print(f"  scene_json: count={nobj}", flush=True)
            for ob in (scene_json.get("objects") or [])[:8]:
                print(
                    f"    - id={ob.get('id')} class={ob.get('class_name')} centroid={ob.get('centroid_3d_m')}",
                    flush=True,
                )
            if nobj == 0:
                print(
                    "  提示: 空场景时 VLM 无法对齐真实 target_id；"
                    "若 count>0 但类别离谱（如 traffic light），多为 COCO YOLO 在仿真上的误检。",
                    flush=True,
                )

        # 1) Use existing CognitiveBrain reflection
        try:
            fallback_reflect = brain.reflect_on_failure(
                current_rgb=rgb,
                history_instruction=args.instruction,
                error_reason="I tried to pick up the green cube but failed.",
                scene_objects=scene_json,
            )
        except Exception as e:
            reflection_record["brain_reflect_error"] = str(e)
            fallback_reflect = {"action": "grasp", "target_id": "", "macro_action": "reset"}
            if args.verbose_reflect:
                print(f"  CognitiveBrain.reflect_on_failure 失败: {e}", flush=True)

        # 2) Try local vLLM multimodal reflection (image + scene), fallback to CognitiveBrain output
        local_reflect, vlm_err = _local_vlm_reflect_with_image(
            image_rgb=rgb,
            scene_objects=scene_json,
            model=args.vlm_model,
            base_url=args.vlm_base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )
        if vlm_err:
            reflection_record["local_vlm_error"] = vlm_err
        reflect = local_reflect or fallback_reflect
        reflection_record["scene_json"] = scene_json
        reflection_record["local_vlm_used"] = bool(local_reflect)

        if args.verbose_reflect:
            src = "local_vlm(multimodal)" if local_reflect else "CognitiveBrain(fallback)"
            print(f"  纠错决策来源: {src}", flush=True)
            if vlm_err and not local_reflect:
                print(f"  local_vlm 未调用成功: {vlm_err}", flush=True)
            print(f"  反思原始 JSON: {json.dumps(reflect, ensure_ascii=False)}", flush=True)

        reflect = _sanitize_reflect_for_policy(reflect, scene_json)
        reflection_record["brain_reflection"] = reflect
        if args.verbose_reflect:
            print(f"  规范化后 JSON: {json.dumps(reflect, ensure_ascii=False)}", flush=True)

        action = str(reflect.get("action", "grasp"))
        target_id = str(reflect.get("target_id", ""))
        class_name = class_for_id(scene_json, target_id)
        current_phrase = build_semantic_phrase_for_film(action, target_id, class_name)
        current_emb = encode_text_emb_for_film(current_phrase, device="cpu").detach().cpu().numpy()[0]
        reflection_record["new_phrase"] = current_phrase

        if args.reset_before_retry:
            raw_obs, info = env.reset(seed=args.seed)
            runner.reset_obs()
            reflection_record["phase_d_reset"] = True
            if args.verbose_reflect:
                print("  Phase D 前已 env.reset()，重试使用完整回合步数预算。", flush=True)
        else:
            reflection_record["phase_d_reset"] = False

        # Re-prime buffer with latest obs + new intent
        runner.update_obs(_inject_language_in_obs(_obs_to_policy(raw_obs, env), current_emb, lang_enabled))

        raw_obs, info, success2, used_steps2, ph_d_term, ph_d_trunc = _run_phase(
            env=env,
            policy=policy,
            runner=runner,
            start_obs=raw_obs,
            steps=args.phase_steps,
            language_emb=current_emb,
            lang_enabled=lang_enabled,
            render_mode=args.render_mode,
            frames=frames,
        )
        total_steps += used_steps2
        reflection_record["phase_d_steps"] = used_steps2
        reflection_record["phase_d_success"] = success2
        reflection_record["phase_d_terminated"] = ph_d_term
        reflection_record["phase_d_truncated"] = ph_d_trunc
        success = success2

        if args.verbose_reflect:
            print(
                f"========== [VLA] Phase D 结束 steps={used_steps2} success={success2} "
                f"terminated={ph_d_term} truncated={ph_d_trunc} ==========\n",
                flush=True,
            )

    env.close()

    if args.render_mode == "rgb_array" and len(frames) > 0:
        import imageio.v2 as imageio

        imageio.mimsave(str(video_path), frames, fps=20)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(reflection_record, ensure_ascii=False) + "\n")

    print("=== VLA Closed Loop Eval Finished ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"steps_total: {total_steps}")
    print(f"success: {success}")
    print(f"reflection_log: {log_path}")
    if args.render_mode == "rgb_array":
        print(f"video: {video_path}")
    if not reflection_record.get("lang_injection_enabled", False):
        print("note: checkpoint does not consume language_emb yet; embedding updates logged but not used by current policy.")


if __name__ == "__main__":
    main()
