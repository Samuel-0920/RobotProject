#!/usr/bin/env python3
"""极简端到端：Mock RGB-D → 感知 JSON → 认知决策 → Motor 张量形状。"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MOCK_H, MOCK_W = 48, 64

MOCK_REFERENCE_LIBRARY = {
    "apple_0": {
        "metadata": {
            "object_id": "apple_0",
            "class_name": "apple",
            "bounding_box": [10, 10, 30, 30],
        },
        "features": {"spatial": {"world_coordinates": (350.0, -120.0, 420.0)}},
    },
}


def main() -> int:
    image_rgb = np.zeros((MOCK_H, MOCK_W, 3), dtype=np.uint8)
    depth_mm = np.full((MOCK_H, MOCK_W), 500, dtype=np.uint16)
    waypoint_data = {
        "world_pos": [320.0, 0.0, 350.0],
        "roll": 179.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }

    scene_json: dict
    _noise = io.StringIO()
    with contextlib.redirect_stdout(_noise), contextlib.redirect_stderr(_noise):
        from perception_engine import ManiSkillPerceptionWrapper, format_reference_library_for_vlm

        try:
            wrapper = ManiSkillPerceptionWrapper()
            result = wrapper.perceive(
                image_rgb=image_rgb,
                depth_image=depth_mm,
                waypoint_data=waypoint_data,
                generate_visualization=False,
            )
            ok = bool(result.get("success", False)) and bool(
                getattr(wrapper.pipeline, "reference_library", None)
            )
            scene_json = wrapper._format_for_vlm() if ok else format_reference_library_for_vlm(MOCK_REFERENCE_LIBRARY)
        except Exception:
            scene_json = format_reference_library_for_vlm(MOCK_REFERENCE_LIBRARY)

    print("[Perception] JSON ->", json.dumps(scene_json, ensure_ascii=False))

    from cognitive_brain import CognitiveBrain, class_for_id

    brain = CognitiveBrain()
    instruction = os.environ.get("VLA_TEST_INSTRUCTION", "grasp the apple")
    decision = brain.decide(instruction, scene_json)
    print("[Brain] Decision ->", json.dumps(decision, ensure_ascii=False))

    from motor_adapter import build_semantic_phrase_for_film, encode_text_emb_for_film

    phrase = build_semantic_phrase_for_film(
        decision.get("action", "grasp"),
        decision.get("target_id", ""),
        class_for_id(scene_json, decision.get("target_id", "")),
    )
    device = "cuda" if os.environ.get("USE_CUDA_CLIP", "").lower() in ("1", "true", "yes") else "cpu"
    try:
        text_emb = encode_text_emb_for_film(phrase, device=device)
        motor_line = str(tuple(text_emb.shape))
    except Exception:
        motor_line = "unavailable"
    print("[Motor] Tensor Shape ->", motor_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
