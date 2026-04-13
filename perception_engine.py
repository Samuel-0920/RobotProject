"""
感知引擎：RGB-D → EnhancedDetectionPipeline → 场景 JSON（与裁剪/可视化元数据）。

职责边界：仅视觉与 3D 锚定；不导入 torch 计算图（检测栈在 vision_ai 侧自行加载）。
可选环境变量 VISION_CONFIG：指向自定义 enhanced_detection_config.json 绝对路径。

上游：`EnhancedDetectionPipeline` 与 `vision_ai` 来自本目录下 `graduate_pro/`（见 外部来源说明.md）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
_VISION_PKG_ROOT = _REPO_ROOT / "graduate_pro" / "src" / "vision_ai"
if _VISION_PKG_ROOT.is_dir() and str(_VISION_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_VISION_PKG_ROOT))

_DEFAULT_CONFIG = (
    _REPO_ROOT
    / "graduate_pro"
    / "src"
    / "vision_ai"
    / "vision_ai"
    / "detection"
    / "config"
    / "enhanced_detection_config.json"
)


def format_reference_library_for_vlm(reference_library: Dict[str, Any]) -> Dict[str, Any]:
    """纯函数：给定 reference_library，生成 VLM 用场景描述（无需实例化检测模型）。"""
    objects: List[Dict[str, Any]] = []

    for dict_key, entry in (reference_library or {}).items():
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata") or {}
        feats = entry.get("features") or {}
        object_id = meta.get("object_id", dict_key)
        class_name = meta.get("class_name", "unknown")
        centroid = ManiSkillPerceptionWrapper._spatial_centroid_meters(feats)

        ob: Dict[str, Any] = {
            "id": str(object_id),
            "class_name": str(class_name),
            "centroid_3d_m": centroid,
        }
        bb = meta.get("bounding_box")
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            try:
                ob["bbox_xyxy"] = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
            except (TypeError, ValueError):
                pass
        objects.append(ob)

    return {"objects": objects, "count": len(objects)}


class ManiSkillPerceptionWrapper:
    """
    封装学长的 EnhancedDetectionPipeline：build_reference_library + _format_for_vlm。
    """

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        from vision_ai.detection.enhanced_detection_pipeline import (  # noqa: WPS433
            EnhancedDetectionPipeline,
        )

        if config_file is not None:
            cfg = Path(config_file)
        else:
            env_cfg = os.environ.get("VISION_CONFIG", "").strip()
            cfg = Path(env_cfg) if env_cfg else _DEFAULT_CONFIG
        if not cfg.is_file():
            raise FileNotFoundError(f"Detection config not found: {cfg}")
        self.pipeline = EnhancedDetectionPipeline(
            config_file=str(cfg),
            output_dir=str(output_dir) if output_dir else None,
        )

    def perceive(
        self,
        image_rgb: np.ndarray,
        depth_image: Optional[np.ndarray],
        waypoint_data: Dict[str, Any],
        generate_visualization: bool = False,
    ) -> Dict[str, Any]:
        """
        运行一次完整感知；成功后结果在 self.pipeline.reference_library。

        Args:
            image_rgb: (H, W, 3) RGB，uint8 或 float
            depth_image: (H, W) 深度，毫米 uint16（内部 /1000 → 米）
            waypoint_data: 需含 world_pos, roll, pitch, yaw
        """
        return self.pipeline.build_reference_library(
            image_rgb=image_rgb,
            depth_image=depth_image,
            waypoint_data=waypoint_data,
            generate_visualization=generate_visualization,
        )

    @staticmethod
    def _spatial_centroid_meters(features: Dict[str, Any]) -> Optional[List[float]]:
        spatial = features.get("spatial") or {}
        if not isinstance(spatial, dict):
            return None
        raw: Optional[tuple] = None
        for key in ("world_centroid", "world_coordinates"):
            v = spatial.get(key)
            if v is None:
                continue
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                raw = (float(v[0]), float(v[1]), float(v[2]))
                break
        if raw is None:
            cam = spatial.get("centroid_3d_camera")
            if isinstance(cam, (list, tuple)) and len(cam) >= 3:
                return [float(cam[0]), float(cam[1]), float(cam[2])]
            return None
        x, y, z = raw
        if max(abs(x), abs(y), abs(z)) > 25.0:
            x, y, z = x / 1000.0, y / 1000.0, z / 1000.0
        return [x, y, z]

    def _format_for_vlm(self) -> Dict[str, Any]:
        lib = getattr(self.pipeline, "reference_library", None) or {}
        return format_reference_library_for_vlm(lib)


if __name__ == "__main__":
    mock_lib = {
        "apple_0": {
            "metadata": {"object_id": "apple_0", "class_name": "apple"},
            "features": {"spatial": {"world_coordinates": (0.1, -0.2, 0.45)}},
        }
    }
    print(format_reference_library_for_vlm(mock_lib))
