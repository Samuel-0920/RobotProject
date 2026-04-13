"""
运动适配层：将 VLM 决策（action / target_id / class）编码为 DP 小脑可用的 CLIP 文本嵌入 [1, 512]。

职责边界：仅此模块在桥接路径上导入 torch + LanguageFiLM。

上游：`LanguageFiLM` 与路径 `ADM_DP/policy/Diffusion-Policy` 来自本目录下 ADM_DP 工程；CLIP 为 OpenAI 开源实现（见 外部来源说明.md）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_REPO_ROOT = Path(__file__).resolve().parent


def build_semantic_phrase_for_film(action: str, target_id: str, class_name: str = "") -> str:
    """供 LanguageFiLM / CLIP 使用的短句。"""
    parts = [f"Action: {action.strip()}."]
    tid = (target_id or "").strip()
    if tid:
        parts.append(f"Target id: {tid}.")
    else:
        parts.append("Target id: unknown.")
    if class_name:
        parts.append(f"Object class: {class_name.strip()}.")
    return " ".join(parts)


def encode_text_emb_for_film(
    phrase: str,
    clip_model_name: str = "ViT-B/32",
    device: str = "cpu",
) -> "torch.Tensor":
    """
    调用 ADM_DP 中 LanguageFiLM.encode_texts，返回形状 (1, 512) 的 float32 张量。
    需要: pip install git+https://github.com/openai/CLIP.git 及 torch
    """
    _dp_root = _REPO_ROOT / "ADM_DP" / "policy" / "Diffusion-Policy"
    if _dp_root.is_dir() and str(_dp_root) not in sys.path:
        sys.path.insert(0, str(_dp_root))

    import torch

    from diffusion_policy.model.vision.language_film import LanguageFiLM  # noqa: E402

    emb = LanguageFiLM.encode_texts([phrase], clip_model_name=clip_model_name, device=device)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    return emb.float()
