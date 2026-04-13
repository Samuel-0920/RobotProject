"""
认知中枢：语音听写（Faster-Whisper）+ VLM 指令解析 + 失败反思。

职责边界：不处理 numpy 图像切片；视觉输入仅通过场景 JSON（及可选的 future 多模态 API）进入 VLM。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import threading
from typing import Any, Dict, Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write


def _prepend_current_env_library_bin_to_path() -> None:
    try:
        base = os.path.dirname(os.path.abspath(sys.executable))
        lib_bin = os.path.join(base, "Library", "bin")
        if os.path.isdir(lib_bin):
            p = os.environ.get("PATH", "")
            parts = p.split(os.pathsep) if p else []
            if lib_bin not in parts:
                os.environ["PATH"] = lib_bin + os.pathsep + p
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(os.path.abspath(lib_bin))
                except OSError:
                    pass
    except Exception:
        pass


_prepend_current_env_library_bin_to_path()

DEFAULT_PARSE_PROMPT = """Task: {user_instruction}
Scene Objects (JSON): {scene_json}

Identify the target object ID and the action.
Output a single JSON object only, no markdown, with exactly these keys: "action" (string) and "target_id" (string).
Example: {{"action": "grasp", "target_id": "apple_0"}}
If no object matches the requested type, use an empty string for target_id. Do not pick a random object."""

REFLECT_ON_FAILURE_PROMPT = """You are the embodied agent planner. Recovery mode.

Original task: {history_instruction}
The system raised an error: {error_reason}

Latest scene objects (JSON):
{scene_json}

Propose a remedy: either pick a new target_id from the scene, or issue a macro_action (e.g. "reset", "relocalize", "abort") if recovery is not possible with the visible objects.

Output a single JSON object only, no markdown, with exactly these keys:
- "action" (string): suggested low-level action if applicable, e.g. "grasp"
- "target_id" (string): object id from the scene, or empty if using macro only
- "macro_action" (string): e.g. "reset", "relocalize", "abort", or empty string if a normal target_id suffices
"""


def _fallback_parse(user_instruction: str, scene: Dict[str, Any]) -> Dict[str, str]:
    objects = scene.get("objects") or []
    text = user_instruction.lower()
    action = "grasp"
    if "place" in text or "put" in text:
        action = "place"
    elif "pick" in text or "抓" in user_instruction:
        action = "grasp"

    for obj in objects:
        cn = str(obj.get("class_name", "")).lower()
        if cn and cn in text:
            return {"action": action, "target_id": str(obj.get("id", ""))}
    return {"action": action, "target_id": ""}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _extract_reflect_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_instruction_with_vlm(
    user_instruction: str,
    scene_objects: Dict[str, Any],
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """调用 OpenAI API 解析指令 + 场景；返回 {"action", "target_id"}。无 Key 时走 _fallback_parse。"""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    scene_str = json.dumps(scene_objects, ensure_ascii=False)
    prompt = DEFAULT_PARSE_PROMPT.format(
        user_instruction=user_instruction,
        scene_json=scene_str,
    )

    if not api_key:
        return _fallback_parse(user_instruction, scene_objects)

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install openai: pip install openai") from e

    client = OpenAI(api_key=api_key)
    model = model or os.environ.get("OPENAI_VLM_MODEL", "gpt-4o-mini")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You only reply with valid JSON objects."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""
    data = _extract_json_object(raw)
    if not data:
        return _fallback_parse(user_instruction, scene_objects)
    action = str(data.get("action", "grasp"))
    target_id = str(data.get("target_id", ""))
    return {"action": action, "target_id": target_id}


def class_for_id(scene: Dict[str, Any], target_id: str) -> str:
    for o in scene.get("objects") or []:
        if str(o.get("id")) == str(target_id):
            return str(o.get("class_name", ""))
    return ""


class CognitiveBrain:
    """VLM 决策与失败反思；可选持有默认模型名与 API Key。"""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key

    def decide(
        self,
        user_instruction: str,
        scene_objects: Dict[str, Any],
    ) -> Dict[str, str]:
        return parse_instruction_with_vlm(
            user_instruction,
            scene_objects,
            model=self.model,
            api_key=self.api_key,
        )

    def reflect_on_failure(
        self,
        current_rgb: Any,
        history_instruction: str,
        error_reason: str,
        *,
        scene_objects: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        结合最新场景 JSON 输出补救决策。current_rgb 预留多模态扩展，当前实现不读取像素。

        返回键：action, target_id, macro_action（均为字符串）。
        """
        _ = current_rgb
        return self._reflect_sync(scene_objects, history_instruction, error_reason)

    async def areflect_on_failure(
        self,
        current_rgb: Any,
        history_instruction: str,
        error_reason: str,
        *,
        scene_objects: Dict[str, Any],
    ) -> Dict[str, str]:
        _ = current_rgb
        return await asyncio.to_thread(
            self._reflect_sync,
            scene_objects,
            history_instruction,
            error_reason,
        )

    def _reflect_sync(
        self,
        scene_objects: Dict[str, Any],
        history_instruction: str,
        error_reason: str,
    ) -> Dict[str, str]:
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        scene_str = json.dumps(scene_objects, ensure_ascii=False)
        prompt = REFLECT_ON_FAILURE_PROMPT.format(
            history_instruction=history_instruction,
            error_reason=error_reason,
            scene_json=scene_str,
        )
        default_out = {"action": "grasp", "target_id": "", "macro_action": "reset"}

        if not api_key:
            return default_out

        try:
            from openai import OpenAI
        except ImportError:
            return default_out

        client = OpenAI(api_key=api_key)
        model = self.model or os.environ.get("OPENAI_VLM_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You only reply with valid JSON objects."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or ""
        data = _extract_reflect_json(raw)
        if not data or not isinstance(data, dict):
            return default_out
        return {
            "action": str(data.get("action", "grasp")),
            "target_id": str(data.get("target_id", "")),
            "macro_action": str(data.get("macro_action", "")),
        }


class LocalSTTCommander:
    """本地 Faster-Whisper：录音路径 → 文本。无 Marian / 无翻译。"""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: Optional[str] = None,
        *,
        cuda_fallback: bool = True,
    ):
        self._model_lock = threading.Lock()
        self._model_size: Optional[str] = None
        self._device: Optional[str] = None
        self._compute_type: Optional[str] = None
        self.model: Optional[WhisperModel] = None
        self.fs = 16000
        self._allow_cuda_fallback = bool(cuda_fallback)
        self.reconfigure(model_size=model_size, device=device, compute_type=compute_type)

    @property
    def model_size(self) -> str:
        return self._model_size or "small"

    @property
    def device(self) -> str:
        return self._device or "cpu"

    @staticmethod
    def _warmup_whisper_cuda(model: WhisperModel) -> None:
        silent = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        silent.close()
        path = silent.name
        try:
            z = np.zeros(int(0.25 * 16000), dtype=np.int16)
            write(path, 16000, z)
            segs, _ = model.transcribe(path, beam_size=1)
            list(segs)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def reconfigure(
        self,
        model_size: str,
        device: str,
        compute_type: Optional[str] = None,
    ) -> None:
        model_size = (model_size or "small").strip()
        device = (device or "cuda").strip()
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        try:
            new_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            if device == "cuda":
                self._warmup_whisper_cuda(new_model)
        except Exception as e:
            if device == "cuda" and self._allow_cuda_fallback:
                device = "cpu"
                compute_type = "int8"
                new_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            else:
                raise
        with self._model_lock:
            self.model = new_model
            self._model_size = model_size
            self._device = device
            self._compute_type = compute_type

    def _record_until(self, wait_before_start, should_continue_recording):
        wait_before_start()
        recording = []

        def callback(indata, frames, time, status):
            if status:
                print(status)
            recording.append(indata.copy())

        with sd.InputStream(samplerate=self.fs, channels=1, callback=callback):
            while should_continue_recording():
                sd.sleep(10)

        if not recording:
            raise ValueError("Recording too short.")
        audio_data = np.concatenate(recording, axis=0)
        temp_wav = tempfile.mktemp(suffix=".wav")
        write(temp_wav, self.fs, audio_data)
        return temp_wav

    def record_audio_push_to_talk(self, trigger_key: str = "space") -> str:
        import keyboard

        def wait_start():
            keyboard.wait(trigger_key)

        path = self._record_until(wait_start, lambda: keyboard.is_pressed(trigger_key))
        return path

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> tuple[str, str]:
        with self._model_lock:
            model = self.model
        if model is None:
            raise RuntimeError("Whisper model not loaded")
        whisper_lang = None if (language in (None, "", "auto")) else language
        segments, info = model.transcribe(audio_path, beam_size=5, language=whisper_lang)
        text_result = "".join(segment.text for segment in segments)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        detected = getattr(info, "language", None) or whisper_lang or ""
        return text_result.strip(), (detected or "").strip()


# 兼容旧 vlm_brain 下划线命名
_class_for_id = class_for_id
