# -*- coding: utf-8 -*-
"""
Headless CLI：语音 → Faster-Whisper（cognitive_brain.LocalSTTCommander）→ 指令 JSON（OpenAI 或规则）。

已移除：Tkinter GUI、Marian/Helsinki 翻译、本地 Qwen 指令预加载。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import httpx

from cognitive_brain import LocalSTTCommander


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

ROBOT_INSTRUCTION_FIELDS = (
    "action",
    "target_object",
    "spatial_constraint",
    "arm_preference",
    "urgency",
)

INSTRUCTION_PARSER_SYSTEM_PROMPT = '''You are the "Instruction Parser" for an advanced dual-arm embodied AI robot.
Your ONLY job is to translate natural language user commands into a structured JSON format.
You must extract the core intent, the target object, any spatial constraints, and arm preferences.

Strict Rules:
1. Output ONLY valid JSON. No conversational text, no markdown blocks, no explanations.
2. If a field is not explicitly mentioned or cannot be inferred, output null for that field.

Output JSON Schema Requirements:
- "action": The primary physical action (e.g., "grasp", "place", "push", "pull", "explore", "stop").
- "target_object": The specific object the robot needs to interact with (e.g., "red_apple", "cup"). Use underscores for spaces.
- "spatial_constraint": Any spatial relationship mentioned (e.g., "behind_the_cup", "on_top_of_box").
- "arm_preference": If the user specifies which hand to use ("left", "right", "bimanual"). Default to "auto" if not specified.
- "urgency": "high" if words like 'quick', 'fast', 'immediately' are used, otherwise "normal".

Your JSON object MUST contain exactly these keys: action, target_object, spatial_constraint, arm_preference, urgency.
Additionally: All JSON string values MUST be in English (use English action verbs; use underscores for multi-word identifiers like "red_apple").'''


def _chat_completions_url() -> str:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    return f"{base}/chat/completions"


def _normalize_robot_instruction(obj: dict) -> dict:
    out = {}
    for k in ROBOT_INSTRUCTION_FIELDS:
        v = obj.get(k, None)
        if isinstance(v, str) and v.strip() == "":
            v = None
        out[k] = v
    ap = out.get("arm_preference")
    if ap is None:
        out["arm_preference"] = "auto"
    if out.get("urgency") is None:
        out["urgency"] = "normal"
    return out


def parse_instruction_with_openai(
    english_command: str,
    *,
    api_key: str,
    model: str = "gpt-4o",
    timeout_s: float = 120.0,
) -> dict:
    english_command = (english_command or "").strip()
    if not english_command:
        return {k: None for k in ROBOT_INSTRUCTION_FIELDS} | {"arm_preference": "auto", "urgency": "normal"}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": INSTRUCTION_PARSER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Translate the following user command into the JSON object.\n\n" + english_command,
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(_chat_completions_url(), json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        snippet = ""
        try:
            snippet = (e.response.text or "")[:2000]
        except Exception:
            snippet = ""
        raise RuntimeError(f"OpenAI API HTTP {e.response.status_code}: {snippet or e}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"OpenAI API 网络请求失败：{e}") from e

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"OpenAI 响应结构异常：{data}") from e

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"模型返回的不是合法 JSON：{content}") from e

    if not isinstance(parsed, dict):
        raise RuntimeError(f"模型 JSON 根类型必须是 object，实际为：{type(parsed).__name__}")

    return _normalize_robot_instruction(parsed)


def parse_robot_instruction_rules(english_command: str) -> dict:
    en = (english_command or "").strip()
    tl = en.lower()
    action = "grasp"
    if any(x in tl for x in ("place", "put down", "lay down", "set down")):
        action = "place"
    elif "push" in tl:
        action = "push"
    elif "pull" in tl:
        action = "pull"
    elif "stop" in tl or "halt" in tl:
        action = "stop"
    elif "explore" in tl:
        action = "explore"

    spatial = None
    if "behind" in tl:
        spatial = "behind_object"
    elif "on the table" in tl or "on table" in tl:
        spatial = "on_table"
    elif "on top" in tl:
        spatial = "on_top"

    arm = "auto"
    if "left hand" in tl or "left arm" in tl:
        arm = "left"
    elif "right hand" in tl or "right arm" in tl:
        arm = "right"
    elif "both hands" in tl or "bimanual" in tl:
        arm = "bimanual"

    urgency = "high" if any(x in tl for x in ("quick", "fast", "immediately", "urgent", "hurry")) else "normal"

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", en)
    skip = {
        "the",
        "a",
        "an",
        "please",
        "grasp",
        "pick",
        "place",
        "put",
        "take",
        "get",
        "give",
        "to",
        "on",
        "in",
        "at",
        "and",
        "me",
        "my",
        "down",
        "up",
        "table",
    }
    meaningful = [w for w in words if w.lower() not in skip]
    target = "_".join(meaningful[-5:]).lower() if meaningful else "object"

    raw = {
        "action": action,
        "target_object": target or "object",
        "spatial_constraint": spatial,
        "arm_preference": arm,
        "urgency": urgency,
    }
    return _normalize_robot_instruction(raw)


def run_cli_voice_loop(
    commander: LocalSTTCommander,
    *,
    openai_model: str | None = "gpt-4o",
    asr_lang: str = "auto",
    instruction_backend: str | None = "openai",
) -> None:
    asr_whisper = None if (asr_lang or "auto").lower() == "auto" else asr_lang
    while True:
        try:
            audio_path = commander.record_audio_push_to_talk()
            transcript, detected = commander.transcribe(audio_path, language=asr_whisper)
            print(f"ASR ({detected or asr_lang}): {transcript}")

            if instruction_backend is None:
                backend = None
            else:
                backend = (instruction_backend or "openai").strip().lower()
                if backend in ("none", "off", "skip", ""):
                    backend = None

            if backend == "openai":
                api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
                om = (openai_model or "").strip()
                if om and api_key and transcript.strip():
                    try:
                        rob = parse_instruction_with_openai(transcript, api_key=api_key, model=om)
                        print(f"ROBOT JSON: {json.dumps(rob, ensure_ascii=False)}")
                    except Exception as e:
                        print(f"Instruction JSON (OpenAI) failed: {e}")
            elif backend == "rules":
                if transcript.strip():
                    rob = parse_robot_instruction_rules(transcript)
                    print(f"ROBOT JSON (rules): {json.dumps(rob, ensure_ascii=False)}")
        except KeyboardInterrupt:
            print("\nExiting voice CLI.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Embodied voice CLI (headless)")
    parser.add_argument("--cli", action="store_true", help="Push-to-talk loop (space key)")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("auto", "cuda", "cpu"),
        help="Whisper device; auto tries CUDA then CPU",
    )
    parser.add_argument("--model-size", default="small", help="Faster-Whisper model size")
    parser.add_argument(
        "--asr-lang",
        default="auto",
        choices=("auto", "zh", "en"),
        help="ASR language (Whisper); transcript is passed through without translation",
    )
    parser.add_argument(
        "--instruction-backend",
        default="openai",
        choices=("openai", "rules"),
        help="Robot instruction JSON backend",
    )
    parser.add_argument("--openai-model", default="gpt-4o", help="Chat Completions model for openai backend")
    parser.add_argument(
        "--disable-instruction-cli",
        action="store_true",
        help="Do not run instruction JSON after ASR",
    )
    args = parser.parse_args()

    dev = (args.device or "auto").strip().lower()
    force_cpu_env = os.environ.get("EMBODIED_WHISPER_FORCE_CPU", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if force_cpu_env:
        whisper_device = "cpu"
        cuda_fallback = True
    elif dev == "auto":
        whisper_device = "cuda"
        cuda_fallback = True
    elif dev == "cpu":
        whisper_device = "cpu"
        cuda_fallback = True
    else:
        whisper_device = "cuda"
        strict_cuda = os.environ.get("EMBODIED_WHISPER_STRICT_CUDA", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        cuda_fallback = not strict_cuda

    commander = LocalSTTCommander(
        model_size=args.model_size,
        device=whisper_device,
        cuda_fallback=cuda_fallback,
    )

    if args.cli:
        run_cli_voice_loop(
            commander,
            openai_model=(None if args.disable_instruction_cli else args.openai_model),
            asr_lang=args.asr_lang,
            instruction_backend=(None if args.disable_instruction_cli else args.instruction_backend),
        )
    else:
        parser.print_help()
        print("\nUse --cli for push-to-talk voice loop.")


if __name__ == "__main__":
    main()
