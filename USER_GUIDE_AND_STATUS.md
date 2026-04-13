# 使用手册与当前进展

> **主文档**：日常开发与运行说明请以 **[使用手册.md](./使用手册.md)** 为准；**第三方来源**见 **[外部来源说明.md](./外部来源说明.md)**。本文档中关于 GUI、`test_end_to_end.py`、Marian/Qwen 等描述可能已过时（工程位于 **`MyRobotProject/`**，架构见 `perception_engine` / `cognitive_brain` / `motor_adapter` / `test_pipeline.py`）。

本文档汇总仓库根目录 **VLA 联调栈** 与 **具身语音中枢** 的用法，并附 **`test_end_to_end.py` 数据源分支**、**`embodied_command_hub.py` 指令解析后端分支** 的函数级说明。更通用的架构说明见 `README_VLA_SYS.md`。

---

## 一、当前进展（能力清单）

| 模块 | 状态 | 说明 |
|------|------|------|
| **感知中台** | 可用（依赖 `g1_env` 等） | `perception_wrapper.ManiSkillPerceptionWrapper` 调用 `graduate_pro` 内 `EnhancedDetectionPipeline`；失败时 `test_end_to_end` 可回退 `FALLBACK_REFERENCE_LIBRARY`。 |
| **场景 JSON → VLM** | 可用 | `vlm_brain.parse_instruction_with_vlm`：有 `OPENAI_API_KEY` 走 API，否则子串匹配 `class_name`。 |
| **FiLM 前置** | 可选 | CLIP 文本向量；`--skip-film` 或中枢「子进程跳过 FiLM」可省略。 |
| **语音中枢 GUI** | 可用 | Faster-Whisper、Marian zh→en、三种指令 JSON 后端、历史与 `logs/`、自动感知子进程、`hub_pointcloud_summary.json`、可滚动标注预览与保存。 |
| **语音 CLI** | 可用 | `python embodied_command_hub.py --cli`，空格键录音，指令后端与 GUI 同源逻辑（见下文）。 |
| **扩散策略 / 真机闭环** | 未在本仓库根目录闭环 | FiLM 向量打印为「可对接」状态；下游 DP/机械臂需另行集成。 |

---

## 二、环境与依赖（摘要）

- **完整感知 + 端到端**：见 `requirements-g1-env.txt` 与 `G1_ENV_SETUP.md`（若存在）。
- **中枢单独跑（不跑 YOLO）**：Whisper、faster-whisper、httpx、keyboard、sounddevice、transformers（翻译 + 可选 Qwen）、Pillow（预览）等；具体以你当前环境为准。
- **Windows**：`EMBODIED_WHISPER_FORCE_CPU=1` 可强制 Whisper CPU，规避 cuDNN 版本问题；`Library\bin` 会在启动时尝试加入 PATH（见 `embodied_command_hub.py` 顶部）。

---

## 三、快速使用

### 3.1 具身指令中枢（GUI，默认）

```text
cd /d "d:\personal project\code"
python embodied_command_hub.py
```

- **流程**：按住说话 → 转写可编辑 → **确认继续** → 按「识别语言」与 `should_run_marian_zh_to_en` 决定是否 Marian 译英 → 若勾选 **生成 JSON**，按所选 **指令 JSON 模型** 生成五字段机器人 JSON → 若勾选 **翻译后自动感知** 且已选 OCID 风格 RGB 图，后台启动 `test_end_to_end.py` 子进程。
- **感知 Python**：默认可填 conda 环境的 `python.exe`，用于跑子进程（建议与 `g1_env` 一致）。

### 3.2 命令行对讲模式

```text
python embodied_command_hub.py --cli --device cpu --asr-lang zh --instruction-backend qwen2.5
```

- `--instruction-backend`：`openai` | `rules` | `qwen2.5`（与 `LOCAL_INSTRUCTION_MODEL_IDS` 键一致）。
- `--disable-instruction-cli`：不生成指令 JSON。

### 3.3 端到端脚本（命令行）

```text
python test_end_to_end.py --ocid-rgb "D:/.../seq09/rgb/result_....png" --viz
python test_end_to_end.py --local-rgb "D:/some/photo.jpg" --instruction "grasp the apple" --viz
```

- 不写参数且无 `test_rgb.png` 时：使用随机 RGB/深度（便于冒烟测试）。
- 环境变量：`OCID_RGB`、`OPENAI_API_KEY`、`OPENAI_VLM_MODEL`、`USE_CUDA_CLIP`、`VLA_SKIP_DEP_WARN` 等（详见 `README_VLA_SYS.md`）。

---

## 四、`test_end_to_end.py`：数据源与主流程（函数级）

### 4.1 `main()` 入口：RGB/深度从哪来

决策顺序（与源码 `main()` 一致）：

1. **`--ocid-rgb` 或环境变量 `OCID_RGB`**  
   - 调用 **`load_ocid_bundle(Path(ocid_rgb))`**。  
   - 若同时传了 `--local-rgb`，打印警告并 **仍以 OCID 为准**。

2. **否则若 `--local-rgb` 非空**  
   - 若 **`_has_ocid_paired_depth_at_path(lp)` 为真**：视为 OCID 目录结构且存在 `../depth/<stem>.png`，调用 **`load_ocid_bundle(lp)`**（与直接传 rgb 路径等价，支持用户误选 depth/label 下文件时在 `load_ocid_bundle` 内纠正到 `rgb/`）。  
   - 否则：调用 **`load_local_rgb_bundle(lp)`**（**合成深度** uint16 mm）。

3. **否则**  
   - **`_load_or_build_rgb_depth()`**：优先读仓库根 `test_rgb.png` + `test_depth.npy`，缺失则随机图与合成深度。

**辅助函数：**

| 函数 | 作用 |
|------|------|
| **`load_ocid_bundle(rgb_path)`** | 解析 `seq_dir = rgb/..`；若路径在 `label/` 或 `depth/` 下则改读 `rgb/<同名>`；读 `depth/<stem>.png`（16-bit mm）；可选 `label/<stem>.png` 统计实例 ID；搜集 `pcd/` 下同 stem 文件；返回 `(image_rgb, depth_mm, waypoint_data, meta)`，`meta` 含 `ocid_frame` 所需字段。 |
| **`load_local_rgb_bundle(rgb_path)`** | 读任意图片；**高斯式随机深度** 与图同尺寸；`meta.dataset == "local_rgb"`。 |
| **`_has_ocid_paired_depth_at_path(path)`** | 父目录名为 `rgb`/`depth`/`label` 之一；`seq_dir/depth/<stem>.png` 存在；若在 `depth`/`label`，还要求 `seq_dir/rgb/<同名文件>` 存在。 |
| **`_load_or_build_rgb_depth()`** | 根目录测试图或 mock。 |
| **`_maybe_warn_missing_perception_deps()`** | 缺 `ultralytics`/`torch` 时打印一次警告（可用 `VLA_SKIP_DEP_WARN` 关闭）。 |

### 4.2 `main()`：感知 → VLM → 可视化 → FiLM

| 阶段 | 函数/行为 |
|------|-----------|
| 感知 | `ManiSkillPerceptionWrapper().perceive(...)`；成功则 `wrapper._format_for_vlm()` → **`scene_objects`**；否则 **`format_reference_library_for_vlm(FALLBACK_REFERENCE_LIBRARY)`**。 |
| 元数据写入场景 | 若 `bundle_meta` 存在：`local_rgb` → `scene_objects["local_frame"]`；否则 → **`scene_objects["ocid_frame"]`**（标签实例、pcd 路径等）。 |
| 认知 | **`vlm_brain.parse_instruction_with_vlm(instruction, scene_objects)`** → `decision`（`action`, `target_id`）。 |
| 高亮图 | **`_draw_user_target_on_full_rgb`**（内部用 **`_objects_for_user_target_view`**，依赖 `target_id` 或回退子串匹配）→ `detection_target_highlight.jpg`。 |
| 全检出图 | 管线返回的 `visualization_image` → `detection_all.jpg`。 |
| 点云摘要（中枢用） | **`_try_save_rgbd_scene_pointcloud`** / 失败或 `--no-scene-pcd` 时 **`_write_hub_pointcloud_summary`** 写 **`hub_pointcloud_summary.json`**。 |
| FiLM | 除非 **`--skip-film`**：`build_semantic_phrase_for_film` + **`encode_text_emb_for_film`**；`vlm_brain._class_for_id` 解析类别名。 |

---

## 五、`embodied_command_hub.py`：指令 JSON 后端（函数级）

### 5.1 UI 与内部键的对应关系

- **`_PARSER_UI_ROWS`**：`(内部键, 下拉显示名)`  
  - `openai` → 「OpenAI」  
  - `qwen2.5` → 「本地 Qwen2.5-3B」  
  - `rules` → 「规则（无大模型）」
- **`_parser_key()`**：由 **`parser_display_var`** 经 **`_parser_key_by_label`** 反查内部键；未知时默认 **`qwen2.5`**。

### 5.2 统一分发：`EmbodiedCommandHubApp._run_instruction_parser(en_text)`

| `backend` | 调用 | 成功返回值 | 常见失败 |
|-----------|------|------------|----------|
| **`openai`** | **`parse_instruction_with_openai`**（httpx → Chat Completions，`response_format=json_object`） | `(dict, None)` | 未配置 API Key → `(None, "未配置 OpenAI API Key")`；HTTP/JSON 异常在 `try/except` 中变为 `robot_err` |
| **`rules`** | **`parse_robot_instruction_rules`** | `(dict, None)` | 无（不联网） |
| **`qwen2.5`**（及 `LOCAL_INSTRUCTION_MODEL_IDS` 内其它键） | **`parse_instruction_local`** → **`shared.ensure_loaded`** + **`LocalInstructionParser.parse`** | `(dict, None)` | 未预加载且首次调用时 `ensure_loaded` 内加载；未加载就 `parse` 会 `RuntimeError` |

**相关函数：**

- **`parse_instruction_with_openai`**：系统提示 `INSTRUCTION_PARSER_SYSTEM_PROMPT`；结果 **`_normalize_robot_instruction`**（补全 `arm_preference`/`urgency` 等）。
- **`LocalInstructionParser.parse`**：chat template 生成；**`_extract_json_object_from_llm_text`** 从模型输出抽 JSON；再 **`_normalize_robot_instruction`**。
- **`parse_robot_instruction_rules`**：英文关键词定 **`action`**（grasp/place/push/pull/stop/explore）、**`spatial_constraint`**、**`arm_preference`**、**`urgency`**；**`target_object`** 由分词去掉停用词后取末段拼接（与 `vlm_brain` 的 `_fallback_parse` 不同，后者面向 `target_id`）。

### 5.3 GUI 中何时调用

- **`_on_confirm_transcript`** 工作线程内：若 **`gen_json_var`** 为真，则 **`_run_instruction_parser(en)`**；否则 `robot_obj` 保持 `None`。
- **`_set_busy` / `_on_parser_backend_changed`**：OpenAI 时启用 API Key 与模型下拉；本地 Qwen 时启用「预加载 Qwen」按钮。

### 5.4 Whisper 切换与本地 Qwen 显存

- **`_apply_settings`** → `commander.reconfigure` 成功后 **`_finish_settings_ok`** 会调用 **`_local_instruction_parser.unload()`**，避免 Whisper 改设备后旧 LLM 仍占 GPU；需再次点 **「预加载 Qwen」**。

### 5.5 CLI `run_loop` 与 GUI 的差异

- **`LocalSTTCommander.run_loop(..., instruction_backend=...)`** 分支逻辑与上表一致：`openai` / `rules` / `LOCAL_INSTRUCTION_MODEL_KEYS`；`instruction_backend is None` 或 `none|off|skip` 时不生成 JSON。
- OpenAI 分支使用环境变量 **`OPENAI_API_KEY`** 与参数 **`openai_model`**（**不**经过 GUI 的 Entry）。

### 5.6 自动感知子进程

- **`_maybe_auto_run_perception_viz(en)`**（在 **`_finish_ok`** 末尾）：条件为勾选自动感知、已选有效图片路径、英文 `en` 非空。  
- 命令：`<VLA_PYTHON 或 sys.executable> test_end_to_end.py --local-rgb ... --instruction <en> --viz --viz-out <临时目录> --pcd-stride 4`，可选 **`--skip-film`**（`hub_skip_film_var`）。  
- 完成后 **`_finish_auto_perception_viz`** 读日志、`hub_pointcloud_summary.json`、**`detection_target_highlight.jpg`**。

---

## 六、日志与产物路径（参考）

| 路径 | 内容 |
|------|------|
| `logs/history.txt`、`logs/history.jsonl` | 中枢会话历史 |
| `logs/robot_instructions.jsonl` | 结构化指令记录（含 `instruction_backend`） |
| `vla_viz_output/<时间戳>/` | 直接运行 `test_end_to_end.py --viz` 且未指定 `--viz-out` 时 |
| 临时 `embodied_hub_viz_*` | 中枢自动感知；关闭或清除时可删除 |

---

## 七、修订记录

- 文档随仓库当前代码整理；若行为与源码不一致，以 **`embodied_command_hub.py`**、**`test_end_to_end.py`** 为准。
