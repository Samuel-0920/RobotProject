# VLA 感知–认知联调系统：部署与操作手册

本文档说明仓库根目录下 **`perception_wrapper.py`**、**`vlm_brain.py`** 与 **`test_end_to_end.py`** 的职责、依赖配置与端到端验证方法。

---

## 1. 架构概述

本联调栈对应分层 **VLA（Vision–Language–Action）** 流水线中的前两级，并与下游 **Diffusion Policy（小脑）** 的 **Language FiLM** 文本条件对接预留接口。

| 模块 | 文件 | 角色 |
|------|------|------|
| **System 1 — 感知中台** | `perception_wrapper.py` | 封装学长 **`EnhancedDetectionPipeline.build_reference_library`**：YOLO 检测 → SAM2 分割 → 3D 去重与特征；将 **`reference_library`** 转为 **`_format_for_vlm()`** 的精简 JSON（物体 **id**、**class_name**、**centroid_3d_m**）。体现 **身份追踪与实例级 3D 锚定**。 |
| **System 2 — 认知大脑** | `vlm_brain.py` | 接收自然语言指令与场景 JSON，输出 **`{"action", "target_id"}`**；可选 **OpenAI API** 做语义解析，无 Key 时 **回退为本地字符串匹配**，便于无网/无账号联调。 |
| **FiLM 前置** | `perception_wrapper.encode_text_emb_for_film` | 将动作与目标拼成短句，经 **CLIP** 编码为 **`[1, 512]`** 向量，供 **ADM_DP `LanguageFiLM`** 或与扩散策略条件模块对接。 |
| **端到端测试** | `test_end_to_end.py` | 串联：加载/合成 RGB-D → `perceive` → `_format_for_vlm` → `parse_instruction_with_vlm` → `encode_text_emb_for_film`，并带异常保护。 |

**数据流（概念）：**

```text
RGB-D + waypoint  →  ManiSkillPerceptionWrapper.perceive()
                  →  pipeline.reference_library
                  →  _format_for_vlm() → scene_objects JSON
       User text  →  parse_instruction_with_vlm() → {action, target_id}
                  →  build_semantic_phrase_for_film()
                  →  encode_text_emb_for_film() → Tensor [1,512] → DP / FiLM
```

---

## 2. 依赖配置

### 2.1 Python 环境

建议使用 **Python 3.9+**，并为 **`graduate_pro`** 感知栈单独建虚拟环境，避免与 ManiSkill / 其他项目冲突。

### 2.2 第三方库（按层级）

| 用途 | 包名 | 说明 |
|------|------|------|
| 感知（完整） | `ultralytics` | YOLO 推理 |
| 感知（完整） | `sam2`（Segment Anything 2） | 与 `vision_ai` 中 `SAM2Segmentor` 一致 |
| 感知（完整） | `opencv-python` | 图像读写与可视化 |
| 感知（3D 特征） | `open3d` | FPFH 等（可选但推荐） |
| 感知（几何） | `numpy`, `scipy`, `scikit-learn` | 点云/邻域等 |
| 认知（API） | `openai` ≥ 1.x | Chat Completions 解析指令 |
| FiLM 前置 | `torch`, `torchvision` | CLIP 推理 |
| FiLM 前置 | `clip`（OpenAI CLIP） | `pip install git+https://github.com/openai/CLIP.git` |
| 扩散侧（后续） | 见 `ADM_DP/requirements.txt` | `diffusers`, `hydra-core` 等 |

完整跑通 **真实感知** 还需：学长工程 **`graduate_pro`** 下 **`enhanced_detection_config.json`** 指向的 **YOLO 权重**、**SAM2 权重** 路径有效（见该配置文件）。

### 2.3 硬件资源

- **GPU（CUDA）**：YOLO + SAM2 + CLIP 同时运行时强烈推荐；仅联调 **VLM + CLIP** 可在 CPU 上运行（较慢）。
- **显存**：依所选 YOLO/SAM2 规模而定；SAM2 Hiera Large 建议预留充足显存。
- 环境变量 **`USE_CUDA_CLIP=1`** 时，端到端脚本中 CLIP 编码会使用 **`device="cuda"`**（需本机 CUDA 可用）。

---

## 3. 环境变量设置

| 变量 | 作用 |
|------|------|
| **`OPENAI_API_KEY`** | 设置后，`vlm_brain.parse_instruction_with_vlm` 调用 **OpenAI API** 解析指令与场景 JSON。 |
| **`OPENAI_VLM_MODEL`** | 可选；默认 **`gpt-4o-mini`**。 |
| **`USE_CUDA_CLIP`** | 设为 **`1` / `true` / `yes`** 时，`test_end_to_end.py` 中 CLIP 使用 GPU。 |

**无 `OPENAI_API_KEY` 时**：系统自动使用 **`_fallback_parse`**（基于指令中出现的 **`class_name` 子串**等简单规则），不发起网络请求，适合本地 CI 与离线演示。

---

## 4. 运行端到端测试

### 4.1 命令

在仓库根目录（与 `test_end_to_end.py` 同级）执行：

```bash
cd /path/to/personal/project/code
python test_end_to_end.py
```

（Windows PowerShell 同理，将路径换为 `d:\personal project\code`。）

### 4.2 使用 OCID 数据集一帧（rgb + depth + label + pcd）

若已下载 [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/)（如 **ARID10**），任选一帧 **RGB** 路径，脚本会自动解析同 **stem** 的：

- `../depth/<stem>.png` — 16-bit 深度（毫米）
- `../label/<stem>.png` — 16-bit 实例标签
- `../pcd/<stem>.pcd` — 点云路径（会在摘要中打印，供 Open3D / CloudCompare 使用）

示例（按你的实际路径调整）：

```bash
python test_end_to_end.py --ocid-rgb "D:/personal project/code/OCID-dataset/OCID-dataset/ARID10/table/top/fruits/seq09/rgb/result_2018-08-23-11-18-31.png"
```

或使用环境变量：

```bash
set OCID_RGB=D:\...\seq09\rgb\result_2018-08-23-11-18-31.png
python test_end_to_end.py
```

成功时终端会先打印 **`[OCID] Loaded bundle summary`** JSON（含 `label` 中实例 ID 列表与 `pcd_paths`）。随后 **`scene_objects`** 中会多出可选字段 **`ocid_frame`**，便于 VLM 了解当前帧有多少个标签实例及点云文件位置。

**说明**：当前 **YOLO+SAM 感知** 仍依赖 `ultralytics` 等；若环境未就绪，脚本会 **回退 mock `scene_objects`**，但 **OCID 的 depth/rgb/label/pcd 仍会如实加载并写入摘要**，便于你先验证数据通路。

### 4.3 可选测试数据（根目录占位文件）

将文件放在 **同一目录**：

- **`test_rgb.png`** — 彩色图（任意 OpenCV 可读格式）
- **`test_depth.npy`** — 二维数组 **`(H, W)`**，单位 **毫米（uint16 或可被解释为 mm 的数组）**，与学长 **`depth_data[y,x] / 1000.0`** 约定一致

若文件不存在，脚本会自动生成 **随机 RGB 与深度**，并打印 **waypoint** 与 **内参提示**；感知失败时会用内置 **fallback `reference_library`** 保证后续 VLM 与 FiLM 步骤仍可执行。

### 4.4 预期终端输出样例

以下为 **典型成功日志结构**（具体数值随随机种子、模型与 API 而变）。

**1）数据与感知**

```text
========================================================================
 VLA end-to-end: System 1 (Perception) -> System 2 (VLM) -> FiLM text_emb
========================================================================
[Data] test_rgb.png not found; using random RGB (480, 640, 3)
[Data] test_depth.npy not found; using random depth uint16 mm (480, 640)
[Data] waypoint_data (TCP / scan pose hint): {"world_pos": [320.0, 0.0, 350.0], ...}
...
[Perception] scene_objects (VLM JSON):
{
  "objects": [
    {"id": "apple_0", "class_name": "apple", "centroid_3d_m": [0.35, -0.12, 0.42]},
    ...
  ],
  "count": 2
}
```

**2）VLM 决策（无 API Key 时为回退匹配）**

```text
[VLM] Instruction: 'Please grasp the apple on the table.'
[VLM] decision:
{
  "action": "grasp",
  "target_id": "apple_0"
}
```

**3）FiLM 文本向量（需 `torch` + CLIP 安装成功）**

```text
[FiLM] Semantic phrase: 'Action: grasp. Target id: apple_0. Object class: apple.'
[FiLM] text_emb shape: (1, 512) (expected (1, 512) for ViT-B/32)
[FiLM] OK — ready to inject into Diffusion Policy / LanguageFiLM path.

========================================================================
 End-to-end script finished (exit 0).
========================================================================
```

若未安装 **`torch`** 或 **CLIP**，会出现 **`[FiLM] Skipped`** 及原因说明，**不视为脚本整体失败**；感知若因缺少 **ultralytics/SAM2** 失败，脚本会打印 traceback 并 **自动切换到 fallback 场景 JSON**，仍应看到合法的 **`scene_objects`** 与 VLM 的 **`decision`** JSON。

---

## 5. 常见问题

- **`ModuleNotFoundError: ultralytics`**：未装完整感知栈；可只依赖 **`test_end_to_end.py`** 的 fallback 验证 VLM + FiLM 链路，或按 `graduate_pro` 要求安装依赖。
- **`OPENAI_API_KEY` 已设置但解析失败**：检查网络与模型名；失败时会回退到本地匹配。
- **深度单位错误**：务必使用 **毫米** 与学长管线一致，否则 3D 质心与去重会偏差。

---

## 6. 相关路径速查

| 资源 | 路径 |
|------|------|
| 增强检测配置 | `graduate_pro/src/vision_ai/vision_ai/detection/config/enhanced_detection_config.json` |
| 感知管道核心 | `graduate_pro/src/vision_ai/vision_ai/detection/enhanced_detection_pipeline.py` |
| Language FiLM | `ADM_DP/policy/Diffusion-Policy/diffusion_policy/model/vision/language_film.py` |

---

*文档版本与 `test_end_to_end.py` 行为一致；若你修改接口，请同步更新本节。*
