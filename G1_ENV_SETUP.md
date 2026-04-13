# g1_env：YOLOv8 + SAM2 感知环境配置（Windows / 本仓库）

用于在 **仅 RGB + Depth** 下调用 `ManiSkillPerceptionWrapper.perceive()` / `EnhancedDetectionPipeline.build_reference_library()`。

---

## 1. 创建并激活 Conda 环境

```powershell
conda create -n g1_env python=3.10 -y
conda activate g1_env
```

---

## 2. 安装 PyTorch（按你的 CUDA 版本）

到 [pytorch.org](https://pytorch.org) 选择 **CUDA 12.x / 11.8** 等对应命令，例如 CUDA 12.4：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

仅 CPU（较慢）：

```powershell
pip install torch torchvision
```

---

## 3. 安装 Python 依赖

在仓库根目录 `d:\personal project\code`：

```powershell
pip install -r requirements-g1-env.txt
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

若 SAM2 git 安装失败，可克隆后本地安装：

```powershell
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

---

## 4. 下载权重文件

仍在本仓库根目录执行：

```powershell
python scripts\download_vision_weights.py
```

将生成（相对 `detection/config/enhanced_detection_config.json` 解析后的绝对路径）：

- `graduate_pro\src\vision_ai\vision_ai\models\yolo\yolov8n.pt`
- `graduate_pro\src\vision_ai\vision_ai\models\sam2\sam2_hiera_large.pt`

配置已改为**相对路径**（见 `enhanced_detection_config.json`），`ConfigManager` 会自动解析为上述目录下的绝对路径。

**自定义 YOLO**：将你的 `best.pt` 放到 `models/yolo/`，并把 JSON 里 `detector.model_path` 改为 `models/yolo/best.pt`。

---

## 5. 验证 `cv2` 与 `perceive`

```powershell
python -c "import cv2; import ultralytics; import sam2; print('ok', cv2.__version__)"
```

端到端（含 OCID 一帧示例）：

```powershell
python test_end_to_end.py --ocid-rgb "D:\personal project\code\OCID-dataset\OCID-dataset\ARID10\table\top\fruits\seq09\rgb\result_2018-08-23-11-18-31.png"
```

若感知成功，应出现 **`[Perception] build_reference_library success=True`** 及非 fallback 的 `scene_objects`。

---

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| `No module named 'cv2'` | `pip install opencv-python` |
| `No module named 'ultralytics'` | `pip install ultralytics` |
| `No module named 'sam2'` | 按第 3 步安装 segment-anything-2 |
| SAM2 找不到 `sam2_hiera_l.yaml` | 确保 pip 安装的是官方 sam2 包，且 `config_name` 与版本一致 |
| CUDA OOM | 换 `yolov8n.pt`、或改 `segmentor.device` 为 `cpu`（JSON） |
| 检测框为 0 | `yolov8n` 为 COCO；水果场景可换学长训练的 `best.pt` 并调 `confidence_threshold` |

---

## 7. 可选：覆盖配置文件路径

`perception_wrapper.ManiSkillPerceptionWrapper` 在未传入 `config_file` 时会读取环境变量 **`VISION_CONFIG`**（指向另一份 `enhanced_detection_config.json` 的绝对路径）；未设置时使用默认：

`graduate_pro\src\vision_ai\vision_ai\detection\config\enhanced_detection_config.json`

```powershell
set VISION_CONFIG=D:\path\to\my_enhanced_detection_config.json
```
