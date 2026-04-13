# VLA 系统说明（RobotProject）

本文档聚焦当前仓库已落地实现：  
**感知（System 1）+ 认知（System 2）+ 运动语义桥接（System 3）+ DP 小脑执行（+1）**。

与综合手册的关系：  
- `使用手册.md`：偏操作与命令。  
- 本文档：偏系统设计、职责边界、演进路线。

---

## 1. 设计目标

1. 让 VLM 不只“看文字”，而是基于真实场景对象 ID 做可执行决策。  
2. 让高层语义可注入到底层控制策略（`language_emb`）。  
3. 失败后可反思、重条件化、再尝试，形成闭环迭代。

---

## 2. 模块职责图

| 系统 | 主要文件 | 核心能力 | 产物 |
|------|------|------|------|
| System 1 感知 | `perception_engine.py` + `graduate_pro/vision_ai` | YOLO + SAM2 + 3D 特征聚合 | `scene_json`（含对象 `id`） |
| System 2 认知 | `cognitive_brain.py` | 指令解析、目标选择、失败反思 | `action/target_id/macro_action` |
| System 3 运动适配 | `motor_adapter.py` | 语义短句构造 + CLIP 编码 | `language_emb (1,512)` |
| +1 执行 | `ADM_DP/policy/Diffusion-Policy` | 扩散策略预测连续动作 | env `step(action)` |

关键闭环脚本：
- `vla_closed_loop_eval.py`：主评测入口。
- `test_pipeline.py`：轻量冒烟。
- `eval_real_sim.py`、`eval_ckpt_smoke.py`：DP 侧验证。

---

## 3. 真实执行链路

```text
obs(rgb,depth,state)
  -> System 1 perceive()
  -> reference_library
  -> scene_json(objects with real ids)

instruction + scene_json
  -> System 2 decide()
  -> {action, target_id}

decision
  -> build_semantic_phrase_for_film()
  -> encode_text_emb_for_film()
  -> language_emb

obs + language_emb
  -> DP policy
  -> action -> env.step()

if fail:
  reflect_on_failure / local VLM multimodal reflect
  -> new decision -> new language_emb
  -> retry phase
```

---

## 4. 当前实现状态（与你仓库一致）

- 已完成 `language_emb` 在数据集、训练、推理、闭环中的端到端接线。
- 已支持本地 vLLM 作为反思引擎（`--vlm_base_url` / `--vlm_model`）。
- 已补齐 SAM2 权重，感知初始化可正常加载 `sam2_hiera_large.pt`。
- 已有反思日志与视频输出，支持分析失败原因。

---

## 5. 现阶段主要风险

1. **场景对象可见性风险**  
   若 YOLO 没框到目标，`scene_json` 为空，即使 SAM2 权重正常也无法产出对象 ID。

2. **动作词表漂移风险**  
   反思模型输出动作若偏离训练动作空间，会降低可执行性。

3. **重试预算不足风险**  
   如果 Phase A 用光步数，Phase D 重试窗口过短，反思价值无法体现。

---

## 6. 下一步发展方向

### A. 让 `target_id` 真正可执行（近期最高优先）
- 目标：稳定让 `scene_json.count > 0`，且 `target_id in objects.id`。
- 手段：调检测阈值、视角、分辨率；记录并统计每轮 `scene_count`。

### B. 反思输出标准化
- 增加动作映射层，把自由文本动作归一到训练词表。
- 对 `target_id` 做存在性校验，不存在时走 `macro_action`（如 `relocalize`）。

### C. 闭环策略优化
- 调整 `phase_steps` 比例，确保“失败后仍有充足重试步数”。
- 探索“反思后 reset 再尝试”策略，提高 recovery 成功率。

### D. 数据与训练增强
- 扩充语义多样性、目标实例多样性。
- 在训练集中增加更贴近闭环场景的失败恢复样本。

### E. 指标化评估
- 固化指标：成功率、ID 命中率、反思触发率、二次成功率、平均恢复步数。
- 输出统一 JSONL 以支撑横向对比。

---

## 7. 关键路径速查

| 资源 | 路径 |
|------|------|
| 闭环评测入口 | `vla_closed_loop_eval.py` |
| 感知封装 | `perception_engine.py` |
| 认知决策 | `cognitive_brain.py` |
| 语义向量桥接 | `motor_adapter.py` |
| H5→Zarr | `scripts/convert_h5_to_dp_zarr.py` |
| 权重下载 | `scripts/download_vision_weights.py` |
| SAM2 checkpoint 目录 | `graduate_pro/src/vision_ai/vision_ai/models/sam2` |
| DP 任务配置 | `ADM_DP/policy/Diffusion-Policy/diffusion_policy/config/task/pickcube_lang_minimal.yaml` |

---

文档约定：若代码与文档冲突，以当前代码为准，并同步更新本文件与 `使用手册.md`。
