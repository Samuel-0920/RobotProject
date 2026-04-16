# MyRobotProject

Headless embodied AI (VLA) integration with a **3+1 layered stack**: perception → cognitive (VLM) → motor/language bridge → diffusion policy execution, plus **failure reflection** and retry.

| Layer | Key files | Role |
|-------|-----------|------|
| Perception | `perception_engine.py`, `graduate_pro/.../vision_ai` | YOLO + SAM2 + 3D features → `scene_json` |
| Cognitive | `cognitive_brain.py` | Instruction parsing, target choice, reflection |
| Motor bridge | `motor_adapter.py` | Text → CLIP-style `language_emb` for FiLM / policy |
| Execution | `ADM_DP/policy/Diffusion-Policy/*` | Diffusion policy training & rollout |
| Closed-loop eval | `vla_closed_loop_eval.py` | Full stack + vLLM reflection + optional video/log |
| DP-only sim eval | `ADM_DP/policy/Diffusion-Policy/eval_real_sim.py` | ManiSkill closed loop without VLM |
| DP smoke | `ADM_DP/policy/Diffusion-Policy/eval_ckpt_smoke.py` | Forward pass vs Zarr (no env) |

**Chinese operational guide:** [使用手册.md](./使用手册.md)

## Quick Start (smoke test)

```bash
git clone https://github.com/Samuel-0920/RobotProject.git
cd RobotProject
python test_pipeline.py
```

Expected:

1. `[Perception] JSON -> ...`
2. `[Brain] Decision -> ...`
3. `[Motor] Tensor Shape -> ...`

If CLIP/torch is missing, the motor line may show `unavailable`.

## Typical workflow (PickCube + DP + VLA)

Details and copy-paste commands: **[使用手册.md §6](./使用手册.md)**. Short outline:

1. **Vision weights** (YOLO/SAM2): `python scripts/download_vision_weights.py`
2. **Demos**: If teleop H5 has empty RGB (`obs_mode: none`), **replay** with RGB + env states, then **convert**: `python scripts/convert_h5_to_dp_zarr.py ...`
3. **Train DP** (from `ADM_DP/policy/Diffusion-Policy`): `python train.py --config-name=robot_dp task=pickcube_lang_minimal ...`
4. **Checkpoints**: Periodic saves go to **`ADM_DP/policy/Diffusion-Policy/checkpoints/<run_name>/`**, not under Hydra’s `data/outputs/...` (training logs/configs live there).
5. **Eval**: `eval_ckpt_smoke.py` → `eval_real_sim.py` → `vla_closed_loop_eval.py` (start **vLLM** before VLA eval; `--vlm_model` must match the server’s `--served-model-name`).

## Dataset sources (used in this project)

Current training/eval pipeline is built on **ManiSkill demonstrations/assets**.

### Primary dataset in use now

- **Task**: `PickCube-v1`
- **Demo file used**: `~/.maniskill/demos/PickCube-v1/teleop/trajectory.rgb.pd_joint_pos.physx_cpu.h5`
- **Converted training set**: `ADM_DP/data/zarr_data/PickCube_v1_lang_bridge.zarr`

Download command:

```bash
python -m mani_skill2.utils.download_demo PickCube-v1
```

### YCB extension dataset (for next-stage experiments)

- **Assets**: YCB object models (used by `PickSingleYCB`)
- **Demo task**: `PickSingleYCB-v0`

Download commands (recommended download target: `dataset/` under repo):

```bash
# in repo root
mkdir -p dataset
MANISKILL2_ASSET_DIR=$PWD/dataset python -m mani_skill2.utils.download_asset ycb
MANISKILL2_DATA_DIR=$PWD/dataset python -m mani_skill2.utils.download_demo PickSingleYCB-v0
```

### Official references

- ManiSkill repo: [https://github.com/haosulab/ManiSkill](https://github.com/haosulab/ManiSkill)
- ManiSkill docs: [https://maniskill.readthedocs.io](https://maniskill.readthedocs.io)
- YCB dataset project page: [https://www.ycbbenchmarks.com](https://www.ycbbenchmarks.com)

## `vla_closed_loop_eval.py` scene modes

- **`perception`** (default): YOLO+SAM2 → `scene_json` for reflection.
- **`sim_gt`**: Skip detectors; use simulator cube/goal poses (debug VLM grounding).
- **`hybrid`**: Run perception first; if empty/failure, fall back to `sim_gt` on PickCube.
- **`--perception_visualization`**: Save detection pipeline visualizations when perceiving.

## Project docs

| Document | Description |
|----------|-------------|
| [使用手册.md](./使用手册.md) | Setup, env vars, commands, Git/GitHub, troubleshooting |
| [外部来源说明.md](./外部来源说明.md) | Third-party / upstream attribution |
| [G1_ENV_SETUP.md](./G1_ENV_SETUP.md) | Environment bootstrap |
| [README_VLA_SYS.md](./README_VLA_SYS.md) | Architecture / design notes |

## Important notes

- Runtime outputs: `outputs/`, `detection_output_*`, local `eval_video/`, repo-root `models/` are **gitignored**; do not commit large weights or recordings.
- `graduate_pro/` and `ADM_DP/` are vendored source trees (not git submodules).
- **Git push**: If `origin` uses `git@github.com:...` but you authenticated `gh` with **HTTPS**, run `git remote set-url origin https://github.com/Samuel-0920/RobotProject.git` or configure SSH keys.
- Without `sudo`, install **GitHub CLI** via conda: `conda install -c conda-forge gh -y`, then `gh auth login` and `gh auth setup-git`.
