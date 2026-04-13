# MyRobotProject

Headless embodied AI (VLA) integration project with a clean **3+1 layered architecture**:

- `perception_engine.py` — perception and 3D anchoring
- `cognitive_brain.py` — speech + VLM decision + failure reflection
- `motor_adapter.py` — text-to-CLIP embedding bridge (`[1, 512]`)
- `test_pipeline.py` — minimal end-to-end smoke test

## Quick Start

```bash
git clone https://github.com/Samuel-0920/RobotProject.git
cd RobotProject
python test_pipeline.py
```

Expected output:

1. `[Perception] JSON -> ...`
2. `[Brain] Decision -> ...`
3. `[Motor] Tensor Shape -> ...`

If CLIP/torch is not installed, the motor line may show `unavailable`.

## Project Docs

| Document | Description |
|----------|-------------|
| [使用手册.md](./使用手册.md) | Full usage guide (setup, env vars, commands, examples) |
| [外部来源说明.md](./外部来源说明.md) | Third-party / upstream source attribution |
| [G1_ENV_SETUP.md](./G1_ENV_SETUP.md) | Environment bootstrap details |
| [README_VLA_SYS.md](./README_VLA_SYS.md) | Historical architecture notes |

## Important Notes

- Runtime artifacts are under `outputs/` and ignored by git.
- Heavy model binaries (`*.pt`, `*.pth`, `*.onnx`) are ignored; download them on each machine.
- `graduate_pro/` and `ADM_DP/` are integrated as source directories (not git submodules).
