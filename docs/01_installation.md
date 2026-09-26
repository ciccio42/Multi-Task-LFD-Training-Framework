# ⚙️ Installation

[← Back to README](../README.md) · Next: [Data →](02_data.md)

## 0. MuJoCo 2.1 (`mujoco-py`)

Install MuJoCo 2.1.0 into `~/.mujoco/mujoco210` by following the *"Old bindings (≤ 2.1.1): mujoco-py"* section of [this guide](https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html). Then export:

```bash
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

Every launcher in `bashes/` sets these variables. Edit their paths to match your machine.

## 1. Conda environment

[`multi_task_lfd.yml`](../multi_task_lfd.yml) pins Python 3.9, PyTorch 2.8 (CUDA 12.8), `mujoco-py` 2.1.2.14, Hydra 1.3, W&B and `transformers`.

```bash
conda env create -f multi_task_lfd.yml
conda activate multi_task_lfd_cuda_12_8
```

> [!IMPORTANT]
> The SLURM launchers use `#SBATCH --export=ALL` and do **not** activate a conda env themselves. Run `conda activate multi_task_lfd_cuda_12_8` *before* `sbatch`.

## 2. robosuite (UR5e IK fork)

The environments need the patched robosuite on the `ur5e_ik` branch:

```bash
git clone -b ur5e_ik https://github.com/ciccio42/robosuite.git
cd robosuite
pip install -r requirements.txt
pip install -e .
```

## 3. This repository's packages

The repository contains three editable packages:

```bash
pip install -e tasks      # multi_task_robosuite_env  (envs, objects, experts)
pip install -e training   # multi_task_il, multi_task_il_gnn, train_scripts
pip install -e test       # multi_task_test
```

> [!WARNING]
> Other conda envs on a shared machine may have packages with the **same names** installed from a different project, such as VLA-Benchmark's `robosuite_test/tasks`. Check `python -c "import multi_task_il; print(multi_task_il.__file__)"` before assuming an env is the right one.

## 4. Optional dependencies

| Needed for | Install |
|---|---|
| MMDetection-based detectors | [MMDetection](https://mmdetection.readthedocs.io/en/stable/get_started.html) (vendored under `repo/mmdetection`) plus `pip install protobuf pyserial matplotlib psutil pyasn1-modules webencodings six==1.11.0 beautifulsoup4 defusedxml setproctitle mmengine` |
| Swin backbones | [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) |
| VIMA baseline | `repo/VIMA` |
| GTI / robomimic baselines | [robomimic fork](https://github.com/ciccio42/robomimic) |
| Remote debugging (`debug=true`) | `pip install debugpy`, then attach on port 5678 |

## 5. Smoke check

```bash
python -c "import robosuite, multi_task_robosuite_env, multi_task_il, torch; print(robosuite.__version__, torch.__version__, torch.cuda.is_available())"
```

`mujoco-py` compiles a native extension the first time it is imported. On clusters, run this check on a **compute node**, because login-node compilers can be incompatible.
