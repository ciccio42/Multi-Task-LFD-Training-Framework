<div align="center">

# 🎬➡️🦾 Multi-Task LfD Training Framework

### *Show it once. Watch it do it.*

**One-shot, multi-task Learning from Demonstration: a robot watches a single demonstration video (from a human or another robot) and performs the same task in its own scene.**

![Simulator](https://img.shields.io/badge/sim-robosuite%20(UR5e%20IK%20fork)%20%2B%20MuJoCo%202.1-blue)
![Robots](https://img.shields.io/badge/robots-UR5e%20%7C%20Panda%20%7C%20Sawyer-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8%20%2B%20CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white)
![Config](https://img.shields.io/badge/config-Hydra-89b8cd)
![Logging](https://img.shields.io/badge/logging-W%26B-FFBE00)

[Installation](docs/01_installation.md) •
[Data](docs/02_data.md) •
[Models](docs/03_models.md) •
[Training](docs/04_training.md) •
[Testing](docs/05_testing.md) •
[Sim-to-Real](docs/06_real_world.md)

</div>

---

A human picks up the **red box** and drops it in the **third bin**. The robot has never been told what the task is. It only sees that video, and it has to do the same thing in its own scene, with its own body, and with the objects in different places.

This repository contains everything needed to study that problem:

- a **multi-task robosuite environment** with scripted experts
- **paired demonstration/agent datasets**
- a collection of **one-shot imitation policies**, from MOSAIC and TOSIL/DAML baselines to our **Conditioned Target Object Detector (COD)** pipeline
- a **simulation test harness** for rollouts
- a **sim-to-real** fine-tuning path for a real UR5e with a wrist camera

## ✨ Highlights

- 🧩 **4 task families, 60+ variations.** Pick & place, nut assembly, block stacking, button press, with UR5e, Panda and Sawyer agents.
- 👤 **Cross-embodiment demonstrations.** Train with `human_rgb` or `panda` demonstrations conditioning a `ur5e` agent.
- 🎯 **Object-centric conditioning.** COD reads the demonstration and outputs a bounding box around the *target object* in the agent's view. A second policy then acts on that box.
- 🔬 **Held-out generalization splits.** Skip task IDs (unseen object→bin pairs, unseen objects) or spawn regions at training time and test on them.
- 🌍 **Sim-to-real.** Fine-tune on real UR5e data with base-frame action conversion and wrist-camera input.
- ⚙️ **Hydra-driven, SLURM-ready.** One entry point (`train_any.py`) for every model. Launchers auto-resubmit to get past walltime limits.

## 🧠 Model zoo

| Model | `policy=` | Class | What it does |
|---|---|---|---|
| **MOSAIC** | `${mosaic}` | `mt_rep.VideoImitation` | Transformer video imitation with self-supervised (contrastive/BYOL) and inverse-dynamics auxiliary losses |
| **MOSAIC + COD (double policy)** ⭐ | `${mosaic}` + `mosaic._target_=…mt_rep_double_policy.VideoImitation` | `mt_rep_double_policy.VideoImitation` | MOSAIC policy conditioned on the bounding boxes COD predicts |
| **COD (Conditioned Target Object Detector)** | `${cond_target_obj_detector}` | `CondTargetObjectDetector` | Demonstration-conditioned (FiLM) detector that localizes the target object |
| **COD policy** | `${cond_policy}` | `CondPolicy` | Lightweight policy on top of a pretrained COD |
| No-MOSAIC double policy | `${mosaic}` + `…no_mosaic_double_policy.VideoImitation` | `no_mosaic_double_policy.VideoImitation` | Ablation without the MOSAIC representation losses |
| **TOSIL** | `${tosil}` | `baselines.InverseImitation` | Transformer one-shot imitation baseline |
| **DAML** | `${daml}` | `baselines.DAMLNetwork` | Meta-learning one-shot imitation baseline |
| **VIMA** | `${vima}` | `vima.policy.Policy` | Multimodal-prompt transformer policy |
| **GNN policy** | via `train_scripts/gnn` | `multi_task_il_gnn` | Scene-graph policy |
| **Command / video encoder** | `${cond_module}` | `CondModule` | Encodes demonstration videos (or text commands) into task embeddings |

➡️ Details: [docs/03_models.md](docs/03_models.md)

## ⚡ Quickstart

```bash
# 1. Environment (see docs/01_installation.md)
conda env create -f multi_task_lfd.yml
conda activate multi_task_lfd_cuda_12_8
pip install -e tasks -e training -e test

# 2. Train MOSAIC + COD on pick & place (SLURM)
cd bashes
sbatch train_mosaic_target_obj_detector_double_policy_convert.sh pick_place

# 3. Roll it out in simulation
python ../test/multi_task_test/test_any_task.py <checkpoint_dir> \
    --env pick_place --saved_step <step> --eval_each_task 10 \
    --controller_path ../tasks/multi_task_robosuite_env/controllers/config/osc_pose.json \
    --human_demo --save_files
```

## 🗂️ Repository structure

```
Multi-Task-LFD-Training-Framework/
├── multi_task_lfd.yml            # Conda env (Python 3.9, torch 2.8 + cu128, mujoco-py 2.1)
├── docs/                         # 📚 Documentation (start here)
│
├── tasks/                        # 🌍 Simulation (package: multi_task_robosuite_env)
│   ├── multi_task_robosuite_env/ #   arenas, objects, task envs, controllers, scripted experts
│   └── collect_data/             #   parallel expert-demonstration collection
│
├── training/                     # 🧠 Learning (package: multi_task_il)
│   ├── experiments/              #   Hydra configs: config*.yaml, tasks_cfgs/, video_encoder/
│   ├── multi_task_il/
│   │   ├── datasets/             #   paired demo/agent datasets, samplers, augmentation, real-world tools
│   │   ├── models/               #   MOSAIC, COD, baselines, VIMA, command encoder
│   │   └── utils/                #   LR schedulers, early stopping
│   ├── multi_task_il_gnn/        #   scene-graph datasets + GNN policy
│   └── train_scripts/            #   train_any.py (entry point), train_utils.py (Trainer), train_vima.py, gnn/, video_encoder/
│
├── test/                         # 🧪 Evaluation (package: multi_task_test)
│   ├── multi_task_test/          #   test_any_task.py + per-task rollout logic
│   └── multi_task_test_gnn/
│
├── bashes/                       # 🚀 SLURM launchers: train_*, test_*, real_*, run_bash_*.py
└── repo/                         # third-party: mmdetection, VIMA
```

## 📚 Documentation

| # | Guide | What's inside |
|---|---|---|
| 1 | [Installation](docs/01_installation.md) | MuJoCo, conda env, robosuite fork, editable packages, optional deps |
| 2 | [Data](docs/02_data.md) | Tasks and variations, dataset layout, trajectory format, collection, pair files |
| 3 | [Models](docs/03_models.md) | Every policy, how COD plugs into MOSAIC, key config knobs |
| 4 | [Training](docs/04_training.md) | Hydra configs, launchers, held-out splits, resuming, auto-resubmission |
| 5 | [Testing](docs/05_testing.md) | Simulation rollouts, metrics, outputs, batch testing |
| 6 | [Sim-to-Real](docs/06_real_world.md) | Real UR5e datasets, action conversion, wrist camera, fine-tuning |

## 🙏 Acknowledgements

Built on [MOSAIC](https://github.com/rll-research/mosaic), [robosuite](https://robosuite.ai), [MuJoCo](https://mujoco.org), [VIMA](https://github.com/vimalabs/VIMA), [MMDetection](https://github.com/open-mmlab/mmdetection) and [Hydra](https://hydra.cc).
