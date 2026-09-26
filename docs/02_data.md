# 🗃️ Data

[← Installation](01_installation.md) · [README](../README.md) · Next: [Models →](03_models.md)

## Tasks

Task definitions live in [`training/experiments/tasks_cfgs/`](../training/experiments/tasks_cfgs/). The default set is `7_tasks.yaml`.

| Task (`name`) | Variations | Image crop `[top, left, bottom, right]` |
|---|:-:|---|
| `pick_place` | 16 (4 objects × 4 bins) | `[20, 25, 80, 75]` |
| `nut_assembly` | 9 | `[20, 25, 80, 75]` |
| `press_button_close_after_reaching` (button) | 6 | `[10, 10, 70, 70]` |
| `stack_block` | 6 | `[20, 25, 80, 75]` |
| `door` | 4 | none |
| `drawer` | 8 | none |
| `basketball` | 12 | none |

Each task entry also sets the following:

| Key | Meaning |
|---|---|
| `task_ids` | Variations to load |
| `skip_ids` | Variations **held out** from training, e.g. `[12,13,14,15]` for pick & place |
| `n_per_task` | Samples per variation in each batch |
| `traj_per_subtask` / `demo_per_subtask` | How many agent and demonstration trajectories to use per variation |
| `crop` / `demo_crop` | Crops for the agent and demonstration frames |

Other presets:

| Preset | Contents |
|---|---|
| `7_tasks_sim_skip_0_5_10_15.yaml` | Compositional split |
| `7_tasks_real.yaml` | Real robot |
| `7_tasks_mix_obj.yaml` | Mixed object sets |

## Dataset layout

Set `EXPERT_DATA` to the dataset root:

```
<EXPERT_DATA>/
└── pick_place/
    ├── human_rgb_pick_place/           # demonstrator: human video
    │   └── task_00 … task_15/traj000.pkl …
    ├── panda_pick_place/               # demonstrator: Panda robot
    ├── ur5e_pick_place/                # agent: UR5e (sim)
    ├── real_new_ur5e_pick_place/       # agent: real UR5e (front camera)
    ├── real_eye_in_hand_ur5e_pick_place/   # agent: real UR5e + wrist camera
    ├── train_<demo>_<agent>_all_file_pairs.json
    └── val_<demo>_<agent>_all_file_pairs.json
```

Training draws **(demonstration, agent) pairs** of the same variation. For each demo/agent combination, the `*_all_file_pairs.json` index maps a sample index to `[task_name, variation_id, demo_pkl, agent_pkl]`. The demonstrator and agent are chosen with `dataset_cfg.demo_name` (`human_rgb` | `panda`) and `dataset_cfg.agent_name` (`ur5e` | `real_new_ur5e` | `real_eye_in_hand_ur5e`).

The pair-index generators are in [`training/multi_task_il/datasets/dataset_paths_generation_utils/`](../training/multi_task_il/datasets/dataset_paths_generation_utils/).

## Trajectory format

Each `trajXXX.pkl` holds a `multi_task_il.datasets.savers.Trajectory`, a sequence of per-step dicts with JPEG-compressed images. For a UR5e agent, each step typically contains:

| Key | Content |
|---|---|
| `camera_front_image` | Third-person RGB (rendered at 200×360) |
| `eye_in_hand_image` | Wrist RGB (only in datasets that include it) |
| `ee_aa` | End-effector position + axis-angle (6) |
| `gripper_qpos` | Gripper joint positions |
| `action` | 7-D: xyz, axis-angle, gripper (−1 open / +1 close) |
| bounding boxes | Per-object boxes, used by COD / `concat_bb` |

Human demonstration trajectories contain only `camera_front_image`.

## Collecting simulated data

The scripted experts in `tasks/multi_task_robosuite_env/controllers/controllers/expert_*.py` generate demonstrations in parallel:

```bash
cd tasks/collect_data
python collect_task.py <SAVE_DIR> \
    --task_name pick_place --robot ur5e \
    --N 1600 --per_task_group 100 --n_tasks 16 \
    --num_workers 10 --collect_cam --overwrite \
    --ctrl_config ../multi_task_robosuite_env/controllers/config/osc_pose.json \
    --object_set 1
```

[`collect_data.sh`](../tasks/collect_data/collect_data.sh) has ready-made configurations for every task and robot.

## Dataset utilities

| Script | Purpose |
|---|---|
| `datasets/optimize_dataset.py` (`.sh`) | Build the optimized (`opt_dataset`) version |
| `datasets/create_bb.py` | Compute bounding boxes for trajectories |
| `datasets/add_eye_in_hand_sim_dataset.py` | Add wrist-camera frames to sim data |
| `datasets/real_world_dataset/sub_sample*.py`, `add_bb_real_dataset.py` | Subsample and annotate real-robot data |
| `datasets/compute_frame_distribution.py` | Frame and object statistics |
| `utils/dataset_analysis_ur.py` | Dataset sanity analysis |
