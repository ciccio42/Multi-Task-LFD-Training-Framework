# 🌍 Sim-to-Real

[← Testing](05_testing.md) · [README](../README.md)

The same models can be fine-tuned on a **real UR5e**, conditioned on human demonstration videos.

## Real datasets

| Agent (`dataset_cfg.agent_name`) | Camera setup |
|---|---|
| `real_new_ur5e` | Front camera |
| `real_eye_in_hand_ur5e` | Front camera + wrist camera |

Both use the same `task_XX/trajYYY.pkl` layout as the sim data, plus `train_/val_*_all_file_pairs.json` pair files (see [Data](02_data.md)). Tools for preparing real data are in [`datasets/real_world_dataset/`](../training/multi_task_il/datasets/real_world_dataset/):

| Script | Purpose |
|---|---|
| `sub_sample.py`, `sub_sample_delta.py` | Temporal subsampling and delta actions |
| `add_bb_real_dataset.py` | Bounding-box annotation |

## Action frame

Real-robot actions are expressed in the robot's **base_link** frame. Set `dataset_cfg.convert_action=true` so sim actions are converted into that frame too, and use a config whose `normalization_ranges` match:

| Setting | Config |
|---|---|
| sim, base-frame | `config_convert_action.yaml` |
| real fine-tuning | `config_real_ConvertAction_finetune.yaml` |
| real fine-tuning + wrist | `config_real_ConvertAction_WristImg_finetune.yaml` |

## Launchers

| Goal | Launcher / driver |
|---|---|
| Real COD | `real_train_cond_target_obj_detector.sh`, `real_train_keypoint_detection.sh` |
| Real MOSAIC + COD | `real_train_mosaic_target_obj_detector_double_policy.sh` |
| Sim → real fine-tuning | `real_train_…_double_policy_sim2real.sh` |
| … with base-frame actions | `…_sim2real_ConvertAction.sh` |
| … + wrist camera | `…_sim2real_ConvertAction_WristImg.sh` |
| Auto-resubmitting drivers | `run_bash_sim2real_policy.py`, `run_bash_sim_convert*_policy.py` |
| Real COD evaluation | `real_test_cond_target_obj_detector.sh`, `real_test_mosaic_cond_target_obj_detector.sh` |

## Recommended recipe

1. Train COD and the base-frame double policy in simulation (`…_convert[_wristimg].sh`).
2. Train or fine-tune COD on real frames.
3. Fine-tune the policy on real data from the sim checkpoint (`finetune=true`, `…_sim2real_ConvertAction[_WristImg].sh`), with `demo_name=human_rgb`.
4. Check offline with `test/multi_task_test/test_real_world_dataset.py` before deploying on the robot.
