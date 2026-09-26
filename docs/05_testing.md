# 🧪 Testing in simulation

[← Training](04_training.md) · [README](../README.md) · Next: [Sim-to-Real →](06_real_world.md)

## Rollouts

[`test/multi_task_test/test_any_task.py`](../test/multi_task_test/test_any_task.py) loads a checkpoint directory (it reads the saved `config.yaml`), builds the robosuite environment, feeds the policy a demonstration and rolls it out:

```bash
python test/multi_task_test/test_any_task.py <checkpoint_dir> \
    --env pick_place \
    --saved_step 70 \
    --eval_each_task 10 \
    --num_workers 1 --gpu_id 0 \
    --controller_path tasks/multi_task_robosuite_env/controllers/config/osc_pose.json \
    --human_demo \
    --save_path <checkpoint_dir>/results_pick_place/run_1 \
    --save_files --wandb_log --project_name <name>
```

| Flag | Meaning |
|---|---|
| `--env` | `pick_place`, `nut_assembly`, `stack_block`, `button`, … |
| `--saved_step` | Which `model_save-<step>.pt` to load |
| `--eval_each_task` | Rollouts per variation |
| `--variation`, `--seed` | Run a single variation or seed |
| `--human_demo` | Condition on human videos instead of robot demonstrations |
| `--validate_on_train_ids` | Also evaluate on training variations |
| `--baseline` | Evaluate a baseline (`daml`, `tosil`, …) |
| `--gt_bb` / `--test_gt` / `--gt_action` | Oracle ablations: ground-truth boxes or actions |
| `--obj_detector_path`, `--obj_detector_step` | External COD checkpoint |
| `--size` / `--shape` / `--color` | Object-appearance perturbations |
| `--save_files` | Save `traj<n>.pkl` + `traj<n>.json` per rollout |
| `--debug` | `debugpy` on port 5678 |

## Metrics

Each rollout reports nested flags:

| Metric | Meaning |
|---|---|
| **reached** | The gripper reached the correct object |
| **picked** | The correct object was lifted |
| **success** | The object ended up in the correct bin, peg or target |

If `traj<n>.json` already exists, it is loaded instead of rerun, so an interrupted test resumes where it stopped.

## Batch testing

The `bashes/test_*.sh` scripts loop over checkpoints, steps and runs:

| Script | Purpose |
|---|---|
| `test_mosaic_cond_target_obj.sh <task>` | MOSAIC + COD |
| `test_cond_target_obj_detector.sh` | Detector-only evaluation |
| `test_cod_spawn_region_variants.sh` | Spawn-region splits |
| `test_mosaic.sh`, `test_tosil.sh`, `test_daml.sh`, `test_vima.sh` | Baselines |
| `smoke_test_*.sh` | Quick checks: RGB/BGR input, wrist image, held-out splits |

## Other tools

| Script | Purpose |
|---|---|
| `test/multi_task_test/predict_bb_on_traj.py` (+ `run_predict_bb_on_traj.sh`) | Draw COD predictions on saved trajectories |
| `test/multi_task_test/test_real_world_dataset.py` | Run a model offline on real-robot trajectories |
| `bashes/create_test_videos.py` (`.sh`) | Render rollout videos |

> Want to compare against modern VLAs on the same UR5e pick & place world? See the companion **VLA-Benchmark** repository.
