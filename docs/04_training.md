# 🏋️ Training

[← Models](03_models.md) · [README](../README.md) · Next: [Testing →](05_testing.md)

## Entry point

All models except VIMA, the GNN and the video encoder train through **one** Hydra entry point:

```bash
python training/train_scripts/train_any.py \
    --config-path ../experiments --config-name config.yaml \
    policy='${mosaic}' \
    EXPERT_DATA=/path/to/opt_dataset \
    save_path=/path/to/checkpoints \
    task_names=[pick_place] set_same_n=2 \
    epochs=90 train_cfg.lr=2e-4 wandb_log=true
```

`train_any.py` selects the tasks, computes the batch size, and hands off to `Workspace` in [`train_utils.py`](../training/train_scripts/train_utils.py). `Workspace` owns the data loaders, the optimizer, the validation loop, checkpointing and W&B logging.

## Config files ([`training/experiments/`](../training/experiments/))

| Config | Use it for |
|---|---|
| `config.yaml` | MOSAIC, TOSIL, DAML and double policies (world-frame actions) |
| `config_convert_action.yaml` | Double policy with **base-frame actions** (`dataset_cfg.convert_action=true`); has matching normalization ranges |
| `config_convert_action*_resume.yaml` | Same, when resuming |
| `config_cond_target_obj_detector.yaml` | COD and COD policy |
| `config_obj_detector.yaml`, `config_target_obj.yaml` | Detector baselines |
| `config_real*.yaml`, `config_cond_target_obj_detector_real.yaml` | Real-robot fine-tuning (see [Sim-to-Real](06_real_world.md)) |
| `config_vima.yaml`, `config_gnn.yaml` | VIMA, GNN |
| `video_encoder/config_cond_module_finetuning.yaml` | Video encoder |

> [!WARNING]
> `convert_action=true` must be paired with `config_convert_action.yaml`. With the default `config.yaml`, base-frame actions get normalized against world-frame bounds.

## Common overrides

| Key | Meaning |
|---|---|
| `task_names=[…]` / `use_all_tasks` / `exclude_task` | Which tasks to train on |
| `set_same_n=N` | Samples per variation; the batch size is derived from it |
| `limit_num_traj`, `limit_num_demo` | Cap the trajectories per variation |
| `dataset_cfg.demo_name`, `dataset_cfg.agent_name` | Who demonstrates and who acts |
| `dataset_cfg.height`, `dataset_cfg.width` | Input resolution (e.g. 100×180) |
| `dataset_cfg.normalize_action`, `dataset_cfg.split_pick_place`, `dataset_cfg.change_command_epoch` | Dataset behavior |
| `samplers.balancing_policy` | How batches are balanced across variations |
| `optimizer`, `train_cfg.lr`, `train_cfg.weight_decay`, `train_cfg.lr_schedule`, `cosine_annealing` | Optimization |
| `early_stopping_cfg.patience` | Early stopping (`-1` disables it) |
| `resume`, `resume_path`, `resume_step`, `finetune` | Resuming or fine-tuning |
| `debug=true` | Waits for `debugpy` on port 5678 |

## SLURM launchers ([`bashes/`](../bashes/))

Each `train_*.sh` script sets per-task hyperparameters inside an `if TASK_NAME == …` block, then calls `train_any.py`:

```bash
conda activate multi_task_lfd_cuda_12_8
cd bashes
sbatch <launcher>.sh <task_name> [resume_folder] [resume_step] [finetune] [resume] [demo_name] [save_path] [use_wrist_img]
```

That is the typical argument order, taken from the `…_convert_wristimg.sh` launcher. Some launchers accept fewer or extra arguments, so check the `echo` block at the top of each script.

| Stage | Launcher |
|---|---|
| ① Train COD | `train_cond_target_obj_detector.sh` |
| ② Train MOSAIC + COD | `train_mosaic_target_obj_detector_double_policy.sh` |
| ② … with base-frame actions | `…_double_policy_convert.sh` |
| ② … + wrist camera | `…_double_policy_convert_wristimg.sh` |
| ② … held-out `0,5,10,15` | `…_convert_skip051015.sh` |
| ② COD policy | `train_cond_target_obj_detector_policy.sh` |
| Baselines | `train_mosaic.sh`, `train_tosil.sh`, `train_daml.sh`, `train_vima.sh`, `train_obj_detector.sh` |

Held-out spawn-region variants come as pairs: `train_cond_target_obj_detector[_policy]_{removed_spawn_regions,rm_one_spawn,rm_central_spawn}.sh`.

## Held-out generalization splits

| Split | How to get it |
|---|---|
| Unseen object (red box) | `skip_ids: [12,13,14,15]` (pick & place default) |
| Compositional (diagonal pairs) | `skip_ids: [0,5,10,15]` / `7_tasks_sim_skip_0_5_10_15.yaml` |
| Unseen spawn region | `*_removed_spawn_regions`, `*_rm_one_spawn`, `*_rm_central_spawn` launchers |

`val_skip` / `train_skip` control whether the samplers skip those IDs. `+dataset_cfg.validation_on_skipped_task=true` validates on exactly the held-out IDs. The key is not in the base config, so it needs the `+` prefix.

## Surviving walltime limits

The `run_bash_*.py` drivers (e.g. [`run_bash_cod_policy.py`](../bashes/run_bash_cod_policy.py)) submit a launcher, wait for the job to end, find the newest `model_save-<step>.pt`, and resubmit with `resume=true` from that step. They repeat this up to 20 times.

```bash
python bashes/run_bash_cod_policy.py
```

Edit `PROJECT_NAME`, `SAVE_PATH` and `BASH_SCRIPT` at the top of the driver. Keep each job's walltime within your partition's limit.

## Outputs

Checkpoints go to `<save_path>/<EXP_NAME>-Batch<bsize>/`:

| File | Content |
|---|---|
| `model_save-<step>.pt` | Model weights |
| optimizer state | Saved when `save_optim=true` |
| `config.yaml` | The resolved config; the tester reads this |

Metrics go to W&B under `project_name`.
