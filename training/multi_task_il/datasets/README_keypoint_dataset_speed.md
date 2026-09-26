# Speed Optimization Suggestions for `multi_task_keypoint_dataset.py`

This document lists practical ideas to speed up sample loading for:

- `training/multi_task_il/datasets/multi_task_keypoint_dataset.py`

It is based on the current code path in:

- `__getitem__` in `multi_task_keypoint_dataset.py`
- `load_traj` in `datasets/__init__.py`
- `make_demo` and `create_sample` in `datasets/utils.py`

## Current Hot Path (Per Sample)

For each `__getitem__` call, the loader currently:

1. Loads two pickle trajectories from disk (`demo_file`, `agent_file`).
2. Builds demo frames (`make_demo`) with augmentations.
3. Builds agent frames, bounding boxes, optional actions/states (`create_sample`).
4. Applies a second augmentation branch when `aug_twice=True`.

This means disk I/O + Python object traversal + augmentation are all in the critical path.

## Highest Impact Suggestions (Priority Order)

## 1) Add an in-memory trajectory cache for `load_traj`

Why:
- `__getitem__` loads two files every time (`multi_task_keypoint_dataset.py`, around line 182).
- `load_traj` unpickles full sample objects every call (`datasets/__init__.py`, lines 33-49).

Suggestion:
- Add an LRU cache (`dict` + max size or `functools.lru_cache`) keyed by file path.
- Cache `(traj, command)` results for both demo and agent files.
- Optional: keep separate cache size limits for demo vs agent.

Expected effect:
- Big speed-up when the same file is reused across multiple pairs/epochs.

## 2) Avoid duplicate augmentation work when not needed

Why:
- `make_demo` and `create_sample` run extra frame processing when `aug_twice=True`.
- This effectively doubles many transform calls (`utils.py` lines 423-429 and 1043-1053).

Suggestion:
- Set `aug_twice=False` for throughput-focused runs.
- If you need two views only for a specific loss, gate it by training phase or probability.

Expected effect:
- Large CPU reduction in data workers.

## 3) Move repeated demo processing out of per-sample path

Why:
- `make_demo` is recomputed every sample even when demo trajectory repeats across pairs (`multi_task_keypoint_dataset.py`, line 192).

Suggestion:
- Cache final `demo_data` keyed by `(demo_file, task_name, demo_T, aug_config, image_size)`.
- If randomness in demo augmentation is required, cache decoded frames and re-augment lightly.

Expected effect:
- Significant reduction in redundant `traj.get(...)` + augmentation calls.

## 4) Reduce Python overhead in `create_sample`

Why:
- Per-timestep loop includes many branches, `copy.copy`, conversions, and repeated attribute lookups (`utils.py` lines 955+).

Suggestion:
- Hoist constant checks outside the loop (e.g. `is_real`, `perform_scale_resize`, `aug_twice`).
- Replace repeated `getattr(dataset_loader, ...)` inside loop with local variables.
- Remove unnecessary `copy.copy` when data is not mutated in-place.

Expected effect:
- Moderate but reliable speed-up, especially with small images and many workers.

## 5) Precompute dataset metadata once (avoid extra unpickling during indexing)

Why:
- `create_train_val_dict` opens many `.pkl` files to compute trajectory lengths while building index structures (`utils.py`, around the section that does `pkl.load` in index construction).

Suggestion:
- Save a sidecar metadata file (JSON/PKL) with per-file trajectory length and command.
- Reuse it unless data files changed.

Expected effect:
- Much faster startup / epoch reset for large datasets.

## DataLoader Configuration Suggestions

These are outside the dataset file but often give immediate gains:

- Use `num_workers > 0` and tune upward until CPU saturates.
- Enable `persistent_workers=True`.
- Increase `prefetch_factor` (e.g. 4-8).
- Use `pin_memory=True` when training on GPU.
- Ensure training data is on local SSD/NVMe when possible (instead of slower/shared storage paths).

## Medium-Term Refactors

## 1) Replace pickle with a random-access format

Why:
- Pickle forces Python-heavy deserialize paths.

Suggestion:
- Store frames/annotations in HDF5, LMDB, WebDataset shards, or zarr.
- Keep image bytes contiguous and indexed by sample id.

## 2) Precompute and store BB targets

Why:
- `create_gt_bb(...)` is executed online (`utils.py` lines 992-1002).

Suggestion:
- Offline-generate BB/class labels per timestep and store with trajectory.
- Keep online path only for rare dynamic cases.

## 3) Separate decode from augment

Why:
- decode + augment in one path makes caching difficult.

Suggestion:
- Stage 1: decode/select frames (cacheable).
- Stage 2: augmentation (random, per-iteration).

## Quick Profiling Checklist

To verify bottlenecks before/after changes:

1. Time `load_traj` separately for demo and agent.
2. Time `make_demo`.
3. Time `create_sample` with `aug_twice=False` and then `True`.
4. Compare end-to-end throughput (`samples/sec`) for different `num_workers`.
5. Check worker CPU utilization and disk read bandwidth.

## Minimal Change Set I Recommend First

If you want the fastest win with small code changes, do this first:

1. Add LRU cache around `load_traj` results.
2. Disable `aug_twice` (or make it conditional).
3. Add cached `demo_data` keyed by `demo_file`.
4. Tune DataLoader (`persistent_workers`, `prefetch_factor`, `num_workers`).

This usually gives the best speed/effort ratio before larger storage-format refactors.
