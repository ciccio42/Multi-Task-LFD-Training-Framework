"""
Extends the optimized simulated pick_place dataset with the wrist/eye-in-hand
camera image.

optimize_dataset.py strips every observation key down to KEY_INTEREST when it
builds opt_dataset from no_opt_dataset, and older runs of it predate
eye_in_hand_image being added to KEY_INTEREST -- so opt_dataset/ur5e_pick_place
is missing it even though the raw no_opt_dataset trajectories were captured
with a wrist camera and still have it.

opt_dataset and no_opt_dataset are otherwise a 1:1, order-preserved match for
this task (same trajectory count, same per-step eef_pos, no timestep
subsampling happens for sim data) -- verified across all 16 tasks before
writing this script. So for each opt trajectory this only has to look up the
same (task, traj) file in the raw dataset and copy eye_in_hand_image over,
timestep for timestep, without touching anything else already in opt_dataset
(front image, bounding boxes, actions, ...).
"""
import glob
import os
import pickle as pkl

import numpy as np

RAW_PATH = os.environ.get(
    'ADD_WRIST_RAW_PATH',
    '/mnt/beegfs/frosa/robot_datasets/dataset/no_opt_dataset/pick_place/ur5e_pick_place')
OPT_PATH = os.environ.get(
    'ADD_WRIST_OPT_PATH',
    '/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/ur5e_pick_place')
OUT_PATH = os.environ.get('ADD_WRIST_OUT_PATH', OPT_PATH)

# eef_pos agreement check between the opt and raw frame at the same index,
# to catch a mismatched/reordered pairing before writing anything.
EEF_POS_ATOL = 1e-6


def _atomic_dump(obj, path):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pkl.dump(obj, f)
    os.replace(tmp_path, path)


def _add_wrist_to_traj(opt_traj, raw_traj, task_name, traj_name):
    # Mutate opt_traj's internal (still-JPEG-compressed) obs dicts directly,
    # instead of going through .get()/.append() (which would decompress and
    # then re-encode every image field, including camera_front_image, adding
    # a needless extra generation of JPEG loss to a field we don't even want
    # to touch). eye_in_hand_image is copied as the same compressed bytes
    # already sitting in the raw trajectory, so there's no re-encode at all.
    for t in range(len(opt_traj)):
        opt_obs = opt_traj._data[t][0]
        raw_obs = raw_traj._data[t][0]

        if not np.allclose(opt_obs['eef_pos'], raw_obs['eef_pos'], atol=EEF_POS_ATOL):
            raise ValueError(
                f'{task_name}/{traj_name} step {t}: eef_pos mismatch between '
                f"opt ({opt_obs['eef_pos']}) and raw ({raw_obs['eef_pos']}) -- "
                'frames are not aligned, refusing to add a mispaired wrist image')

        opt_obs['eye_in_hand_image'] = np.array(raw_obs['eye_in_hand_image'])
        opt_traj.change_obs(t, opt_obs)
    return opt_traj


if __name__ == '__main__':
    task_paths = sorted(glob.glob(os.path.join(OPT_PATH, 'task_*')))

    n_done = 0
    n_skipped = 0

    for task_path in task_paths:
        task_name = os.path.basename(task_path)
        print(f'Processing {task_name}')

        trjs = sorted(glob.glob(os.path.join(task_path, 'traj*.pkl')))
        for trj in trjs:
            traj_name = os.path.basename(trj)
            raw_trj_path = os.path.join(RAW_PATH, task_name, traj_name)

            if not os.path.exists(raw_trj_path):
                print(f'\tWarning: no raw trajectory for {task_name}/{traj_name}, skipping')
                n_skipped += 1
                continue

            with open(trj, 'rb') as f:
                opt_data = pkl.load(f)
            with open(raw_trj_path, 'rb') as f:
                raw_data = pkl.load(f)

            opt_traj = opt_data['traj']
            raw_traj = raw_data['traj']

            if len(opt_traj) != len(raw_traj):
                print(f'\tWarning: length mismatch for {task_name}/{traj_name} '
                      f'(opt={len(opt_traj)}, raw={len(raw_traj)}), skipping')
                n_skipped += 1
                continue

            try:
                new_traj = _add_wrist_to_traj(opt_traj, raw_traj, task_name, traj_name)
            except ValueError as e:
                print(f'\tWarning: {e}')
                n_skipped += 1
                continue

            os.makedirs(os.path.join(OUT_PATH, task_name), exist_ok=True)
            out_trj_path = os.path.join(OUT_PATH, task_name, traj_name)
            _atomic_dump({
                'traj': new_traj,
                'len': opt_data['len'],
                'env_type': opt_data['env_type'],
                'task_id': opt_data['task_id'],
            }, out_trj_path)
            n_done += 1

    print(f'\nDone. Trajectories updated: {n_done}, skipped: {n_skipped}')
