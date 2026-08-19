from multi_task_il.datasets.savers import Trajectory
import pickle as pkl
import numpy as np
import cv2
import os
import glob

# In-place overwrite: the subsampled trajectories are written back to the
# same dataset directory they are read from.
DATASET_PATH = '/mnt/beegfs/frosa/robot_datasets/dataset/opt_dataset/pick_place/real_eye_in_hand_ur5e_pick_place'
OUT_PATH = DATASET_PATH

# Directory where the debug visualizations (camera_front_image +
# eye_in_hand_image side by side, annotated with the action) are written.
# Kept outside of DATASET_PATH so it never gets picked up as trajectory data.
VIS_PATH = DATASET_PATH.rstrip('/') + '_subsample_vis'

# Minimum end-effector displacement (in meters) required between two
# successively kept frames.
STEP_SIZE = 0.05


def _make_vis_image(obs, action):
    """Builds a single image with camera_front_image and eye_in_hand_image
    side by side, with the action values printed on top of it."""
    front = obs.get('camera_front_image')
    eih = obs.get('eye_in_hand_image')

    if front is None:
        return None
    if eih is None:
        eih = np.zeros_like(front)
    if eih.shape != front.shape:
        eih = cv2.resize(eih, (front.shape[1], front.shape[0]))

    composite = np.concatenate([front, eih], axis=1).copy()

    action_str = 'action: [' + ', '.join(f'{a:.4f}' for a in np.asarray(action).flatten()) + ']'
    (text_w, text_h), baseline = cv2.getTextSize(
        action_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(composite, (0, 0), (min(text_w + 10, composite.shape[1]),
                  text_h + baseline + 10), (0, 0, 0), -1)
    cv2.putText(composite, action_str, (5, text_h + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return composite


def _save_vis_image(obs, action, task_name, traj_name, step_idx):
    composite = _make_vis_image(obs, action)
    if composite is None:
        return
    traj_stem = os.path.splitext(traj_name)[0]
    out_dir = os.path.join(VIS_PATH, task_name, traj_stem)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f'step_{step_idx:04d}.png'), composite)


def _atomic_dump(obj, path):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pkl.dump(obj, f)
    os.replace(tmp_path, path)


if __name__ == '__main__':
    task_paths = glob.glob(os.path.join(DATASET_PATH, 'task_*'))

    for task_path in task_paths:
        print(f'Processing {task_path}')
        task_name = task_path.split('/')[-1]

        trjs = glob.glob(os.path.join(task_path, 'traj*.pkl'))
        trjs.sort()

        for trj in trjs:
            traj_name = trj.split('/')[-1]
            with open(trj, 'rb') as f:
                data = pkl.load(f)

            traj = data['traj']
            new_traj = Trajectory()
            step_idx = 0

            def _keep_frame(t, previous_t):
                global step_idx
                new_traj.append(traj[previous_t]['obs'],
                                 traj[previous_t]['reward'],
                                 traj[previous_t]['done'],
                                 traj[previous_t]['info'],
                                 traj[t - 1]['action'])
                _save_vis_image(traj[previous_t]['obs'], traj[t - 1]['action'],
                                 task_name, traj_name, step_idx)
                step_idx += 1

            for t in range(len(traj)):
                if t == 0:
                    previous_pos = traj[t]['obs']['eef_pos']
                    previous_t = t
                    previous_gripper_state = traj[t]['action'][-1]
                else:
                    current_pos = traj[t]['obs']['eef_pos']
                    gripper_state = traj[t - 1]['action'][-1]
                    distance = np.linalg.norm(current_pos - previous_pos)

                    if distance > STEP_SIZE and gripper_state == previous_gripper_state:
                        _keep_frame(t, previous_t)
                        previous_pos = current_pos
                        previous_t = t
                    elif gripper_state != previous_gripper_state:
                        _keep_frame(t, previous_t)
                        previous_pos = current_pos
                        previous_t = t
                        previous_gripper_state = gripper_state

            if len(new_traj) < 10:
                print(f'\t\tWarning: Trajectory {traj_name} is too short ({len(new_traj)} steps)')

            os.makedirs(os.path.join(OUT_PATH, task_name), exist_ok=True)
            new_trj_path = os.path.join(OUT_PATH, task_name, traj_name)
            _atomic_dump({'traj': new_traj}, new_trj_path)
