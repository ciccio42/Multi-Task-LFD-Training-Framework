from multi_task_il.datasets.savers import Trajectory
import pickle as pkl
import numpy as np
from PIL import Image

# DATASET_PATH = '/home/rsofnc000/dataset/opt_dataset/pick_place/real_new_ur5e_pick_place_old_2'
# OUT_PATH = '/home/rsofnc000/dataset/opt_dataset/pick_place/real_new_ur5e_pick_place'

# if __name__ == '__main__':
#     import os
#     import glob
#     import shutil
#     import debugpy
    
#     # print('Starting debugpy')
#     # debugpy.listen(('0.0.0.0', 5678))
#     # debugpy.wait_for_client()

#     # task paths
#     task_paths = glob.glob(os.path.join(DATASET_PATH, 'task_*'))
    
#     for task_path in task_paths:
#         print(f'Processing {task_path}')
#         task_name = task_path.split('/')[-1]
        
#         trjs= glob.glob(os.path.join(task_path, 'traj*.pkl'))
#         trjs.sort()
        
#         for trj in trjs:
#             # print(f'\tProcessing {trj}')
#             traj_name = trj.split('/')[-1]
#             # load pkl trajectory
#             with open(trj, 'rb') as f:
#                 data = pkl.load(f)
            
#             traj = data['traj']
#             new_traj = Trajectory()
            
#             for t in range(len(traj)):
                
#                 if t == 0:
#                     previous_pos = traj[t]['obs']['eef_pos']
#                     previous_t = t
#                     previous_gripper_state = traj[t]['action'][-1]
#                 else:
#                     # compute distance between current and previous position
#                     current_pos = traj[t]['obs']['eef_pos']
#                     gripper_state = traj[t-1]['action'][-1]
#                     distance = np.linalg.norm(current_pos - previous_pos)
#                     if distance > 0.05 and gripper_state == previous_gripper_state:
#                         # save the previous trajectory
#                         # pil_img = Image.fromarray(traj[previous_t]['obs']['camera_front_image'])
#                         # pil_img.save(f'{previous_t}.png')
#                         # pil_img = Image.fromarray(traj[t]['obs']['camera_front_image'])
#                         # pil_img.save(f'{t}.png')
#                         new_traj.append(traj[previous_t]['obs'], 
#                                         traj[previous_t]['reward'], 
#                                         traj[previous_t]['done'], 
#                                         traj[previous_t]['info'], 
#                                         traj[t-1]['action'])
#                         previous_pos = current_pos
#                         previous_t = t
#                     elif gripper_state != previous_gripper_state:
#                         # save the previous trajectory
#                         # pil_img = Image.fromarray(traj[previous_t]['obs']['camera_front_image'])
#                         # pil_img.save(f'change_gripper_{previous_t}.png')
#                         # pil_img = Image.fromarray(traj[t]['obs']['camera_front_image'])
#                         # pil_img.save(f'change_gripper_{t}.png')
#                         # save the previous trajectory
#                         new_traj.append(traj[previous_t]['obs'], 
#                                         traj[previous_t]['reward'], 
#                                         traj[previous_t]['done'], 
#                                         traj[previous_t]['info'], 
#                                         traj[t-1]['action'])
#                         previous_pos = current_pos
#                         previous_t = t
#                         previous_gripper_state = gripper_state
            
#             if len(new_traj) < 10:
#                 print(f'\t\tWarning: Trajectory {traj_name} is too short ({len(new_traj)} steps)')
                
#             # save the new trajectory
#             os.makedirs(os.path.join(OUT_PATH, task_name), exist_ok=True)
#             new_trj_path = os.path.join(OUT_PATH, task_name, traj_name)
            
#             with open(new_trj_path, 'wb') as f:
#                 pkl.dump({'traj': new_traj}, f)
#             # print(f'\t\tSaved {new_trj_path}')
    

import os
import glob
import pickle as pkl

def count_trajectories_length(out_folder):
    """
    Reads all tasks and .pkl files in out_folder, 
    loads each trajectory and prints its length.
    
    Args:
        out_folder (str): Path to the OUT_PATH directory
    """
    task_paths = glob.glob(os.path.join(out_folder, 'task_*'))
    
    total_files = 0
    total_steps = 0
    
    for task_path in task_paths:
        task_name = os.path.basename(task_path)
        traj_files = glob.glob(os.path.join(task_path, 'traj*.pkl'))
        
        print(f"Task: {task_name} ({len(traj_files)} trajectories)")
        
        for traj_file in traj_files:
            with open(traj_file, 'rb') as f:
                data = pkl.load(f)
            traj = data['traj']
            traj_len = len(traj)
            
            if traj_len < 10:
                print(f"Warning: Trajectory {traj_file} is too short ({traj_len} steps)")
                
                for t in range(traj_len):
                    print(f"Step {t}: {traj[t]['obs']['eef_pos']}")
                    img = traj[t]['obs']['camera_front_image']
                    pil_img = Image.fromarray(img)
                    pil_img.save(f'{traj_file}_step_{t}.png')
    
    print(f"\nSummary: {total_files} trajectories, {total_steps} total steps.")

# Example usage
if __name__ == '__main__':
    OUT_PATH = '/home/rsofnc000/dataset/opt_dataset/pick_place/real_new_ur5e_pick_place'
    count_trajectories_length(OUT_PATH)
            
                        

    
                
                
                