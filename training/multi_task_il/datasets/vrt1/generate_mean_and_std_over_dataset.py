import argparse
import pickle as pkl
import numpy as np
import glob
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt


def find_values_info(values):
    info = {
        "mean": np.mean(values),
        "std": np.std(values),
        "min_value": np.min(values),
        "max_value": np.max(values)
    }
    
    return info


def plot_values_info(info: dict, title: str, output_path, filename: str):
    labels = ['Mean', 'Standard Deviation', 'Minimum value', 'Maximum value']
    metrics = [info["mean"], info["std"], info["min_value"], info["max_value"]]

    plt.figure(figsize=(8, 5))
    # sns.barplot(x=np.arange(len(labels)), y=metrics)
    plt.bar(labels, metrics, color=[(30.0/255, 144.0/255, 255.0/255), (240.0/255, 230.0/255, 140.0/255), (34.0/255, 139.0/255, 34.0/255), (178.0/255, 34.0/255, 34.0/255)])
    plt.title(title)
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_file_path = os.path.join(output_path, filename)
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"File saved to {output_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot action bin histograms for VRT1 dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the VRT1 dataset.")
    
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        
    dx_values = list()
    dy_values = list()
    dz_values = list()
    roll_values = list()
    pitch_values = list()
    yaw_values = list()
    gripper_values = list()
    
    task_paths = glob.glob(os.path.join(args.dataset_path, 'task_*'))
    task_paths.sort()
    
    num_frame = 0
    for task_path in task_paths:
        task_name = task_path.split('/')[-1]
        
        trjs= glob.glob(os.path.join(task_path, 'traj*.pkl'))
        trjs.sort()
        
        for trj in tqdm(trjs, desc=f'Processing {task_name}'):
            traj_name = trj.split('/')[-1]
            
            with open(trj, 'rb') as f:
                data = pkl.load(f)
            
            traj = data['traj']
            
            for t in range(len(traj)):
                try:
                    action = traj.get(t)['action']
                except:
                    continue
                
                dx, dy, dz, roll, pitch, yaw, gripper = action
                
                dx_values.append(dx)
                dy_values.append(dy)
                dz_values.append(dz)
                roll_values.append(roll)
                pitch_values.append(pitch)
                yaw_values.append(yaw)
                gripper_values.append(gripper)
                
                num_frame += 1

    print(f"Total frames processed: {num_frame}")

    # transforming all lists to numpy arrays
    dx_values = np.array(dx_values)
    dy_values = np.array(dy_values)
    dz_values = np.array(dz_values)
    roll_values = np.array(roll_values)
    pitch_values = np.array(pitch_values)
    yaw_values = np.array(yaw_values)
    gripper_values = np.array(gripper_values)


    output_path = os.path.join(args.dataset_path, "images")
    os.makedirs(output_path, exist_ok=True)
    
    plot_values_info(info=find_values_info(dx_values),
                     title="Delta x info",
                     output_path=output_path,
                     filename="delta_x_info.png")
    
    plot_values_info(info=find_values_info(dy_values),
                     title="Delta y info",
                     output_path=output_path,
                     filename="delta_y_info.png")
    
    plot_values_info(info=find_values_info(dz_values),
                     title="Delta z info",
                     output_path=output_path,
                     filename="delta_z_info.png")
    
    plot_values_info(info=find_values_info(roll_values),
                     title="Roll info",
                     output_path=output_path,
                     filename="roll_info.png")
    
    plot_values_info(info=find_values_info(pitch_values),
                     title="Pitch info",
                     output_path=output_path,
                     filename="pitch_info.png")
    
    plot_values_info(info=find_values_info(yaw_values),
                     title="Yaw info",
                     output_path=output_path,
                     filename="yaw_info.png")
    
    plot_values_info(info=find_values_info(gripper_values),
                     title="Gripper info",
                     output_path=output_path,
                     filename="gripper_info.png")
