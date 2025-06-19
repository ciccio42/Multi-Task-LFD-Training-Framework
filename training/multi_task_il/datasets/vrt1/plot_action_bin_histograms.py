import argparse
import pickle as pkl
import numpy as np
import glob
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt


def compute_bin_index(value, lower_bound, upper_bound, number_of_bins):
    return int((value-lower_bound)/(upper_bound-lower_bound) * (number_of_bins - 1))


def plot_action_bin_histograms(values, filename):
    plt.figure()  # Create a new figure
    sns.barplot(x=np.arange(len(values)), y=values)
    plt.title("Bin Counts")
    plt.xlabel("Bin Index")
    plt.ylabel("Count")
    plt.savefig(f"{filename}", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot action bin histograms for VRT1 dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the VRT1 dataset.")
    
    parser.add_argument("--number_of_bins", type=int, default=256, help="Number of bins for histogram.")
    parser.add_argument("--normalization_range", type=float, nargs=2, default=[-0.25, 0.25], help="Normalization range for the histogram.")
    
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")
    args = parser.parse_args()

    number_of_bins = args.number_of_bins
    normalization_lower_bound = args.normalization_range[0]
    normalization_upper_bound = args.normalization_range[1]
    
    dx_bins = np.zeros(number_of_bins)
    dy_bins = np.zeros(number_of_bins)
    dz_bins = np.zeros(number_of_bins)

    if args.debug:
        import debugpy
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        
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
                
                dx, dy, dz = action[:3]
                
                dx_bin = compute_bin_index(dx, normalization_lower_bound, normalization_upper_bound, number_of_bins)
                dy_bin = compute_bin_index(dy, normalization_lower_bound, normalization_upper_bound, number_of_bins)
                dz_bin = compute_bin_index(dz, normalization_lower_bound, normalization_upper_bound, number_of_bins)
                
                dx_bins[dx_bin] += 1
                dy_bins[dy_bin] += 1
                dz_bins[dz_bin] += 1
                
                num_frame += 1

    print(f"Total frames processed: {num_frame}")

    plot_action_bin_histograms(dx_bins, os.path.join(args.dataset_path, 'dx_histogram.png'))
    plot_action_bin_histograms(dy_bins, os.path.join(args.dataset_path, 'dy_histogram.png'))
    plot_action_bin_histograms(dz_bins, os.path.join(args.dataset_path, 'dz_histogram.png'))
