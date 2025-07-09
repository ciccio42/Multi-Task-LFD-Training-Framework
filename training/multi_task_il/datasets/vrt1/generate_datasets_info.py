import argparse
import pickle as pkl
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt


label_color_dict = {
    "Delta X": "red",
    "Delta Y": "green",
    "Delta Z": "blue"
}


def compute_bin_index(value, lower_bound, upper_bound, number_of_bins):
    return int((value-lower_bound)/(upper_bound-lower_bound) * (number_of_bins - 1))


def initialize_dataset_dict(datasets_dict, dataset_name):
    datasets_dict[dataset_name] = dict()
    datasets_dict[dataset_name]["global"] = dict()
    
    datasets_dict[dataset_name]["global"]["x_list"] = list()
    datasets_dict[dataset_name]["global"]["y_list"] = list()
    datasets_dict[dataset_name]["global"]["z_list"] = list()

    datasets_dict[dataset_name]["global"]["dx_list"] = list()
    datasets_dict[dataset_name]["global"]["dy_list"] = list()
    datasets_dict[dataset_name]["global"]["dz_list"] = list()


def initialize_variation_dict(datasets_dict, dataset_name, variation_name):
    datasets_dict[dataset_name][variation_name] = dict()
    
    datasets_dict[dataset_name][variation_name]["x_list"] = list()
    datasets_dict[dataset_name][variation_name]["y_list"] = list()
    datasets_dict[dataset_name][variation_name]["z_list"] = list()

    datasets_dict[dataset_name][variation_name]["dx_list"] = list()
    datasets_dict[dataset_name][variation_name]["dy_list"] = list()
    datasets_dict[dataset_name][variation_name]["dz_list"] = list()


def update_variation_dict(datasets_dict, dataset_name, variation_name, eef_pos, action):
    # global 
    datasets_dict[dataset_name]['global']['x_list'].append(eef_pos[0])
    datasets_dict[dataset_name]['global']['y_list'].append(eef_pos[1])
    datasets_dict[dataset_name]['global']['z_list'].append(eef_pos[2])
    
    datasets_dict[dataset_name]['global']['dx_list'].append(action[0])
    datasets_dict[dataset_name]['global']['dy_list'].append(action[1])
    datasets_dict[dataset_name]['global']['dz_list'].append(action[2])
    
    # variation
    datasets_dict[dataset_name][variation_name]['x_list'].append(eef_pos[0])
    datasets_dict[dataset_name][variation_name]['y_list'].append(eef_pos[1])
    datasets_dict[dataset_name][variation_name]['z_list'].append(eef_pos[2])
    
    datasets_dict[dataset_name][variation_name]['dx_list'].append(action[0])
    datasets_dict[dataset_name][variation_name]['dy_list'].append(action[1])
    datasets_dict[dataset_name][variation_name]['dz_list'].append(action[2])


def plot_dataset_info(info, dataset_name, variation_name, dataset_path, output_path, labels: list):
    # converting lists to numpy arrays
    dxs = np.array(datasets_dict[dataset_name][variation_name]['dx_list'])
    dys = np.array(datasets_dict[dataset_name][variation_name]['dy_list'])
    dzs = np.array(datasets_dict[dataset_name][variation_name]['dz_list'])
    
    # Dati di esempio
    x_pos = np.arange(len(labels))

    # Dati numerici
    y_min = [np.min(dxs), np.min(dys), np.min(dzs)]
    y_max = [np.max(dxs), np.max(dys), np.max(dzs)]
    medians = [np.median(dxs), np.median(dys), np.median(dzs)]
    means = [np.mean(dxs), np.mean(dys), np.mean(dzs)]
    std_devs = [np.std(dxs), np.std(dys), np.std(dzs)]

    # Crea il grafico
    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(len(labels)):
        if i == 0:
            # a vuoto
            ax.vlines(x=x_pos[i], ymin=means[i] - std_devs[i], ymax=means[i] + std_devs[i],
                color='black', linewidth=3, label='Standard deviation')
        
            ax.plot(x_pos[i], y_min[i], 's', color='black', label='Min/Max')
            ax.plot(x_pos[i], y_max[i], 's', color='black')

            ax.plot(x_pos[i], medians[i], 'o', color='black', label='Median')

            ax.plot(x_pos[i], means[i], '^', color='black', label='Mean')
            
        ax.vlines(x=x_pos[i], ymin=means[i] - std_devs[i], ymax=means[i] + std_devs[i],
                  color=label_color_dict[labels[i]], linewidth=3)
        
        ax.plot(x_pos[i], y_min[i], 's', color='black')
        ax.plot(x_pos[i], y_max[i], 's', color='black')

        ax.plot(x_pos[i], medians[i], 'o', color='black')

        ax.plot(x_pos[i], means[i], '^', color='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Values', fontsize=14)
    ax.set_xlabel('Variables', fontsize=14)
    ax.legend()
    ax.grid(True, axis='y')
    plt.title(f"Info on {variation_name.replace('_', ' ')}")
    plt.tight_layout()

    os.makedirs(os.path.join(output_path, dataset_name), exist_ok=True)
    output_filename = f"{variation_name}.jpg"
    output_file_path = os.path.join(output_path, dataset_name, output_filename)
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"File saved to {output_file_path}")


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
    parser.add_argument("--output_path", type=str, required=True)
    
    parser.add_argument("--number_of_bins", type=int, default=256, help="Number of bins for histogram.")
    parser.add_argument("--normalization_range", type=float, nargs=2, default=[-1, 1], help="Normalization range for the histogram.")
    
    parser.add_argument("--debug", action='store_true', help="Enable debug mode.")
    args = parser.parse_args()

    number_of_bins = args.number_of_bins
    normalization_lower_bound = args.normalization_range[0]
    normalization_upper_bound = args.normalization_range[1]
    
    dx_bins = np.zeros(number_of_bins)
    dy_bins = np.zeros(number_of_bins)
    dz_bins = np.zeros(number_of_bins)
    
    datasets_dict = dict()

    if args.debug:
        import debugpy
        debugpy.listen(("localhost", 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

    dataset_path_dict = {"asu_table_top_delta": "/user/frosa/multi_task_lfd/datasets/datasets_delta/asu_table_top_delta", # ASU
                         "berkeley_autolab_ur5_delta": "/user/frosa/multi_task_lfd/datasets/datasets_delta/berkeley_autolab_ur5_delta", # Berkeley
                         "iamlab_cmu_pickup_insert_delta": "/user/frosa/multi_task_lfd/datasets/datasets_delta/iamlab_cmu_pickup_insert_delta", # CMU
                         "sim_panda_pick_place_converted_delta": "/user/frosa/multi_task_lfd/datasets/datasets_delta/sim_panda_pick_place_converted_delta", # Panda pick place
                         "taco_play_delta": "/user/frosa/multi_task_lfd/datasets/datasets_delta/taco_play_delta"} # Taco play
    
    # per ogni dataset, per ogni variazione
    # dx, dy e dz
    # x, y e z
    
    if os.path.exists("datasets_info.pkl"):
        print("Path exists.")
        with open("datasets_info.pkl", "rb") as f:
            datasets_dict = pkl.load(f)
    else:
        print("Path does not exist.")
        for dataset_path in dataset_path_dict.values():
            dataset_name = dataset_path.split("/")[-1]
            
            if dataset_name not in datasets_dict:
                initialize_dataset_dict(datasets_dict, dataset_name)
            
            for root, dirs, files in os.walk(dataset_path):
                # if "dataset_info.pkl" not in files:
                if not dirs:
                    # variation folder found
                    variation_name = root.split("/")[-1]
                    
                    initialize_variation_dict(datasets_dict=datasets_dict,
                                            dataset_name=dataset_name,
                                            variation_name=variation_name)
                    
                    trjs = [os.path.join(root, file) for file in files if file != "task_embedding.pkl"]
                    trjs.sort()
                    
                    for trj in tqdm(trjs, desc=f'Processing {root}'):
                        traj_name = trj.split('/')[-1]
                        
                        with open(trj, 'rb') as f:
                            data = pkl.load(f)
                        
                        traj = data['traj']
                        
                        for t in range(len(traj)):
                            # getting action for delta
                            try:
                                action = traj[t]['action']
                            except:
                                action = traj[t-1]['action']
                                
                            # getting eef position
                            eef_pos = traj[t]['obs']['eef_pos']
                            
                            # update datasets dict info
                            update_variation_dict(datasets_dict=datasets_dict,
                                                dataset_name=dataset_name,
                                                variation_name=variation_name,
                                                eef_pos=eef_pos,
                                                action=action)

        with open("datasets_info.pkl", "wb") as f:
            pkl.dump(datasets_dict, f)
        print("datasets_info.pkl saved!")
    
    for dataset_name in datasets_dict.keys():
        dataset_info = datasets_dict[dataset_name]
        for variation in dataset_info.keys():
            plot_dataset_info(info=dataset_info[variation], 
                              dataset_name=dataset_name,
                              variation_name=variation, 
                              dataset_path=dataset_path_dict[dataset_name],
                              output_path=args.output_path,
                              labels=['Delta X', 'Delta Y', 'Delta Z'])
