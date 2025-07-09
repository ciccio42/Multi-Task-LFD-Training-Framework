import argparse
import os
import torch
import random
import numpy as np
from omegaconf import OmegaConf
import hydra
from collections import OrderedDict
from multi_task_il.datasets.command_encoder.command_encoder_dataset import CommandEncoderFinetuningDataset
import json
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
from collections import defaultdict


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


mivia_task_to_color = {
    "human_rgb_pick_place/task_00": (0.0/255, 100.0/255, 0.0/255),       # verde scuro
    "human_rgb_pick_place/task_01": (34.0/255, 139.0/255, 34.0/255),     # verde foresta
    "human_rgb_pick_place/task_02": (50.0/255, 205.0/255, 50.0/255),     # limegreen
    "human_rgb_pick_place/task_03": (124.0/255, 252.0/255, 0.0/255),     # verde prato
    "human_rgb_pick_place/task_04": (255.0/255, 215.0/255, 0.0/255),     # oro
    "human_rgb_pick_place/task_05": (255.0/255, 165.0/255, 0.0/255),     # arancione
    "human_rgb_pick_place/task_06": (255.0/255, 255.0/255, 0.0/255),     # giallo classico
    "human_rgb_pick_place/task_07": (240.0/255, 230.0/255, 140.0/255),   # khaki
    "human_rgb_pick_place/task_08": (0.0/255, 0.0/255, 139.0/255),       # blu scuro
    "human_rgb_pick_place/task_09": (0.0/255, 0.0/255, 205.0/255),       # blu medio
    "human_rgb_pick_place/task_10": (30.0/255, 144.0/255, 255.0/255),    # dodgerblue
    "human_rgb_pick_place/task_11": (135.0/255, 206.0/255, 250.0/255),   # azzurro
    "human_rgb_pick_place/task_12": (139.0/255, 0.0/255, 0.0/255),       # rosso scuro
    "human_rgb_pick_place/task_13": (178.0/255, 34.0/255, 34.0/255),     # firebrick
    "human_rgb_pick_place/task_14": (255.0/255, 69.0/255, 0.0/255),      # orangered
    "human_rgb_pick_place/task_15": (255.0/255, 99.0/255, 71.0/255),     # tomato
}


def get_color_variations(base_color, n_variations, s_range=(0.5, 1.0), v_range=(0.7, 1.0)):
    """Generate variations of a base color in HSV space."""
    h, s, v = colorsys.rgb_to_hsv(*base_color)
    variations = []
    for i in range(n_variations):
        sat = s_range[0] + (s_range[1] - s_range[0]) * (i / max(n_variations - 1, 1))
        val = v_range[0] + (v_range[1] - v_range[0]) * (i / max(n_variations - 1, 1))
        variations.append(colorsys.hsv_to_rgb(h, sat, val))
    return variations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Debug mode') 
    parser.add_argument('--compute_embeddings', action='store_true')
    parser.add_argument('--compute_tsne', action='store_true')
    parser.add_argument('--all_dataset_plot', action='store_true')
    parser.add_argument('--split', type=str, default='val', help='Split to use for evaluation (train/val)')
    parser.add_argument('--model_path', type=str, default='/home/rsofnc000/checkpoint_save_folder/Video_Encoder/Video_Encoder_multi-Batch74', help='Path to the model file')
    parser.add_argument('--ckpt', type=int, default=99, help='Checkpoint number')
    parser.add_argument('--path_train_val_json', type=str, default='/home/rsofnc000/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/video_encoder/datasets_paths_absolute', help='Path to the train/val JSON file')
    
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    seed_everything()
    split = args.split
    
    if args.compute_embeddings:
        # 1. Load the model
        model_path = os.path.join(args.model_path, f'model_save-{args.ckpt}.pt')
        config_path = os.path.join(args.model_path, 'config.yaml')
        config = OmegaConf.load(config_path)
        model = hydra.utils.instantiate(config.policy)
        loaded = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(loaded)
        model = model.to(torch.device('cuda'))
        model.eval()
        print(f"\n---- Model loaded from {model_path} ----\n")
        
        # 2. Load the dataset
        
        config.dataset_cfg.mode = split
        dataset = hydra.utils.instantiate(config.get('dataset_cfg', None))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"Dataset length: {len(dataset)}")
        print(f"\n---- Dataset loaded ----\n")
        
        predicted_embeddings = OrderedDict()
        predicted_embeddings[split] = OrderedDict()
        
        for data_item in tqdm(data_loader):
            # 3. Get the data
            # print(data_item)
            # demo = data_item['demo_data']['demo'].to(torch.device('cuda'))
            # traj_path = data_item['traj_path'][0]
            # dataset_name = data_item['dataset_name']
            # print(f"Running inference on demo: {dataset_name} - {data_item['task_name']}")
            
    
            demo = data_item['demo_data']['demo'].to(torch.device('cuda'))
            print(f"Running inference on demo: {data_item['dataset_name']} - {data_item['task']}")
            traj_path = data_item['traj_path'][0]
            task_name = traj_path.split('/')[-3]
            variation_name = traj_path.split('/')[-2]
            traj_name = traj_path.split('/')[-1]
            
            if task_name not in predicted_embeddings[split]:
                predicted_embeddings[split][task_name] = OrderedDict()
            if variation_name not in predicted_embeddings[split][task_name]:
                predicted_embeddings[split][task_name][variation_name] = OrderedDict()
                predicted_embeddings[split][task_name][variation_name]['text_embedding'] = data_item['embedding_data']
            if traj_name not in predicted_embeddings[split][task_name][variation_name]:
                predicted_embeddings[split][task_name][variation_name][traj_name] = []
            
            with torch.no_grad():
                demo_embedding = model(demo)
            
            predicted_embeddings[split][task_name][variation_name][traj_name].append(demo_embedding.cpu().detach().numpy())
            
        
        # 4. Save the embeddings
        save_path = os.path.join(args.model_path, f'prediction_{split}_{args.ckpt}')
        os.makedirs(save_path, exist_ok=True)
        npz_dict = {}
        for task_name, variations in predicted_embeddings[split].items():
            for variation_name, trajs in variations.items():
                for traj_name, embeddings_list in trajs.items():
                    key = f"{task_name}/{variation_name}/{traj_name}"
                    npz_dict[key] = np.array(embeddings_list[0])

        # Save to NPZ file
        np.savez(os.path.join(save_path, 'predictions.npz'), **npz_dict)

    if args.all_dataset_plot:
        print("Plotting all datasets")
        # --- Compute t-SNE ---
        save_path = os.path.join(args.model_path, f'prediction_{split}_{args.ckpt}')
        npz_path = os.path.join(save_path, 'predictions.npz')

        loaded = np.load(npz_path)

        labels = []
        embeddings = []
        markers = []
        groups = []

        dataset_labels = []
        task_labels = []

        for key, embedding in loaded.items():
            embeddings.append(embedding.flatten())
            marker_type = "text" if "text_embedding" in key else "demo"
            markers.append(marker_type)
            group = "/".join(key.split('/')[:2])  # task_name/variation_name
            if 'panda_' in group:
                group = group.split('panda_')[1]
            groups.append(group)
            labels.append(key)
            dataset_labels.append(key.split('/')[0])
            task_labels.append(key.split('/')[1])

        # Remove duplicates
        unique_dataset_labels = sorted(set(dataset_labels))
        unique_task_labels = sorted(set(task_labels))
        
        embeddings = np.stack(embeddings)
        # cosine_dist = cosine_distances(embeddings)
        perplexity = 3
        tsne = TSNE(n_components=2,
                    metric='cosine',
                    init='random',
                    perplexity=perplexity,
                    random_state=42)
        tsne_result = tsne.fit_transform(embeddings)

        tsne_x = tsne_result[:, 0]
        tsne_y = tsne_result[:, 1]

        # Map tasks to their groups
        task_to_variations = defaultdict(list)
        for group in groups:
            task_name = group.split('/')[0]
            task_to_variations[task_name].append(group)

        # --- Assign specific base colors for known tasks ---

        # Manually set base RGB colors
        task_base_colors = {
            "pick_place": (0.0, 0.0, 1.0),       # Blue
            "nut_assembly": (1.0, 0.0, 0.0),     # Red
            "stack_block": (0.0, 1.0, 0.0),     # Green
            "button": (1.0, 1.0, 0),     # Yellow
        }

        # Group variations under tasks
        task_to_variations = defaultdict(set)
        for group in groups:
            task_name = group.split('/')[0]
            if 'panda_' in task_name:
                task_name = task_name.split('panda_')[1]
            task_to_variations[task_name].add(group)

        # Generate color variations
        group_to_color = {}
        for task, variation_list in task_to_variations.items():
            base_color = task_base_colors.get(task, (0.7, 0.7, 0.7))  # default gray if unknown task
            unique_variations = sorted(set(variation_list))
            color_variants = get_color_variations(base_color, len(unique_variations))
            for group, color in zip(unique_variations, color_variants):
                group_to_color[group] = color

        # --- Plot ---
        plt.figure(figsize=(10, 8))
        plotted_legend_labels = set()

        for x, y, marker_type, group in zip(tsne_x, tsne_y, markers, groups):
            if "human_rgb" in group:
                color = mivia_task_to_color.get(group) 
            else:
                color = group_to_color.get(group, (0.5, 0.5, 0.5))  # fallback gray
            if marker_type == "text":
                try:
                    variation_label = int(group.split('_')[-1])
                except Exception:
                    variation_label = "?"
                plt.text(x + 0.5, y, variation_label, fontsize=9, weight='bold', color=color)
                
                plt.scatter(x, y, c=[color], marker='*', s=150)
            else:
                if group not in plotted_legend_labels:
                    plt.scatter(x, y, c=[color], marker='o', s=60, label=group)
                    plotted_legend_labels.add(group)
                else:
                    plt.scatter(x, y, c=[color], marker='o', s=60)

        plt.title("Cosine Similarity between Predicted Embeddings")
        plt.legend(title="Task/Variation", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'tsne_plot_with_perplexity_{perplexity}.png'))
    