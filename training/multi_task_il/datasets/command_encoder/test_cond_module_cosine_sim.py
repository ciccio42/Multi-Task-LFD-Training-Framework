
from multi_task_il.datasets.command_encoder.command_encoder_dataset import CommandEncoderFinetuningDataset, FinetuningCommandEncoderSampler
from multi_task_il.datasets.command_encoder.cond_module import CondModule
from multi_task_il.datasets.utils import collate_by_task
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2
import os
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import seaborn as sns
from multi_task_il.datasets.command_encoder.multi_task_command_encoder import CommandEncoderSampler, CosineLossCalculator
from tqdm import tqdm
import pickle as pkl

DATA_AUGS = {
            "old_aug": False,
            "brightness": [0.9, 1.1],
            "contrast": [0.9, 1.1],
            "saturation": [0.9, 1.1],
            "hue": [0.0, 0.0],
            "p": 0.5,
            "horizontal_flip_p": 0.1,
            "brightness_strong": [0.875, 1.125],
            "contrast_strong": [0.5, 1.5],
            "saturation_strong": [0.5, 1.5],
            "hue_strong": [-0.05, 0.05],
            "p_strong": 0.5,
            "horizontal_flip_p_strong": 0.5,
            "null_bb": False,
        }

dataset_samples_spec = {
    "asu_table_top_converted": {
        "name": "asu_table_top_converted",
        "crop": [0, 35, 0, 0],
        "image_channel_format": "RGB",
    },
    "berkeley_autolab_ur5_converted": {
        "name": "berkeley_autolab_ur5_converted",
        "crop": [0, 0, 0, 0],
        "image_channel_format": "RGB",
    },
    "iamlab_cmu_pickup_insert_converted": {
        "name": "iamlab_cmu_pickup_insert_converted",
        "crop": [0, 0, 0, 0],
        "image_channel_format": "RGB",
    },
    "taco_play_converted": {
        "name": "taco_play_converted",
        "crop": [0, 0, 0, 0],
        "image_channel_format": "RGB",
    },
    "droid_converted": {
        "name": "droid_converted",
        "crop": [0, 0, 0, 0],
        "image_channel_format": "RGB",
    },
    "sim_new_ur5e_pick_place_converted": {
        "name": "sim_new_ur5e_pick_place_converted",
        "crop": [20, 25, 80, 75],
        "image_channel_format": "RGB",
    },
    "real_new_ur5e_pick_place_converted": {
        "name": "real_new_ur5e_pick_place_converted",
        "n_tasks": 16,
        "crop": [20, 25, 80, 75], ############################
        "image_channel_format": "BGR",
    },
    "panda_pick_place": {
        "name": "panda_pick_place",
        "crop": [20, 25, 80, 75],
        "image_channel_format": "RGB",
    },
}

def create_val_loader(tasks_spec, black_list, data_augs):
    val_dataset = CommandEncoderFinetuningDataset(mode='val',
                                                tasks_spec=tasks_spec,
                                                dataset_samples_spec=dataset_samples_spec,
                                                jsons_folder='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes',
                                                black_list=black_list,
                                                data_augs=DATA_AUGS)

    samplerClass = FinetuningCommandEncoderSampler
    val_sampler = samplerClass(val_dataset,
                                shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=20,
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return val_loader

def create_train_loader(tasks_spec, black_list, data_augs):
    train_dataset = CommandEncoderFinetuningDataset(mode='train',
                                                tasks_spec=tasks_spec,
                                                dataset_samples_spec=dataset_samples_spec,
                                                jsons_folder='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes',
                                                black_list=black_list,
                                                data_augs=DATA_AUGS)

    samplerClass = FinetuningCommandEncoderSampler
    train_sampler = samplerClass(train_dataset,
                                shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=20,
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader

def list_of_strings(arg):
    return arg.split(',')

def make_centroids(embedding_dict):
    centroids_per_task = {}
    for task in embedding_dict.keys():
        for idx in range(len(embedding_dict[task])):
            if idx == 0:
                subtask_tensor = torch.from_numpy(embedding_dict[task][idx]).unsqueeze(0)
            else:
                subtask_tensor = torch.cat((subtask_tensor, torch.from_numpy(embedding_dict[task][idx]).unsqueeze(0)))
            
        subtask_tensor = torch.mean(subtask_tensor, 0)
        centroids_per_task[task] = deepcopy(subtask_tensor)
            
    return centroids_per_task



            
            
    # task_bsize = 160
    # target = torch.ones(160).to(device)
    # cosine_loss_calculator = CosineLossCalculator(task_bsize, target, device)
    # # loss = cosine_loss_calculator.compute_cosine_similarity(out, model_inputs['embedding_data'])
    # cosine_loss_calculator.compute_cosine_similarity(embeddings_tensor.to(device), se_embeddings.repeat_interleave(10, dim=0).to(device))
     
    # from torch.nn import CosineEmbeddingLoss
    # target = torch.ones(160).to(device)
    # cosine_loss = CosineEmbeddingLoss(reduction='none')
    # cosine_loss(embeddings_tensor.to(device), se_embeddings.repeat_interleave(10, dim=0).to(device), target)
    
    
    # from torch.nn import CosineSimilarity
    
    # cos = CosineSimilarity(dim=1, eps=1e-6)
    # output = cos(embeddings_tensor.to(device), se_embeddings.repeat_interleave(10, dim=0).to(device))
    # loss = cosine_loss_calculator.compute_cosine_similarity(torch.from_numpy(embedding_dict['Pick the green box and place it into the first bin'][0]).unsqueeze(0).to(device), se_embeddings[0].unsqueeze(0).to(device))
    
            
    # for k in range(15):
    #     for j in range(15):
    #         print(cos(se_embeddings[k].unsqueeze(0), se_embeddings[j].unsqueeze(0)))
            
            
    # a = se_embeddings[0:2]
    # b = se_embeddings[0:2]
    
    # c = se_embeddings[1:3]
    
    # mse = torch.nn.MSELoss(reduction='none')
    
    # mse(a,b)
    # mse(b,c)
    
    # a = torch.tensor([[3.,3.,3.], [3.,3.,3.]])
    # b = torch.tensor([[7.,7.,7.], [8.,8.,8.]])
    
    # mse = torch.nn.MSELoss(reduction='none')
    # torch.mean(torch.sqrt(mse(a,b).sum(dim=1)))
            
    # [(torch.from_numpy(i) - se_embeddings[0]).sum() for i in embedding_dict['Pick the green box and place it into the first bin']]          

def get_palette(num_classes):
    import colorcet as cc
    palette = sns.color_palette(cc.glasbey, n_colors=num_classes)
    palette[0] = (0.0, 0.37, 0.0)
    palette[1] = (0.0, 0.5, 0.0)
    palette[2] = (0.0, 0.75, 0.0)
    palette[3] = (0.0, 1.0, 0.0)
    
    palette[4] = (0.37, 0.37, 0.0)
    palette[5] = (0.5, 0.5, 0.0)
    palette[6] = (0.75, 0.75, 0.0)
    palette[7] = (1.0, 1.0, 0.0)
    
    palette[8] = (0.0, 0.0, 0.37)
    palette[9] = (0.0, 0.0, 0.5)
    palette[10] = (0.0, 0.0, 0.75)
    palette[11] = (0.0, 0.0, 1.0)
    
    palette[12] = (0.37, 0.0, 0.0)
    palette[13] = (0.5, 0.0, 0.0)
    palette[14] = (0.75, 0.0, 0.0)
    palette[15] = (1.0, 0.0, 0.0)


def create_embedding_plot(train_preds, val_preds, test_preds, se_embeddings, y):

    train_preds_np, val_preds_np, test_preds_np, se_embeddings_np = \
        train_preds.cpu().numpy(), val_preds.cpu().numpy(), test_preds.cpu().numpy(), se_embeddings.cpu().numpy()

    feat_cols = [ 'e'+str(i) for i in range(train_preds_np.shape[1]) ]
    
    df_train, df_val, df_test, df_se = \
        pd.DataFrame(train_preds_np,columns=feat_cols), \
            pd.DataFrame(val_preds_np,columns=feat_cols), \
                pd.DataFrame(test_preds_np,columns=feat_cols), \
                    pd.DataFrame(se_embeddings_np,columns=feat_cols)
            
    df_train['y'] = y * 90
    df_val['y'] = y * 10
    df_test['y'] = y * 10
    df_se['y'] = y
    
    #----------create TSNE object
    num_classes = 16
    all_tensor = torch.cat((train_preds, val_preds, test_preds, se_embeddings)).cpu().numpy()
    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500) # vedere se cambiare parametri
    tsne_results = tsne.fit_transform(all_tensor)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    #----------add columns to df
    df_train['tsne-2d-one'] = tsne_results[:90*num_classes,0]
    df_train['tsne-2d-two'] = tsne_results[:90*num_classes,1]
    
    df_val['tsne-2d-one'] = tsne_results[90*num_classes:100*num_classes,0]
    df_val['tsne-2d-two'] = tsne_results[90*num_classes:100*num_classes,1]
    
    df_test['tsne-2d-one'] = tsne_results[100*num_classes:110*num_classes, 0]
    df_test['tsne-2d-two'] = tsne_results[100*num_classes:110*num_classes, 1]
    
    df_se['tsne-2d-one'] = tsne_results[110*num_classes:, 0]
    df_se['tsne-2d-two'] = tsne_results[110*num_classes:, 1]

    #----------plotting
    palette = get_palette(num_classes)

    plt.figure(figsize=(15,10))
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y", # per ora non la uso visto che ogni campione è a se
        palette=palette,
        data=df_train,
        legend="full",
        # alpha=0.3
    )
    
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=palette,
        data=df_val,
        marker="*",
        legend=False,
        ax=ax
    )
    
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=palette,
        data=df_test,
        marker="X",
        legend=False,
        ax=ax
    )
    
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=palette,
        data=df_se,
        marker="s",
        legend=False,
        ax=ax
    )
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    from datetime import datetime
    ts = datetime.now().strftime("%m-%d_%H:%M")
    ## save clusters plot
    try:
        plt.savefig(f"finetuning_centroid_figures/embeddings_clusters_{ts}.png")
    except Exception:
        os.mkdir("finetuning_centroid_figures/")
        plt.savefig(f"finetuning_centroid_figures/embeddings_clusters_{ts}.png")
            
def init_cond_module():
    ## loading model
    # cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True).to(device)
    cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True)
    weights = torch.load('/user/frosa/multi_task_lfd/checkpoint_save_folder/cond_module_ALLBUTDROID_20epochs_RGB_weak_aug-Batch32/model_save-1012.pt', weights_only=True)

    cond_module.load_state_dict(weights)
    cond_module.eval()

    model_parameters = filter(lambda p: p.requires_grad, cond_module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print(cond_module)
    print('Total params in cond module before freezing:', params)

    # freeze cond module
    for p in cond_module.parameters():
        p.requires_grad = False
        
    model_parameters = filter(lambda p: p.requires_grad, cond_module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print(cond_module)
    print('Total params in cond module after freezing:', params)
    
    return cond_module 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default='')
    parser.add_argument('--cuda_device', default=1)
    parser.add_argument("--black_list", type=list_of_strings, default=['', ''], help="datasets to exclude")
    parser.add_argument("--debug", default=False, help="whether or not attach the debugger")
    
    args = parser.parse_args()
    
    if args.debug == 'True' or args.debug == 'true':
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # cuda device
    print(f"current device: {torch.cuda.current_device()}")
    if args.cuda_device != 0:
        print(f"switching to {args.cuda_device}...")
        torch.cuda.set_device(int(args.cuda_device))
        print(f"current device: {torch.cuda.current_device()}")
    
    tasks_spec = [
        {
            "name": "pick_place",
            "n_tasks": 16,
            "crop": [20, 25, 80, 75],
            "n_per_task": 2,
            "task_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "skip_ids": [],
            "loss_mul": 1,
            "task_per_batch": 16,
            "traj_per_subtask": 100,
            "demo_per_subtask": 100,
        }
    ]

    train_loader = create_train_loader(tasks_spec, args.black_list, DATA_AUGS)
    val_loader = create_val_loader(tasks_spec, args.black_list, DATA_AUGS)
    
    cond_module = init_cond_module()

    embedding_dict = {} # store embeddings for each task
    # batch_count = 0
    with torch.no_grad():
        
        ## train preds
        for batch_idx, inputs in tqdm(enumerate(train_loader)):
            
            demos_sorted = inputs['finetuning']['demo_data']['demo'].to(device)
            if batch_idx == 0:
                train_preds = cond_module(demos_sorted)
                se_embeddings = inputs['finetuning']['embedding_data']
                y = inputs['finetuning']['sentence']
            else:
                train_preds = torch.cat((train_preds, cond_module(demos_sorted)))
            
        ## val preds
        for batch_idx, inputs in tqdm(enumerate(val_loader)):
            
            demos_sorted = inputs['finetuning']['demo_data']['demo'].to(device)
            if batch_idx == 0:
                val_preds = cond_module(demos_sorted)
            else:
                val_preds = torch.cat((val_preds, cond_module(demos_sorted)))
            
        ## test preds
        test_contexts_path = '/user/frosa/multi_task_lfd/checkpoint_save_folder/rt1_sim_abs_aa_weakaug_-1_1-Batch48/results_pick_place/run_1/step-16200_nocorr'
        test_demos = [f'context{i}.pkl' for i in range(160)]
        test_demos = [f'{test_contexts_path}/{i}' for i in test_demos]
        
        for idx, test_demo_path in tqdm(enumerate(test_demos)):
            with open(test_demo_path, "rb") as f:
                test_demo = pkl.load(f).to(device)
            
            if idx == 0:    
                test_preds = cond_module(test_demo)
            else:
                test_preds = torch.cat((test_preds, cond_module(test_demo)))
        
        print('end')
    
    
    # centroids_per_task = make_centroids(embedding_dict)

    
    create_embedding_plot(train_preds, val_preds, test_preds, se_embeddings.to(device), y)
    
    