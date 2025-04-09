<!-- Cond Module -->
## to: 1) generate paths and 2) embeddings from the commands of the finetuning dataset
* run `bashes/generate_train_val_paths_finetuning.sh`
## [train] cond module on the embeddings of the finetuning dataset
* dataset, batchsampler are defined in `Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets/command_encoder/command_encoder_dataset.py`
* for training, run the bash script `train_cond_module_finetuning` in `bashes/` 
# [testing] cond-module via T-SNE plot
* run bash script `test_cond_module_finetuning` -> plots of the embedding (and centroids) are saved in `bashes/finetuning_centroid_figures` folder

<!-- RT1 -->
## create (demo, traj) couples for rt1 training
* run `bashes/generate_traj_demo_couples_from_json.sh`
## [train] RT-1 video-conditioned on the finetuning dataset
* dataset, batchsampler are defined in `training/multi_task_il/datasets/rt1/finetuning_paired_dataset.py`
* to launch training, run bash script `train_RT1_video_cond_finetuning.sh` in `bashes/`

# [testing] RT-1
in order to test
* open `test_RT1_video_cond.sh` bash script
* select the root of your project with `PROJECT_NAME` var
* select the step(s)
* run with `nohup ./test_RT1_video_cond.sh > test_RT1_video_cond_module_freezed_output_epoch90.txt &`
* get the results in the same folder where the checkpoints are stored

# create video from test
* navigate to `multi-task-lfd-framework` folder (`multi-task-lfd-framework/utils/analysis`)
* change the project folder in the `--base_path` argument
* execute `create_video_from_test.sh` script

# visualize finetuning dataset
* run python script `training/multi_task_il/datasets/rt1/dataset_analysis/visualize_actions.py`

# visualization
<!-- ## visualize video + action for a single dataset
`visualize_action_single_dataset.py` -->
<!-- ## visualize video + action for all the datasets
`visualize_actions.py` -->
## visualize action (no video) for all the datasets
* run python script `training/multi_task_il/datasets/rt1/dataset_analysis/visualize_actions_static.py`

# datasets
## to convert original real and sim dataset to delta actions
`bashes/convert_real_sim_delta.sh`
## to find min and max for the actions components for every dataset
`bashes/find_min_max_actions.sh`
## to plot histogram for bin distribution given an interval
`bashes/plot_bin_histogram.sh`


### embedding cond module in RT1 input
`training/multi_task_il/models/rt1/diagnosys_script/plot_embedding_cond_module.py`

## come sono fatti batch sampler

# create plots from checkpoints (sim models)!!!
bashes/plot_results_checkpoint.sh
# create real dataset number of frame distribution (for variation and for phase)
bashes/real_dataset_frame_distribution.sh


# Subsampling
training/multi_task_il/datasets/optimize_dataset.sh
# visualize mean and std dev of number of frames for task
bashes/real_dataset_frame_distribution.sh


# Trajectories workflow
Optimize: ./optimize_dataset.sh
Convert: ./convert_real_sim_delta.sh
Generate paths: ./generate_train_val_paths_finetuning.sh
Generate couples (video + traj): ./generate_traj_demo_couples_from_json.sh
find min max: ./find_min_max_actions.sh
Plot histograms: ./plot_bin_histogram.sh

## heatmap
```bash
/raid/home/frosa_Loc/Multi-Task-LFD-Framework/utils/action_distribution.sh
```
# RT-1 and mosaic-CTOD grad-CAM
```bash
bashes/RT1_inference_demo.sh
```

## failure analysis



complete black list:

<!-- asu_table_top_converted_absolute_pose
berkeley_autolab_ur5_converted_absolute_pose
iamlab_cmu_pickup_insert_converted_absolute_pose
taco_play_converted_absolute_pose
droid_converted_absolute_pose
sim_ur5e_pick_place_shifted_converted_absolute
real_new_ur5e_pick_place_converted_absolute
panda_pick_place -->

  black_list: ['asu_table_top_converted_absolute_pose', 'berkeley_autolab_ur5_converted_absolute_pose', 'iamlab_cmu_pickup_insert_converted_absolute_pose','taco_play_converted_absolute_pose','droid_converted_absolute_pose','sim_ur5e_pick_place_shifted_converted_absolute', 'real_new_ur5e_pick_place_converted_absolute','sim_panda_pick_place_converted_absolute', 'panda_pick_place']

  BLACK_LIST = ['asu_table_top_converted_absolute_pose', 'berkeley_autolab_ur5_converted_absolute_pose', 'iamlab_cmu_pickup_insert_converted_absolute_pose','taco_play_converted_absolute_pose','droid_converted_absolute_pose','sim_ur5e_pick_place_shifted_converted_absolute', 'real_new_ur5e_pick_place_converted_absolute', 'sim_panda_pick_place_converted_absolute', 'panda_pick_place']

<!-- 


  BLACK_LIST=asu_table_top_converted_absolute_pose,berkeley_autolab_ur5_converted_absolute_pose,sim_ur5e_pick_place_shifted_converted_absolute, real_new_ur5e_pick_place_converted_absolute,panda_pick_place -->

  <!-- black_list: ['asu_table_top_converted_absolute_pose', 'berkeley_autolab_ur5_converted_absolute_pose', 'sim_ur5e_pick_place_shifted_converted_absolute', 'panda_pick_place'] -->

<!-- # TEMP CHANGES
line 1569 and 1673 of command_encoder/utils -->

