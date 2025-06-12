# VRT1
## Before starting
1. Generate training and validation json, for all datasets (this is for video encoder). Generate embeddings json.
```bash
# from scripts directory
source generate_train_val_paths_json.sh
```
2. Generate training and validation couples json.
```bash
# from scripts directory
source generate_train_val_couples_json.sh
```

```bash
PATH_TO_PT_MODEL="../../training/multi_task_il/models/command_encoder/muse/models/model.pt"
PATH_TO_TF_MODEL="../../training/multi_task_il/models/command_encoder/muse/models/universal-sentence-encoder-multilingual-large-3"

PATH_TO_PANDA_SIMULATED_DATASET="/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/panda_pick_place"
PATH_TO_UR5E_SIMULATED_DATASET="/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/ur5e_pick_place/real_new_ur5e_pick_place_delta_action"

PATH_TO_UR5E_REAL_DELTA_SUBSAMPLED_DATASET="/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/real_new_ur5e_pick_place_delta_action
"
```

## Video Encoder
For training **video encoder**, run:
```bash
# from video encoder folder
source train_video_encoder.sh
```

---

For testing **video encoder**, run:
```bash
# from video encoder folder
source test_video_encoder.sh
```
The scripts generates a TSNE plot by computing the embeddings of all the samples in the validation dataset and comparing them with the ground truth ones.

## Policy
For **training** **VRT1 policy**, run:
```bash
# from vrt1 folder
source train_vrt1.sh
```
