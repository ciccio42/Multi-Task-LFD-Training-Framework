
#!/bin/bash


DATASET_PATH='/user/frosa/multi_task_lfd/datasets/real_new_ur5e_pick_place_converted_absolute'

python -u ../training/multi_task_il/datasets/real_dataset_frame_distribution.py \
            --dataset_path ${DATASET_PATH}

