#!/bin/bash

DATASET_PATH=/user/frosa/multi_task_lfd/backup_datasets/
TASK_NAME=pick_place
OUT_PATH=/user/frosa/multi_task_lfd/datasets/

python optimize_dataset.py --dataset_path ${DATASET_PATH} --task_name ${TASK_NAME} --robot_name ur5e --out_path ${OUT_PATH} --real --debug

# python optimize_dataset.py --dataset_path ${DATASET_PATH} --task_name ${TASK_NAME} --robot_name panda --out_path ${OUT_PATH}
