# Multi-Task-LDF-Training-Framework

# Main Folders

* **tasks**: It contains all the code related to the robosuite simulation environment
* **training**: It contains all the code related to the training 
    * **experiment**: It contains the .yaml files that describe the test
    * **train_scripts**: It contains the .py files for running the training
* **test**: It contains all the code used for the simulation

# How the training works
*) *tasks_spec*, this parameter indicates the which sub-task use for each task, and how many trajectories for each sub-task

*) *train_utils.py*: 
    *) make_data_loaders: It creates the dataloaders for train and validation. 

*) *MultiTaskPairedDataset*: 