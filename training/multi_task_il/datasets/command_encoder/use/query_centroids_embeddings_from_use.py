import argparse
import json
from multi_task_il.models.command_encoder.muse.muse import get_model
import pickle
import os

command_dict = {
    'task_00': 'Pick the green box and place it into the first bin',
    'task_01': 'Pick the green box and place it into the second bin',
    'task_02': 'Pick the green box and place it into the third bin',
    'task_03': 'Pick the green box and place it into the fourth bin',
    'task_04': 'Pick the yellow box and place it into the first bin',
    'task_05': 'Pick the yellow box and place it into the second bin',
    'task_06': 'Pick the yellow box and place it into the third bin',
    'task_07': 'Pick the yellow box and place it into the fourth bin',
    'task_08': 'Pick the blue box and place it into the first bin',
    'task_09': 'Pick the blue box and place it into the second bin',
    'task_10': 'Pick the blue box and place it into the third bin',
    'task_11': 'Pick the blue box and place it into the fourth bin',
    'task_12': 'Pick the red box and place it into the first bin',
    'task_13': 'Pick the red box and place it into the second bin',
    'task_14': 'Pick the red box and place it into the third bin',
    'task_15': 'Pick the red box and place it into the fourth bin',
}

def create_emb_and_save_pickle(traj_path, model_torch, tokenizer, output_folder, dataset_name):
    ''' query USE to produce a 512 embedding from the command
        associated to the task at traj_path
    '''
    print(f"Computing embedding for {traj_path}")
    with open(traj_path, "rb") as f:
        traj_data = pickle.load(f)
    
    command = traj_data.get('command')
    if command is None and "task_" in traj_path: # it is one of our datasets
        command = command_dict[traj_path.split("/")[-2]]
        
    print(f"Generating embedding for command: {command}")
    command_emb = model_torch(tokenizer(command)).detach().numpy()
    
    index = traj_path.split("/").index(dataset_name)
    save_path_command_emb = os.path.join(output_folder, *traj_path.split("/")[index:-1])

    if not os.path.exists(save_path_command_emb):
        os.makedirs(save_path_command_emb)

    save_path_command_emb = os.path.join(save_path_command_emb, 'task_embedding.pkl')

    pickle.dump(command_emb, open(save_path_command_emb, 'wb'))
    print(f"saved embedding at {save_path_command_emb}")
    
    return save_path_command_emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_json', default='./all_pkl_paths.json')
    parser.add_argument("--path_to_tokenizer")
    parser.add_argument("--path_to_muse")
    parser.add_argument("--path_to_dataset")
    parser.add_argument("--output_json_folder")
    parser.add_argument("--embedding_save_folder")
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    with open(args.task_json, 'r') as file:
        data = json.load(file)
    
    print(f"Getting Universal Sentence Encoder and Tokenizer...")
    model_torch, tokenizer = get_model(args.path_to_muse, args.path_to_tokenizer)
    
    black_list = []
    
    embeddings_data = {}
    for dataset_name in data.keys():
        if dataset_name not in black_list:
            embeddings_data[dataset_name] = {}
            for task in data[dataset_name].keys():
                if type(data[dataset_name][task]) == list:
                    embeddings_data[dataset_name][task] = []
                    # USE for the embedding and save into embeddings_data
                    istances_list = data[dataset_name][task]
                    traj_path = istances_list[0] # take only one traj, the command string is the same for all elements in the folder
                    save_path_command_emb = create_emb_and_save_pickle(traj_path, model_torch, tokenizer, args.embedding_save_folder, dataset_name)
                    embeddings_data[dataset_name][task].append(save_path_command_emb)
                    
                elif type(data[dataset_name][task]) == dict:
                    embeddings_data[dataset_name][task] = {}
                    for subtask in data[dataset_name][task].keys():
                        assert type(data[dataset_name][task][subtask]), f'error, data is of type {type(data[dataset_name][task][subtask])}'
                        embeddings_data[dataset_name][task][subtask] = []
                        istances_list = data[dataset_name][task][subtask]
                        traj_path = istances_list[0] # take only one traj
                        save_path_command_emb = create_emb_and_save_pickle(traj_path, model_torch, tokenizer, args.embedding_save_folder, dataset_name)
                        embeddings_data[dataset_name][task][subtask].append(save_path_command_emb)

    # saving the embeddings paths to a json file
    save_path = os.path.join(args.output_json_folder, 'embeddings_data.json')
    if not os.path.exists(args.output_json_folder):
        os.makedirs(args.output_json_folder)
        
    with open(save_path, "w") as outfile: 
        json.dump(embeddings_data, outfile, indent=2) 
    