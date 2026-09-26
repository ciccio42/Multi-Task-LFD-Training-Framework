import json
from multi_task_il.models.command_encoder.muse.muse import get_model
import pickle
import os


def create_emb_and_save_pickle(text_command, save_path, model_torch, tokenize):
    ''' query USE to produce a 512 embedding from the command
        associated to the task at traj_path
    '''
    
    command_emb = model_torch(tokenize(text_command)).detach().numpy()
    save_path_command_emb = os.path.join(save_path, 'task_embedding.pkl')
    pickle.dump(command_emb, open(save_path_command_emb, 'wb'))
    print(f"saved embedding at {save_path_command_emb}")
    return save_path_command_emb


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_json', default='./all_pkl_paths.json')
    parser.add_argument("--debug", default=False, help="whether or not attach the debugger")
    parser.add_argument("--path_to_tokenizer", default='')
    parser.add_argument("--path_to_muse", default='')
    parser.add_argument("--path_to_dataset", default='')
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    with open(args.task_json, 'r') as file:
        data = json.load(file)
    
    print(f"getting USE and tokenizer...")
    model_torch, tokenize = get_model(args.path_to_muse, args.path_to_tokenizer)
    print(f"done.")
    
    #----------------------- TODO -------------------------------------------
    #  1)usare task_count_info.json e fornire questa all'USE
    #  2)una volta prodotto l'embedding, combinarlo con il gruppo di comandi
    #  3)OPPURE creare un file separato in cui c'è corrispondenza con i task
    #------------------------------------------------------------------------
    
    dataset_name = args.task_json.split('/')[-1].split('.')[0].split('_')[0]
    print(f"Elaborating commands from: {dataset_name}")
    
    embeddings_data = {}
    
    if dataset_name not in embeddings_data.keys():
        embeddings_data[dataset_name] = {}
    
    
    for task_name in data.keys():
        if task_name not in embeddings_data[dataset_name].keys():
            embeddings_data[dataset_name][task_name] = {}
        
        for variation in data[task_name].keys():
            if variation not in embeddings_data[dataset_name][task_name].keys():
                embeddings_data[dataset_name][task_name][variation] = []
            
            command = data[task_name][variation]
            save_folder = os.path.join(args.path_to_dataset, "command_text_embeddings", dataset_name, task_name, variation)
            print(f"\tSave Folder: {save_folder}")
            os.makedirs(save_folder, exist_ok=True)
            
            
            print(f"\tcommand: {command}")
            save_path_command_emb = create_emb_and_save_pickle(
                                       text_command=command,
                                       save_path=save_folder,
                                       model_torch=model_torch,
                                       tokenize=tokenize)
            embeddings_data[dataset_name][task_name][variation].append(save_path_command_emb)
               
                
    #----------------------- TODO ---------------------------
    #  visualizzare nello spazio gli embedding
    #-------------------------------------------------------             
            
    with open("embeddings_data.json", "w") as outfile: 
        json.dump(embeddings_data,outfile,indent=2) 
    
    # if args.asu:
    #     pass
    
    