import os
import wandb
from os.path import join
import torch
import copy
from omegaconf import OmegaConf
import torch.distributed as dist
import hydra
from multi_task_il.utils.lr_scheduler import build_scheduler, BaseScheduler
from multi_task_il.datasets.command_encoder.multi_task_command_encoder import CosineLossCalculator
from multi_task_il.datasets.command_encoder.sampler import FinetuningCommandEncoderSampler
import time
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.multiprocessing import cpu_count
import torch.nn as nn
from tqdm import tqdm

def collate_by_task(batch):
    """ Use this for validation: groups data by task names to compute per-task losses """
    collate_time = time.time()
    per_task_data = defaultdict(list)
    start_batch = time.time()
    for b in batch:
        per_task_data[b['task_name']].append(
            {k: v for k, v in b.items() if k != 'task_name' and k != 'task_id'}
        )

    collate_time = time.time()
    for name, data in per_task_data.items():
        per_task_data[name] = default_collate(data)
    return per_task_data

def make_model(config, local_rank):
   
    resume = config.get('resume', False)
    finetune = config.get('finetune', False)
    
    
    model = hydra.utils.instantiate(config.policy)
    model = model.to(torch.device("cuda:" + str(local_rank)))

    print("Model initialized to: {}".format(config.policy._target_))
    if resume or finetune:
        rpath = join(config.save_path, config.resume_path,
                            f"model_save-{config.resume_step}.pt")
        assert os.path.exists(rpath), "Can't seem to find {} anywhere".format(
            rpath)
        print('Finetuning model: load model from ...%s' %
                rpath)
        state_dict = torch.load(
            rpath, map_location=torch.device("cuda:" + str(local_rank)))
        if finetune:
            state_dict_keys = list(state_dict.keys())
            for key in state_dict_keys:
                if '_object_detector' in key or '_cond_backbone' in key or '_agent_backbone' in key:
                    state_dict.pop(key)
        model.load_state_dict(state_dict,strict=False)
        optimizer_state_dict = None
        if resume:
            try:
                # create path for loading state dict
                optimizer_state_dict = join(
                    config.save_path, config.resume_path, f"model_save-optim.pt")
                optimizer_state_dict = torch.load(
                    optimizer_state_dict, map_location=torch.device("cuda:" + str(local_rank)))
            except:
                print("Exception during loading optimizer state dict")
                optimizer_state_dict = None
    else:
        optimizer_state_dict = None
        
    return model, optimizer_state_dict


def make_optimizer_schedule( optimizer, optim_weights, optimizer_state_dict, config):
    
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            optim_weights,
            config.lr,
            weight_decay=config.get('weight_decay', 0))
    elif optimizer == 'RMSProp':
        optimizer = torch.optim.RMSprop(
            optim_weights,
            config.lr,
            weight_decay=config.get('weight_decay', 0))
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            optim_weights,
            config.lr,
            weight_decay=config.weight_decay)
        

    
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    print(
        f"Creating {optimizer}, with lr {optimizer.param_groups[0]['lr']}")

    lr_schedule = dict()
    if config.lr_schedule == 'None':
        lr_schedule['type'] = None
    else:
        lr_schedule['type'] = config.lr_schedule
    print(f"Lr-scheduler {config.lr_schedule}")
    
    return optimizer, build_scheduler(optimizer, lr_schedule)


def make_loss_function(config):
    
    print(f"Creating loss function")
    loss_function = CosineLossCalculator(batch_size=config.bsize)    

    return loss_function

def make_data_loaders(config, dataset_cfg, num_replicas: int = 1, global_rank: int = 1):
    dataset_cfg.mode = 'train'
    dataset = hydra.utils.instantiate(dataset_cfg)

    train_step = int(config.get('epochs') *
                     int(len(dataset)/(num_replicas*config.get('bsize'))))
    epoch_step = int(len(dataset)/(num_replicas*config.get('bsize')))

    print(f"Mode {dataset_cfg.mode} Train step {train_step}, epoch step {epoch_step}")
    train_sampler = FinetuningCommandEncoderSampler(dataset=dataset, 
                                                    batch_size=config.get('bsize'),
                                                    n_sampler_per_task=config.set_same_n,
                                                    epoch_step=epoch_step,
                                                    shuffle=True)
    
    
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=config.get('loader_workers', cpu_count()),
        collate_fn=collate_by_task,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    #\if dataset_cfg.split[1] > 0.0:
    dataset_cfg.mode = 'val'
    val_dataset = hydra.utils.instantiate(dataset_cfg)
    config.samplers.batch_size = config.train_cfg.val_size
    val_step = int(config.get('epochs') *
                    int(len(val_dataset)/config.get('vsize')))
    val_epoch_step = int(len(val_dataset)/config.get('vsize'))
    val_sampler = FinetuningCommandEncoderSampler(val_dataset, 
                                                    batch_size=config.get('bsize'),
                                                n_sampler_per_task=config.set_same_n,
                                                epoch_step=val_epoch_step,
                                                shuffle=True)
    val_loader = DataLoader(
        dataset,
        batch_sampler=val_sampler,
        num_workers=config.get('loader_workers', cpu_count()),
        collate_fn=collate_by_task,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
        
        
        
    return train_loader, val_loader

class Trainer:
    
    def __init__(self, allow_val_grad=False, hydra_cfg=None):
        assert hydra_cfg is not None, "Need to start with hydra-enabled yaml file!"
        
        self.config = hydra_cfg
        self.train_cfg = hydra_cfg.train_cfg
        # initialize device
        self._device_list = None
        self._device_list = self.device_list()
        print(f"List of devices {self._device_list}") 
        
        self._allow_val_grad = allow_val_grad
        self._world_size = hydra_cfg.get('num_gpus', 1) * hydra_cfg.get('num_nodes', 1)

        self.task_names = [task['name'] for task in self.config.tasks]
    
        # set of file saving
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)

        assert self.config.exp_name != -1, 'Specify an experiment name for log data!'
        self._best_validation_loss = float('inf')
        self._best_validation_weights = None

        append = "-Batch{}".format(int(self.config.bsize))
        
        self.config.exp_name += append

        save_dir = join(self.config.get('save_path', './'),
                        str(self.config.exp_name))
        save_dir = os.path.expanduser(save_dir)
        self._save_fname = join(save_dir, 'model_save')
        self.save_dir = save_dir
        print(f"Saving dir {self.save_dir}")
        self._step = None

    @property
    def device_count(self):
        return torch.cuda.device_count()

    def device_list(self):
        if self._device_list is None:
            dev_list = []
            for i in range(torch.cuda.device_count()):
                print(f"Adding device {i}")
                dev_list.append(torch.device("cuda:{}".format(i)))
            return dev_list
        return copy.deepcopy(self._device_list)
    
    
    def worker(self, local_rank, *args):
        
        global_rank = args[0] * args[1] + local_rank 
        dist.init_process_group( 
        backend='nccl',  
        world_size=self._world_size, 
        rank=global_rank 
        )
        
        if global_rank == 0:
                        
            if self.config.wandb_log:
                config_keys = ['train_cfg', 'tasks', 'samplers', 'dataset_cfg', 'policy']
                # for k in config_keys:
                #     print(k, self.config.get(k))
                #     print(k, dict(self.config.get(k)))
                #     print('-'*20)
                wandb_config = {k: self.config.get(k) for k in config_keys}
                wandb.login(key='227ed2fded06f63748a7a29dae55acdda7d131ff', relogin=True)
                print(f"Exp name: {self.config.exp_name}")
                self.config.project_name = self.config.exp_name.split('-Batch')[0]
                run = wandb.init(project=self.config.project_name,
                                name=self.config.exp_name,
                                sync_tensorboard=False)
        
        self.train(num_replicas=self._world_size,
                   global_rank=global_rank,
                   local_rank=local_rank)

    def train_loop(self, train_loader, scheduler, loss_function, global_rank, local_rank, model, optimizer, raw_stats: dict = dict(), epoch: int = 0, log_freq: int = -1, print_freq: int = -1, frac: float = 0.0):
        #### ---- Train loop ----####
        model = model.train()
    
        train_step = len(train_loader)/ self.config.get('bsize')
        print(f"Training for {train_step} steps")
        epoch_steps = 0
        for inputs in tqdm(train_loader):
            torch.cuda.empty_cache()
            
            model_inputs = inputs['finetuning']['demo_data']['demo'].to(self._device_list[local_rank])
            generated_embedding = model(input=model_inputs)
            gt_embedding = inputs['finetuning']['embedding_data'].to(self._device_list[local_rank])
            task_losses = loss_function.compute_cosine_similarity(generated_embedding, 
                                                                  gt_embedding)
            
            task_losses.backward()
            optimizer.step()
            
            
            # log stats
            # calculate train iter stats
            if global_rank == 0:
                tolog = dict()
                if self._step % log_freq == 0:
                    print(f"training epoch {epoch} - Logging step {self._step} - Loss value {task_losses.item()}")
                
                if self.config.wandb_log:
                    tolog['train/loss'] = task_losses.item()
                    tolog['train/epoch'] = epoch
                    tolog['train/step'] = self._step
                    wandb.log(tolog)
                    
            # update step
            epoch_steps += 1
            self._step += 1
    
    
    def val_loop(self, val_loader, scheduler, loss_function, global_rank, local_rank, model, optimizer, task_names, raw_stats: dict = dict(), epoch: int = 0,):
        
        validate = True
        tolog = dict()
        model = model.eval()
        
        accumulated_loss = 0.0
        for i, val_inputs in tqdm(enumerate(val_loader), total=len(val_loader)):
            
            torch.cuda.empty_cache()
            with torch.no_grad():
                model_inputs = val_inputs['finetuning']['demo_data']['demo'].to(self._device_list[local_rank])
                generated_embedding = model(input=model_inputs)
                gt_embedding = val_inputs['finetuning']['embedding_data'].to(self._device_list[local_rank])
                task_losses = loss_function.compute_cosine_similarity(generated_embedding, 
                                                                      gt_embedding)
                
                accumulated_loss += task_losses.item()
                
        # log stats
        accumulated_loss /= len(val_loader)
        print(f"Validation epoch {epoch} - Logging step {self._step} - Loss value {accumulated_loss}")
        
        if global_rank == 0 and self.config.wandb_log:
            tolog['val/loss'] = accumulated_loss
            tolog['val/epoch'] = epoch
            tolog['val/step'] = self._step
            wandb.log(tolog)
            
        if accumulated_loss < self._best_validation_loss:
            # save 
            self._best_validation_loss = accumulated_loss
            if global_rank == 0:
                self.save_checkpoint(model, optimizer, None, None)
                    
        
       
    def save_checkpoint(self, model, optimizer, weights_fn=None, save_fn=None, save_name=None):
        

        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        if save_name is not None:
            torch.save(model_to_save.state_dict(),
                    self._save_fname +'-{}-{}.pt'.format(save_name,self._epoch))
        else:
            torch.save(model_to_save.state_dict(),
                    self._save_fname + '-{}.pt'.format(self._epoch))
            
        if self.config.get('save_optim', False):
            torch.save(optimizer.state_dict(), self._save_fname +
                       '-optim.pt')
        
        print(f'Model checkpoint saved at epoch {self._epoch}')
        return 
    
    def train(self, weights_fn=None, save_fn=None, optim_weights=None, num_replicas: int = 1, global_rank: int = 0, local_rank: int = 0):
        model, optimizer_state_dict = make_model(self.config, local_rank)
        
        optim_weights = optim_weights if optim_weights is not None else model.parameters()
        optimizer, lr_scheduler = make_optimizer_schedule(self.config.train_cfg.optimizer,
                                                          optim_weights,
                                                          optimizer_state_dict,
                                                          self.config.train_cfg)

        
        loss_function = make_loss_function(self.config) 
        
        
        train_loader, val_loader = make_data_loaders(
            self.config, 
            self.config.train_cfg.dataset,
            num_replicas=num_replicas,
            global_rank=global_rank)
        
        
        dist.barrier()
        
        # wrap model in DataParallel if needed and transfer to correct device
        print('\n-------------------\nTraining stage\nFound {} GPU devices \n'.format(self.device_count))

        
        print('Model on device: {}'.format("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu"))
        device = torch.device("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, 
                                                    device_ids=[local_rank],
                                                    find_unused_parameters=True)
        dist.barrier()
        
        
        # initialize constants:
        # compute epochs
        if self.config.resume:
            # epochs = self.config.epochs - \
            #     int(self.config.resume_step/len(train_loader))
            # print(f"\n---- Remaining epochs {epochs} ----\n")
            # self._step = int(self.config.resume_step)
            # print(f"\n----Starting step {self._step} ----\n")
            
            # remaining epochs
            epochs = self.config.epochs #- (self.config.resume_step + 1)
            self._step = len(train_loader) * (self.config.resume_step +1)
            print(f"\n---- Remaining epochs {self.config.epochs - (self.config.resume_step+1)} ----\n")
            print(f"\n----Starting step {self._step} ----\n")
            
        else:
            epochs = self.train_cfg.get('epochs', 1)
            self._step = 0
            self.config.resume_step = 0
            

        log_freq = self.train_cfg.get('log_freq', 1000)
        val_freq = self.train_cfg.get('val_freq', 1000)
        print_freq = self.train_cfg.get('print_freq', 10000)
        save_freq = self.train_cfg.get('save_freq', 10000)
        if save_freq == -1:
            save_freq = len(train_loader)
        if val_freq == -1:
            val_freq = len(train_loader)
        print(f"Save frequency {save_freq}")
        print(f"Val frequency {val_freq}")
        
        raw_stats = dict()
        for e in range(self.config.resume_step+1, epochs):
            self._epoch = e
            frac = e / epochs
            print(f"Training frac {frac}")
            # with tqdm(train_loader, unit="batch") as tepoch:
            
            self.train_loop(train_loader=train_loader,
                            scheduler=lr_scheduler,
                            loss_function=loss_function,
                            global_rank=global_rank,
                            local_rank=local_rank,
                            model=model,
                            optimizer=optimizer,
                            raw_stats=raw_stats,
                            epoch=e,
                            log_freq=log_freq,
                            print_freq=print_freq,
                            frac=frac)
                
            dist.barrier()
            
            #### ---- Validation step ----####
            if val_loader is not None:
                val_metric = self.val_loop(val_loader=val_loader,
                                        scheduler=lr_scheduler, 
                                        loss_function=loss_function, 
                                        global_rank=global_rank, 
                                        local_rank=local_rank, 
                                        model=model, 
                                        optimizer=optimizer,
                                        task_names=self.task_names, 
                                        raw_stats=raw_stats, 
                                        epoch= e,)
                

        
class Workspace(object):
    
    def __init__(self, cfg):
        self.trainer = Trainer(allow_val_grad=False, hydra_cfg=cfg)
        print("Finished initializing trainer")
        self.config = self.trainer.config
        
        # map between task and number of tasks
        n_tasks = []
        tasks = dict()
        start = 0
        for i, task in enumerate(cfg.tasks):
            n_tasks.append(task['n_tasks'])
            tasks[task['name']] = (start, task['n_tasks'])
            start += task['n_tasks']

        # move log path to here!
        print('\n----Done initializing Workspace, saving config.yaml to directory: {}----\n'.format(
            self.trainer.save_dir))

        try:
            os.makedirs(self.trainer.save_dir, exist_ok=(
                'burn' in self.trainer.save_dir))
            os.makedirs(join(self.trainer.save_dir, 'stats'), exist_ok=True)
        except:
            pass

        save_config = copy.deepcopy(self.trainer.config)
        OmegaConf.save(config=save_config, f=join(
            self.trainer.save_dir, 'config.yaml'))
        
        
    def run(self):

        torch.multiprocessing.spawn(self.trainer.worker,
                                    nprocs=self.config.num_gpus, 
                                    args=(  self.config.node_id,
                                            self.config.num_gpus,
                                            self.config))
        
        print("Done training")