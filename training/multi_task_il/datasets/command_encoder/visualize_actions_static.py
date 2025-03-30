import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import cv2


def plot_actions(i, dataset_names, actions_data, fig, gs00):
    
    if len(actions_data[dataset_names[i]].keys()) == 7:
        ax01 = fig.add_subplot(gs00[3,3])
        ax01.title.set_text('x')
        ax02 = fig.add_subplot(gs00[3,4])   
        ax02.title.set_text('y')
        ax03 = fig.add_subplot(gs00[3,5])
        ax03.title.set_text('z')
        # ax02 = fig.add_subplot(gs00[-2,2], projection='polar')   
        # ax03 = fig.add_subplot(gs00[-2,3], projection='polar')   
        # ax04 = fig.add_subplot(gs00[-1,2], projection='polar')
        ax04 = fig.add_subplot(gs00[4,3])
        ax04.title.set_text('roll')
        ax05 = fig.add_subplot(gs00[4,4])
        ax05.title.set_text('pitch')
        ax06 = fig.add_subplot(gs00[4,5])
        ax06.title.set_text('yaw')
        ax07 = fig.add_subplot(gs00[5,5])
        ax07.title.set_text('gripper')
        
        
        key_dataset = dataset_names[i]
        ax01.plot(actions_data[key_dataset]['x'])
        ax02.plot(actions_data[key_dataset]['y'])
        ax02.sharey(ax01)
        ax03.plot(actions_data[key_dataset]['z'])
        ax03.sharey(ax02)
        
        ax04.plot(actions_data[key_dataset]['roll'])
        ax04.title.set_text(r'$\theta$')
        ax05.plot(actions_data[key_dataset]['pitch'])
        ax05.sharey(ax04)
        ax05.title.set_text('e1')
        ax06.plot(actions_data[key_dataset]['yaw'])
        ax06.sharey(ax05)
        ax06.title.set_text('e2')
        
        ax07.plot(actions_data[key_dataset]['gripper'])
        
    elif len(actions_data[dataset_names[i]].keys()) == 8:
        ax01 = fig.add_subplot(gs00[3,3])
        ax01.title.set_text('x')
        ax02 = fig.add_subplot(gs00[3,4])   
        ax02.title.set_text('y')
        ax02.sharey(ax01)
        ax03 = fig.add_subplot(gs00[3,5])
        ax03.title.set_text('z')
        ax03.sharey(ax02)
        # ax02 = fig.add_subplot(gs00[-2,2], projection='polar')   
        # ax03 = fig.add_subplot(gs00[-2,3], projection='polar')   
        # ax04 = fig.add_subplot(gs00[-1,2], projection='polar')
        ax04 = fig.add_subplot(gs00[4,3])
        ax04.title.set_text('q1')
        ax05 = fig.add_subplot(gs00[4,4])
        ax05.title.set_text('q2')
        ax05.sharey(ax04)
        ax06 = fig.add_subplot(gs00[4,5])
        ax06.title.set_text('q3')
        ax06.sharey(ax05)
        ax07 = fig.add_subplot(gs00[5,4])
        ax07.title.set_text('q4')
        ax07.sharey(ax06)
        ax08 = fig.add_subplot(gs00[5,5])
        ax08.title.set_text('gripper')
        
        key_dataset = dataset_names[i]
        ax01.plot(actions_data[key_dataset]['x'])
        ax02.plot(actions_data[key_dataset]['y'])
        ax03.plot(actions_data[key_dataset]['z'])
        
        ax04.plot(actions_data[key_dataset]['q1'])
        ax05.plot(actions_data[key_dataset]['q2'])
        ax06.plot(actions_data[key_dataset]['q3'])
        ax07.plot(actions_data[key_dataset]['q4'])
        
        ax08.plot(actions_data[key_dataset]['gripper'])


def get_traj_data(finetuning_datasets_path, ur5_uni_paths):
    # search for 1 traj for each dataset
    actions_data = {}
    dataset_names = []
    traj_paths = []
    finetuning_datasets = os.listdir(finetuning_datasets_path)
    finetuning_datasets = [i for i in finetuning_datasets if 'converted' in i and not 'old' in i and not 'droid' in i]
    for dataset_name in finetuning_datasets:
        if 'converted' in dataset_name and not 'old' in dataset_name:
            dataset_path = finetuning_datasets_path + f'/{dataset_name}'
            for dir,subdirs,files in os.walk(dataset_path):
                found_pkl = False
                for file in files:
                    if '.pkl' in file and not 'task_embedding' in file:
                        traj_path = f'{dir}/{file}'    
                        traj_paths.append(traj_path)
                        
                        dataset_names.append(dataset_name)
                        actions_data[dataset_name] = {}
                        found_pkl = True
                        break
                if found_pkl:
                    break
    
    # for folder_path in ur5_uni_paths:
    #     traj_path = f'{folder_path}/task_00/traj000.pkl'
    #     traj_paths.append(traj_path)
        
    #     dataset_names.append(folder_path.split('/')[-1])
    #     actions_data[folder_path.split('/')[-1]] = {}
        
    # open trajectories
    traj_datas = []
    for traj_path in traj_paths:
        with open(traj_path, "rb") as f:
            traj_data = pickle.load(f)
        traj_datas.append(traj_data)
        
    return traj_datas, dataset_names, actions_data


def fill_actions_data(traj_datas, actions_data):
        # save actions
    
    for i, traj_data in enumerate(traj_datas):
        if len(traj_data['traj'].get(1)['action']) == 7:
            print(f'{dataset_names[i]} has action len 7')
            key_dataset = dataset_names[i]
            actions_data[key_dataset]['x'] = []
            actions_data[key_dataset]['y'] = []
            actions_data[key_dataset]['z'] = []
            actions_data[key_dataset]['roll'] = []
            actions_data[key_dataset]['pitch'] = []
            actions_data[key_dataset]['yaw'] = []
            actions_data[key_dataset]['gripper'] = []
            
            traj = traj_data['traj']
            for step in range(traj_data['len']):
                try:
                    action = traj.get(step)['action']
                    actions_data[key_dataset]['x'].append(action[0])
                    actions_data[key_dataset]['y'].append(action[1])
                    actions_data[key_dataset]['z'].append(action[2])
                    actions_data[key_dataset]['roll'].append(action[3])
                    actions_data[key_dataset]['pitch'].append(action[4])
                    actions_data[key_dataset]['yaw'].append(action[5])
                    actions_data[key_dataset]['gripper'].append(action[6])
                except KeyError:
                    print(f'[WARNING] there is shift in {key_dataset}')
                    pass
        
        elif len(traj_data['traj'].get(1)['action']) == 8:
            print(f'{dataset_names[i]} has action len 8')
            key_dataset = dataset_names[i]
            actions_data[key_dataset]['x'] = []
            actions_data[key_dataset]['y'] = []
            actions_data[key_dataset]['z'] = []
            actions_data[key_dataset]['gripper'] = []
            actions_data[key_dataset]['q1'] = []
            actions_data[key_dataset]['q2'] = []
            actions_data[key_dataset]['q3'] = []
            actions_data[key_dataset]['q4'] = []
            
            traj = traj_data['traj']
            for step in range(traj_data['len']):
                try:
                    action = traj.get(step)['action']
                    actions_data[key_dataset]['x'].append(action[0])
                    actions_data[key_dataset]['y'].append(action[1])
                    actions_data[key_dataset]['z'].append(action[2])
                    actions_data[key_dataset]['q1'].append(action[3])
                    actions_data[key_dataset]['q2'].append(action[4])
                    actions_data[key_dataset]['q3'].append(action[5])
                    actions_data[key_dataset]['q4'].append(action[6])
                    actions_data[key_dataset]['gripper'].append(action[7])
                except KeyError:
                    print(f'[WARNING] there is shift in {key_dataset}')
                    pass

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--save_png_name", default='prova_x_3')
    parser.add_argument("--one_image", action='store_true', help="if plot all the datasets in one image")
    parser.add_argument("--splitted_images",action='store_true', help="use this if you want images saved splitted")
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    save_file_name = args.save_png_name + '.png'
    print(f'saving to {save_file_name}')
        
    finetuning_datasets_path = '/user/frosa/multi_task_lfd/datasets'
    real_ur5_dataset_path = '/raid/home/frosa_Loc/opt_dataset/pick_place/real_new_ur5e_pick_place'
    sim_ur5_dataset_path = '/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/ur5e_pick_place'
    ur5_uni_paths = [real_ur5_dataset_path, sim_ur5_dataset_path]
    
    traj_datas, dataset_names, actions_data = get_traj_data(finetuning_datasets_path, ur5_uni_paths)
    
    fill_actions_data(traj_datas, actions_data)
    
    
    # find global min and global max in order to make 3d plots aligned
    mins_x = []
    mins_y = []
    mins_z = []
    maxs_x = []
    maxs_y = []
    maxs_z = []
    
    for k, a in actions_data.items():
        mins_x.append(np.min(a['x']))
        mins_y.append(np.min(a['y']))
        mins_z.append(np.min(a['z']))
        maxs_x.append(np.max(a['x']))
        maxs_y.append(np.max(a['y']))
        maxs_z.append(np.max(a['z']))

    global_mins = [np.min(np.array(x)) for x in [mins_x, mins_y, mins_z]]
    global_maxs = [np.max(np.array(x)) for x in [maxs_x, maxs_y, maxs_z]]
            
    if not args.one_image and not args.splitted_images: # if we want to save each dataset to a different png
        # create folder
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_png_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        for i in range(len(traj_datas)):
            
            fig = plt.figure()
            fig.set_figheight(12.0)
            fig.set_figwidth(12.0)
            gs0 = gridspec.GridSpec(6,6, figure=fig)
            
            # get info
            sample_traj = traj_datas[i]['traj']
            len_traj = traj_datas[i]['len']
            # command_traj = traj_datas[i]['command']
            
            ax_image = fig.add_subplot(gs0[:3,:])
            ax_image.title.set_text('Frame samples')
            
            # get 3 frame: first, middle and last
            try:
                frames = np.concatenate([sample_traj.get(i)['obs']['image'] for i in [0, len_traj // 2, -1]], axis=1)
            except KeyError:
                frames = np.concatenate([sample_traj.get(i)['obs']['camera_front_image'] for i in [0, len_traj // 2, -1]], axis=1)
                
            
            key_dataset = dataset_names[i]
            
            if key_dataset == 'real_new_ur5e_pick_place_converted_absolute':
                frames = frames[:,:,::-1]
            ax_image.imshow(frames)
            
            ax_3d = fig.add_subplot(gs0[3:,0:3], projection='3d')
            ax_3d.title.set_text('3d trajectory plot')
            
            # set limits according to max and min value
            
            try:
                min_x, max_x = np.min(actions_data[key_dataset]['x']), np.max(actions_data[key_dataset]['x'])
                min_y, max_y = np.min(actions_data[key_dataset]['y']), np.max(actions_data[key_dataset]['y'])
                min_z, max_z = np.min(actions_data[key_dataset]['z']), np.max(actions_data[key_dataset]['z'])
                ax_3d.axes.set_xlim3d(left=min_x, right=max_x)
                ax_3d.axes.set_ylim3d(bottom=min_y, top=max_y) 
                ax_3d.axes.set_zlim3d(bottom=min_z, top=max_z) 
            except IndexError:
                pass
            
            # plot 3d
            im = ax_3d.scatter(actions_data[key_dataset]['x'], actions_data[key_dataset]['y'], actions_data[key_dataset]['z'], label='traj', c=range(len_traj), cmap=plt.viridis())
            ax_3d.set_xlabel('x')
            ax_3d.set_ylabel('y')
            ax_3d.set_zlabel('z')
            cbar = fig.colorbar(im, ax=ax_3d)
            cbar.set_label('step number')
            
            # plot actions
            plot_actions(i, dataset_names, actions_data, fig, gs0)
            save_dataset_plot_path = os.path.join(save_dir, dataset_names[i])
            # plt.suptitle(f'Sample from: {dataset_names[i]}', fontsize='xx-large')
            fig.tight_layout()
            plt.savefig(save_dataset_plot_path, bbox_inches='tight')
            print(f'Saved: {save_dataset_plot_path}')
    elif args.splitted_images:
        # create folder
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_png_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        for i in range(len(traj_datas)):
            
            key_dataset = dataset_names[i]
            
            save_dir_single_dataset = os.path.join(save_dir, key_dataset)
            if not os.path.exists(save_dir_single_dataset):
                os.mkdir(save_dir_single_dataset)
            
            # get info
            sample_traj = traj_datas[i]['traj']
            len_traj = traj_datas[i]['len']
            try:
                command_traj = traj_datas[i]['command']
                print(f'[{key_dataset}] Command: {command_traj}')
            except Exception:
                print(f'[{key_dataset}] No command found!')
            
            
            # get 3 frame: first, middle and last
            # get 3 frame: first, middle and last
            try:
                frames = np.concatenate([sample_traj.get(i)['obs']['image'] for i in [0, len_traj // 2, -1]], axis=1)
            except KeyError:
                frames = np.concatenate([sample_traj.get(i)['obs']['camera_front_image'] for i in [0, len_traj // 2, -1]], axis=1)    
            
            key_dataset = dataset_names[i]
            
            if key_dataset == 'real_new_ur5e_pick_place_converted_absolute':
                frames = frames[:,:,::-1]
            fig = plt.figure()
            plt.imsave(os.path.join(save_dir_single_dataset, f'frames.png'), frames)
            plt.close()
                    
            # print(f'[{key_dataset}] Original frame shape: {frames[0].shape}')
                
            # for f_idx,f in enumerate(frames):
            #     fig = plt.figure()
            #     # fig,ax = plt.figure()
            #     # plt.savefig(os.path.join(save_dir_single_dataset, f'frame_{f_idx}'), bbox_inches='tight')
            #     plt.imsave(os.path.join(save_dir_single_dataset, f'frame_{f_idx}.png'), f)
            #     plt.close()
            
            ax_3d = plt.figure().add_subplot(projection='3d')
            # ax_3d.title.set_text('3d trajectory plot')
            
            # set limits according to max and min value
            
            try:
                min_x, max_x = global_mins[0], global_maxs[0]
                min_y, max_y = global_mins[1], global_maxs[1]
                min_z, max_z = global_mins[2], global_maxs[2]
                ax_3d.axes.set_xlim3d(left=min_x, right=max_x)
                ax_3d.axes.set_ylim3d(bottom=min_y, top=max_y) 
                ax_3d.axes.set_zlim3d(bottom=min_z, top=max_z) 
            except IndexError:
                pass
            
            # plot 3d
            im = ax_3d.scatter(actions_data[key_dataset]['x'], actions_data[key_dataset]['y'], actions_data[key_dataset]['z'], label='traj', c=range(len_traj), cmap=plt.viridis())
            ax_3d.set_xlabel('x')
            ax_3d.set_ylabel('y')
            ax_3d.set_zlabel('z')
            cbar = fig.colorbar(im, ax=ax_3d)
            cbar.set_label('Step number')
            
            # plt.suptitle(f'Sample from: {dataset_names[i]}', fontsize='xx-large')
            fig.tight_layout()
            plt.savefig(os.path.join(save_dir_single_dataset, f'3d_plot'), bbox_inches='tight')
            # print(f'Saved: {os.path.join(save_dir, f'3d_plot')}')     
    else:
        fig = plt.figure()
        fig.set_figheight(40.0)
        fig.set_figwidth(40.0)
        
        n_cols, n_rows  = 2, len(traj_datas) // 2
        gs0 = gridspec.GridSpec(n_rows,n_cols, figure=fig)
        for i in range(len(traj_datas)):
            gs00 = gridspec.GridSpecFromSubplotSpec(6,6, subplot_spec=gs0[i])
            
            ax_image = fig.add_subplot(gs00[:3,:])
            ax_image.title.set_text('image')
            ax_3d = fig.add_subplot(gs00[3:,0:3], projection='3d')
            ax_3d.title.set_text('3d pos plot')
            
            # set limits according to max and min value
            key_dataset = dataset_names[i]
            try:
                min_x, max_x = np.min(actions_data[key_dataset]['x']), np.max(actions_data[key_dataset]['x'])
                min_y, max_y = np.min(actions_data[key_dataset]['y']), np.max(actions_data[key_dataset]['y'])
                min_z, max_z = np.min(actions_data[key_dataset]['z']), np.max(actions_data[key_dataset]['z'])
                ax_3d.axes.set_xlim3d(left=min_x, right=max_x)
                ax_3d.axes.set_ylim3d(bottom=min_y, top=max_y) 
                ax_3d.axes.set_zlim3d(bottom=min_z, top=max_z) 
            except IndexError:
                pass
            
            # plot frames
            
            # plot 3d
            
            # plot actions
            plot_actions(i, dataset_names, actions_data, fig, gs00)

            
            
            
        plt.savefig(save_file_name)

    print('blalblablal')
    print('csdovnvsdnc')  
    
    
    
# index = 4
# for i in range(traj_datas[index]['len']):
#         if traj_datas[index]['traj'].get(i)['action'][3] > 1.0 or traj_datas[index]['traj'].get(i)['action'][3] < -1.0:
#             action = traj_datas[index]['traj'].get(i)['action'][3]
#             print(f'{action}, step: {i}')

# index = -2
# for i in range(traj_datas[index]['len']):
#     action = traj_datas[index]['traj'].get(i)['action'][3]
#     print(f'{action}, step: {i}')
        
    # draw
    
