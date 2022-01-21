## Maintainer: Jingyu Song #####
## Contact: jingyuso@umich.edu #####
## edit to fit the input of JS3CNet

import os
import numpy as np
import random
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from Data.Generation.ShapeContainer import ShapeContainer


def collate_fn_test(data):
    input_batch = [bi[0] for bi in data]
    output_batch = [bi[1] for bi in data]
    counts_batch = [bi[2] for bi in data]
    return input_batch, torch.from_numpy(np.array(output_batch)).cuda(), torch.from_numpy(np.array(counts_batch)).cuda()


class CarlaDataset(Dataset):
    """Carla Simulation Dataset for 3D mapping project
    
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """
    def __init__(self, directory,
        device='cuda',
        num_frames=4,
        cylindrical=True
        ):
        '''Constructor.
        Parameters:
            directory: directory to the dataset

        '''

        self._directory = directory
        self._num_frames = num_frames
        self._device = device
        
        self._scenes = sorted(os.listdir(self._directory))
        if cylindrical:
            self._scenes = [os.path.join(scene, "cylindrical") for scene in self._scenes]
        else:
            self._scenes = [os.path.join(scene, "cartesian") for scene in self._scenes]

        self._num_scenes = len(self._scenes)
        self._num_frames_scene = []

        param_file = os.path.join(self._directory, self._scenes[0], 'evaluation', 'params.json')
        with open(param_file) as f:
            self._eval_param = json.load(f)
        
        self._out_dim = self._eval_param['num_channels']
        self._grid_size = self._eval_param['grid_size']
        self._eval_size = list(np.uint32(self._grid_size))

        self._velodyne_list = []
        self._label_list = []
        self._pred_list = []
        self._eval_labels = []
        self._eval_counts = []
        self._frames_list = []
        self._timestamps = []
        self._poses = [] 

        for scene in self._scenes:
            velodyne_dir = os.path.join(self._directory, scene, 'velodyne')
            label_dir = os.path.join(self._directory, scene, 'labels')
            pred_dir = os.path.join(self._directory, scene, 'predictions')
            eval_dir = os.path.join(self._directory, scene, 'evaluation')
            
            self._num_frames_scene.append(len(os.listdir(velodyne_dir)))

            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))]
            self._frames_list.extend(frames_list)
            self._velodyne_list.extend([os.path.join(velodyne_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._label_list.extend([os.path.join(label_dir, str(frame).zfill(6)+'.label') for frame in frames_list])
            self._pred_list.extend([os.path.join(pred_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._eval_labels.extend([os.path.join(eval_dir, str(frame).zfill(6)+'.label') for frame in frames_list])
            self._eval_counts.extend([os.path.join(eval_dir, str(frame).zfill(6) + '.bin') for frame in frames_list])
            self._timestamps.append(np.loadtxt(os.path.join(self._directory, scene, 'times.txt')))
            self._poses.append(np.loadtxt(os.path.join(self._directory, scene, 'poses.txt')))
            # for poses and timestamps
        self._timestamps = np.array(self._timestamps).reshape(sum(self._num_frames_scene))
        self._poses = np.array(self._poses).reshape(sum(self._num_frames_scene), 12)
        
        self._cum_num_frames = np.cumsum(np.array(self._num_frames_scene) - self._num_frames + 1)
        

    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return sum(self._num_frames_scene)


    def __getitem__(self, idx):
        # -1 indicates no data
        # the final index is the output
        idx_range = self.find_horizon(idx)
         
        self._current_horizon = []
        for i in idx_range:
            if i == -1: # Zero pad
                points = np.zeros((1, self._out_dim + 3 + 3), dtype=np.float32)
                
            else:
                pcl = np.fromfile(self._velodyne_list[i],dtype=np.float32).reshape(-1,4)[:, :3]
                pred = np.fromfile(self._pred_list[i],dtype=np.float32).reshape(-1,3)
                label = np.fromfile(self._label_list[i],dtype=np.uint32)

                # one hot encoding for each label
                label_oh = np.zeros((label.size, self._out_dim))
                label_oh[np.arange(label.size),label] = 1
                label_oh = np.float32(label_oh)
                points = np.c_[pcl, pred, label_oh]
            self._current_horizon.append(torch.from_numpy(points).to(self._device))
        
        output = np.fromfile(self._eval_labels[idx_range[-1]],dtype=np.uint32).reshape(self._eval_size).astype(np.uint8)
        counts = np.fromfile(self._eval_counts[idx_range[-1]],dtype=np.float32).reshape(self._eval_size)

        return self._current_horizon, output, counts
        
        # no enough frames
    
    def find_horizon(self, idx):
        end_idx = idx
        idx_range = np.arange(idx-self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[end_idx]) - int(self._frames_list[i]) for i in idx_range])
        good_difs = -1 * (np.arange(-self._num_frames, 0) + 1)
        
        idx_range[good_difs != diffs] = -1

        return idx_range