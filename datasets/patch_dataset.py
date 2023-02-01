import os
import pandas as pd
import numpy as np
import PIL

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor



class PatchDataset(Dataset):
    def __init__(self, crop_patch_dir,  transform=None):

        map_dir_list = os.listdir(crop_patch_dir)

        self.transform = transform
        self.crop_patch_dir = crop_patch_dir
        self.map_dir_list = map_dir_list



    def __len__(self):
        return len(self.map_dir_list)

    def __getitem__(self, idx):
        cur_dir = os.path.join(self.crop_patch_dir, self.map_dir_list[idx]) # get one image dir

        
        jpg_list = sorted(os.listdir(cur_dir))

        # crop folder might be empty. if empty, get a random image folder
        if len(jpg_list) == 0:
            return self.__getitem__(np.random.randint(0, len(self.map_dir_list)))

        h_max, w_max = jpg_list[-1].split('.jpg')[0].split('_')
        h_max, w_max = int(h_max[1:]), int(w_max[1:])

        # get one random patch from the map image dir 
        if h_max > 1 and w_max > 1: 
            h_randoms, w_randoms = np.random.randint(1, h_max, size = 2), np.random.randint(1,w_max, size=2) # (do not select the border patches)
        else:
            h_randoms, w_randoms = np.random.randint(0, h_max, size = 2), np.random.randint(0,w_max, size=2)
        

        img_path1 = os.path.join(cur_dir, 'h'+str(h_randoms[0]) + '_' + 'w' + str(w_randoms[0]))+'.jpg'
        img1 = PIL.Image.open(img_path1)

        img_path2 = os.path.join(cur_dir, 'h'+str(h_randoms[1]) + '_' + 'w' + str(w_randoms[1]))+'.jpg'
        img2 = PIL.Image.open(img_path2)


        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return {'img1':img1, 'img2':img2, 'idx': self.map_dir_list[idx]}


