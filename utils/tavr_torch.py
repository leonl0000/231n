import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os
from os import listdir, mkdir
from os.path import join, isdir

data_root_dir = "data_numpy"
train_dir = join(data_root_dir, "train")
valid_dir = join(data_root_dir, "valid")
test_dir = join(data_root_dir, "test")

if isdir(train_dir):
    print("Training directory found, %d series" % len(listdir(train_dir)))
else:
    print("WARNING: Training directory not found!!!")
    
if isdir(valid_dir):
    print("Validation directory found, %d series" % len(listdir(valid_dir)))
else:
    print("WARNING: Validation directory not found!!!")
    
if isdir(test_dir):
    print("Testing directory found, %d series" % len(listdir(test_dir)))
else:
    print("WARNING: Testing directory not found!!!")

data_dirs = {'__train': train_dir, '__val': valid_dir, '__valid':valid_dir,
             '__validation':valid_dir, '__test': test_dir}
    
def basic_transform(a):
    return torch.tensor(a)
    
class TAVR_3_Frame(Dataset):
    """
    A customized data loader for the TAVR dataset from UKY
    """
    def __init__(self,
                 root,
                 transform=basic_transform,
                 preload=False):
        """
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
    
        """
            Each series (By the name of "ZX-...") has some number of frames, usually 10, 20, or 21
            - self.series_names will just be a list of those names
            - self.frames and self.filenames will each be a list of lists:
                - self.filenames[i][j] will be the filename of the jth timestep of the ith series
                    something like "data_numpy/train/ZX..../j.npy"
            
            An item will be 3 frames, so for a series with N frames, it can have N-2 items.
            When accessing an item, say the 43rd item, we first have to figure out which series
                that is in, and then which time it starts at.
            So, say the 0th series has 20 frames and the 1st has 21. Then, the 0th
                series will have 18 items and the 1st will have 19. Thus, the 43rd item
                will start the 43-18-19 = 6th frame of the of the 2nd series.
            self.itemIndex will be a list to convert an item index to the tuple (series #, frame #).
                With the above example, the 43rd entry in self.itemIndex will be (2, 6)
        """
        
        if root in data_dirs:
            root = data_dirs[root]
        self.root = root
        self.seriesnames = []
        self.itemIndex = []
        self.filenames = []
        self.frames = None
        
        self.transform = transform

        # read filenames
        self.seriesnames = sorted([join(self.root, s) for s in listdir(self.root) if 'ZX' in s])
        self.filenames = [[join(s, f) for f in sorted(listdir(s), key=lambda x: int(x[:-4]))] for s in self.seriesnames]
        
        # Since each item is 3 frames, the starting frame can be at most the third to last frame in a series
        #   so, the total number of items is the count of all the frames except the last 2 in each series
        self.len = sum([len(s) for s in self.filenames]) - 2*len(self.filenames)
        for s_num in range(len(self.filenames)):
            for f_num in range(len(self.filenames[s_num]) - 2):
                self.itemIndex.append((s_num, f_num))
        assert(self.len == len(self.itemIndex))
                
        # if preload dataset into memory
        if preload:
            self._preload()
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.frames = [[np.load(f).astype(np.float32) for f in s] for s in self.filenames]

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        s_num, f_num = self.itemIndex[index]
        if self.frames is not None:
            # If dataset is preloaded
            A = self.frames[s_num][f_num]
            B = self.frames[s_num][f_num + 1]
            C = self.frames[s_num][f_num + 2]
        else:
            # If on-demand data loading
            A = np.load(self.filenames[s_num][f_num]).astype(np.float32)
            B = np.load(self.filenames[s_num][f_num + 1]).astype(np.float32)
            C = np.load(self.filenames[s_num][f_num + 2]).astype(np.float32)
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            A = self.transform(A)
            B = self.transform(B)
            C = self.transform(C)
        
        return A, B, C

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    
    
class TAVR_1_Frame(Dataset):
    """
    A customized data loader for the TAVR dataset from UKY
    """
    def __init__(self,
                 root,
                 transform=basic_transform,
                 preload=False):
        # See TAVR_3_Frame for details
        if root in data_dirs:
            root = data_dirs[root]       
        self.root = root
        self.seriesnames = []
        self.itemIndex = []
        self.filenames = []
        self.frames = None
        
        self.transform = transform

        # read filenames
        self.seriesnames = sorted([join(self.root, s) for s in listdir(self.root) if 'ZX' in s])
        self.filenames = [[join(s, f) for f in sorted(listdir(s), key=lambda x: int(x[:-4]))] for s in self.seriesnames]
        
        # Since each item is 3 frames, the starting frame can be at most the third to last frame in a series
        #   so, the total number of items is the count of all the frames except the last 2 in each series
        self.len = sum([len(s) for s in self.filenames])
        for s_num in range(len(self.filenames)):
            for f_num in range(len(self.filenames[s_num])):
                self.itemIndex.append((s_num, f_num))
        assert(self.len == len(self.itemIndex))
                
        # if preload dataset into memory
        if preload:
            self._preload()
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.frames = [[np.load(f).astype(np.float32) for f in s] for s in self.filenames]

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        s_num, f_num = self.itemIndex[index]
        if self.frames is not None:
            # If dataset is preloaded
            A = self.frames[s_num][f_num]
        else:
            # If on-demand data loading
            A = np.load(self.filenames[s_num][f_num]).astype(np.float32)
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            A = self.transform(A)
        
        return A

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

def collator_3_frame(batch):
    shape = [len(batch), max([b[0].shape[0] for b in batch]),batch[0][0].shape[1], batch[0][0].shape[2]]
    A, B, C, D = torch.zeros(*shape), torch.zeros(*shape), torch.zeros(*shape), torch.zeros(*shape)
    max_z = torch.zeros(len(batch), 1, 1, 1)
    for i, (a, b, c) in enumerate(batch):
        H = a.shape[0]
        A[i, 0:H, :, :] += torch.tensor(a)
        B[i, 0:H, :, :] += torch.tensor(b)
        C[i, 0:H, :, :] += torch.tensor(c)
        D[i, 0:H, :, :] = 1
        max_z[i, 0, 0, 0] = H
    return A, B, C, D, max_z

def collator_1_frame(batch):
    shape = [len(batch), max([b[0].shape[0] for b in batch]),batch[0][0].shape[1], batch[0][0].shape[2]]
    A, D = torch.zeros(*shape), torch.zeros(*shape)
    max_z = torch.zeros(len(batch), 1, 1, 1)
    for i, a in enumerate(batch):
        H = a.shape[0]
        A[i, 0:H, :, :] += torch.tensor(a)
        D[i, 0:H, :, :] = 1
        max_z[i, 0] = H
    return A, B, C, D, max_z


def tavr_dataloader(dset, **kwargs):
    """
        Returns a pytorch dataloader with the appropriate collator
    """
    if 'collate_fn' not in kwargs:
        kwargs['collate_fn'] = collator_3_frame if str(type(dset)) == "<class 'utils.tavr_torch.TAVR_3_Frame'>" \
                                else collator_1_frame
    return DataLoader(dset, **kwargs)