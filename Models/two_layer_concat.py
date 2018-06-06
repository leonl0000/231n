import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from utils.tavr_torch import get_mean_slice, get_mew_slice

class two_layer_concat(nn.Module):
    def __init__(self, a_layers=[8],
                        b_layers=[8], 
                        ab_layers=[1],
                        standardize_slice=False, device='cpu'):
        super().__init__()
        self.conv_a1 = nn.Conv3d(1, 7, 3, padding=1)
        self.conv_b1 = nn.Conv3d(1, 7, 3, padding=1)
        self.final = nn.Conv3d(16, 1, 1)
        nn.init.kaiming_normal_(self.conv_a1.weight)
        nn.init.kaiming_normal_(self.conv_b1.weight)
        nn.init.kaiming_normal_(self.final.weight)
        
        self.standardize_slice = standardize_slice
        if standardize_slice:
            self.mean = get_mean_slice().to(device=device)
            self.mew = get_mew_slice().to(device=device)
    
    def forward(self, x):
        x1, x2 = x
        
        if self.standardize_slice:
            x1 = (x1 - self.mean)/self.mew
            x2 = (x2 - self.mean)/self.mew
            
        a0 = x1[:,None,:,:,:]
        b0 = x2[:,None,:,:,:]
        a1 = F.relu(self.conv_a1(a0))
        b1 = F.relu(self.conv_b1(b0))
        ab = torch.cat((a1, a0, b1, b0), 1)
        y_hat = self.final(ab)
        
        if self.standardize_slice:
            y_hat = (y_hat * self.mew) + self.mean
        
        return y_hat