import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from utils.tavr_torch import get_mean_slice, get_mew_slice

class two_layer_concat(nn.Module):
    def __init__(self, a_layers=[8],
                        b_layers=[8], 
                        ab_layers=[1]):
        super().__init__()
        self.conv_a1 = nn.Conv3d(1, 7, 3, padding=1)
        self.conv_b1 = nn.Conv3d(1, 7, 3, padding=1)
        self.final = nn.Conv3d(16, 1, 1)
        nn.init.kaiming_normal_(self.conv_a1.weight)
        nn.init.kaiming_normal_(self.conv_b1.weight)
        nn.init.kaiming_normal_(self.final.weight)
    
    def forward(self, x):
        x1, x2 = x
        
        a0 = x1[:,None,:,:,:]
        b0 = x2[:,None,:,:,:]
        a1 = F.relu(self.conv_a1(a0))
        b1 = F.relu(self.conv_b1(b0))
        ab = torch.cat((a1, a0, b1, b0), 1)
        y_hat = self.final(ab)
        
        return y_hat