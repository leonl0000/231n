import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
# from torch.nn.parameter import Parameter
from utils.tavr_torch import get_mean_slice, get_mew_slice, get_mean_pixel, get_mew_pixel

class average_model(nn.Module):
    def forward(self, X):
        x1, x2 = X
        return (x1 + x2)/ 2
    
class post_process(nn.Module):
    def __init__(self, kind="pixel"):
        super(post_process, self).__init__()
        if kind=="pixel":
            self.register_buffer('mean', get_mean_pixel())
            self.register_buffer('mew', get_mew_pixel())
        elif kind=="slice":
            self.register_buffer('mean', get_mean_slice())
            self.register_buffer('mew', get_mew_slice())
        else:
            self.register_buffer('mean', torch.zeros(1))
            self.register_buffer('mew', torch.ones(1))
    
    def forward(self, x):
        return x * self.mew + self.mean


class two_layer_basic(nn.Module):
    def __init__(self, a_layers=[8],
                        b_layers=[8], 
                        ab_layers=[1]):
        super().__init__()
        self.conv_a1 = nn.Conv3d(1, 8, 3, padding=1)
        self.conv_b1 = nn.Conv3d(1, 8, 3, padding=1)
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
        ab = torch.cat((a1, b1), 1)
        y_hat = self.final(ab)
        return y_hat
    
class two_d_basic(nn.Module):
    def __init__(self, a_layers=[8],
                        b_layers=[8],
                        ab_layers=[1]):
        super().__init__()
        self.conv_a1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv_b1 = nn.Conv2d(1, 8, 3, padding=1)
        self.final = nn.Conv2d(16, 1, 1)
        nn.init.kaiming_normal_(self.conv_a1.weight)
        nn.init.kaiming_normal_(self.conv_b1.weight)
        nn.init.kaiming_normal_(self.final.weight)
    
    def forward(self, x):
        x1, x2 = x
        N, Z, Y, X = x1.shape
        x1 = x1.view(-1, X, Y)
        x2 = x2.view(-1, X, Y)
        a0 = x1[:,None,:,:]
        b0 = x2[:,None,:,:]
        a1 = F.relu(self.conv_a1(a0))
        b1 = F.relu(self.conv_b1(b0))
        ab = torch.cat((a1, b1), 1)
        y_hat = self.final(ab)
        return y_hat.view(N, Z, Y, X)
    
class two_d_two_layer(nn.Module):
    def __init__(self, a_layers=[8],
                        b_layers=[8],
                        ab_layers=[1]):
        super().__init__()
        self.conv_a1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv_b1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv_a2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_b2 = nn.Conv2d(8, 8, 3, padding=1)
        self.final = nn.Conv2d(16, 1, 1)
        nn.init.kaiming_normal_(self.conv_a1.weight)
        nn.init.kaiming_normal_(self.conv_b1.weight)
        nn.init.kaiming_normal_(self.conv_a2.weight)
        nn.init.kaiming_normal_(self.conv_b2.weight)
        nn.init.kaiming_normal_(self.final.weight)
    
    def forward(self, x):
        x1, x2 = x
        N, Z, Y, X = x1.shape
        x1 = x1.view(-1, X, Y)
        x2 = x2.view(-1, X, Y)
        a0 = x1[:,None,:,:]
        b0 = x2[:,None,:,:]
        
        a1 = F.relu(self.conv_a1(a0))
        b1 = F.relu(self.conv_b1(b0))
        
        a2 = F.relu(self.conv_a2(a1))
        b2 = F.relu(self.conv_b2(b1))
        
        ab = torch.cat((a2, b2), 1)
        y_hat = self.final(ab)
        return y_hat.view(N, Z, Y, X)
        
class two_d_three_layer(nn.Module):
    def __init__(self, a_layers=[8],
                        b_layers=[8],
                        ab_layers=[1]):
        super().__init__()
        
        self.conv_a1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv_b1 = nn.Conv2d(1, 8, 5, padding=2)
        
        self.conv_a2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_b2 = nn.Conv2d(8, 8, 3, padding=1)
        
        self.conv_a3 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_b3 = nn.Conv2d(8, 8, 3, padding=1)
        self.final = nn.Conv2d(16, 1, 1)
        
        nn.init.kaiming_normal_(self.conv_a1.weight)
        nn.init.kaiming_normal_(self.conv_b1.weight)
        
        nn.init.kaiming_normal_(self.conv_a2.weight)
        nn.init.kaiming_normal_(self.conv_b2.weight)
        
        nn.init.kaiming_normal_(self.conv_a3.weight)
        nn.init.kaiming_normal_(self.conv_b3.weight)
        nn.init.kaiming_normal_(self.final.weight)
    
    def forward(self, x):
        x1, x2 = x
        N, Z, Y, X = x1.shape
        x1 = x1.view(-1, X, Y)
        x2 = x2.view(-1, X, Y)
        a0 = x1[:,None,:,:]
        b0 = x2[:,None,:,:]
        
        a1 = F.relu(self.conv_a1(a0))
        b1 = F.relu(self.conv_b1(b0))
        
        a2 = F.relu(self.conv_a2(a1))
        b2 = F.relu(self.conv_b2(b1))
        
        a3 = F.relu(self.conv_a3(a2))
        b3 = F.relu(self.conv_b3(b2))
        
        ab = torch.cat((a3, b3), 1)
        y_hat = self.final(ab)
        return y_hat.view(N, Z, Y, X)