import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 


class nm_layer_net(nn.Module):
    def __init__(self, sep_layers=[8],
                        cat_layers=[1], residual=False):
        super().__init__()
        
        self.num_sep = len(sep_layers)
        self.num_cat = len(cat_layers)
        self.residual = residual
        sep_layers = [1] + sep_layers
        cat_layers = [sep_layers[-1]*2] + cat_layers
        
        for i in range(1, len(sep_layers)):
            self.add_module("conv_a%d"%i, nn.Conv3d(sep_layers[i-1], sep_layers[i], 3, padding=1))
            self.add_module("conv_b%d"%i, nn.Conv3d(sep_layers[i-1], sep_layers[i], 3, padding=1))
            nn.init.kaiming_normal_(getattr(self, "conv_a%d"%i).weight)
            nn.init.kaiming_normal_(getattr(self, "conv_b%d"%i).weight)
        for i in range(1, len(cat_layers)):
            self.add_module("conv_ab%d"%i, nn.Conv3d(cat_layers[i-1], cat_layers[i], 3, padding=1))
            nn.init.kaiming_normal_(getattr(self, "conv_ab%d"%i).weight)
    
    def forward(self, x):
        x1, x2 = x
            
        a = [x1[:,None,:,:,:]]
        b = [x2[:,None,:,:,:]]
        for i in range(1, self.num_sep+1):
            a.append(F.relu(getattr(self, "conv_a%d"%i)(a[-1])))
            b.append(F.relu(getattr(self, "conv_b%d"%i)(b[-1])))
        ab = [torch.cat((a[-1], b[-1]), 1)]
        for i in range(1, self.num_cat+1):
            ab.append(F.relu(getattr(self, "conv_ab%d"%i)(ab[-1])))
        y_hat = ab[-1]
        if self.residual:
            y_hat += (a[0]+b[0])/2
        return ab[-1]

class Parallel_Residual(nn.Module):
    def __init__(self, num_modules=2, sep_layers=[8],
                        cat_layers=[1]):
        super().__init__()
        self.num_modules=num_modules
        for i in range(self.num_modules):
            self.add_module("para_%d"%i, nm_layer_net(sep_layers, cat_layers, False))

    def forward(self, x):
        x1, x2 = x
        ave = ((x1+x2)/2)[:,None,:,:]
        residuals = [getattr(self, "para_%d"%i)(x) for i in range(self.num_modules)]
        return sum(residuals) + ave
            
        
    
    