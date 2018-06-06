import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

class batch_l2_loss(nn.Module):
    def forward(self, results):
        y, y_hat, mask, max_z = results
        L2 = (torch.squeeze(y_hat)-y)**2 * mask/(torch.sum(max_z) * 256 * 256)
        L2_total = torch.sqrt(torch.sum(L2))
        return L2_total #, np.sqrt(torch.max(L2).detach().numpy()), L2_total.detach().numpy()
    
class batch_mse_loss(nn.Module):
    def forward(self, results):
        y, y_hat, mask, max_z = results
        SE = (torch.squeeze(y_hat)-y)**2 * mask
        MSE = SE.sum()/(torch.sum(max_z) * 256 * 256)
        return MSE
    
    def hist(self, results, bins=20):
        y, y_hat, mask, max_z = results
        SE = (torch.squeeze(y_hat)-y)**2
        h = torch.hist(torch.cat([frame[:,max_z[i],:,:].reshape(-1) for i, frame in enumerate(SE)]), bins=bins)
        return h
    
# class batch_l4_loss(nn.Module):
#     def forward(self, results):
#         y, y_hat, mask, max_z = results
#         L4 = (torch.squeeze(y_hat)-y)**4 * mask/(torch.sum(max_z) * 256 * 256)
#         L4_total = torch.sqrt(torch.sum(L2))
#         return L4_total #, np.sqrt(torch.max(L2).detach().numpy()), L2_total.detach().numpy()