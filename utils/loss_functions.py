import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

class batch_l2_loss(nn.Module):
    def forward(self, results):
        y, y_hat, mask, max_z = results
        L2 = (y_hat-y)**2 * mask
        L2_total = torch.sqrt(torch.sum(L2))
        L2_ave = L2_total/(torch.sum(max_z) * 256 * 256)
        return L2_ave #, np.sqrt(torch.max(L2).detach().numpy()), L2_total.detach().numpy()