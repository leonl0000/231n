import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import sys
from time import time

def printw(s):
    sys.stdout.write(s)
    sys.stdout.flush()

#TODO: Implement model saving

def test(model, loader, loss_fn, device):
    with torch.no_grad():
        model.eval()  # set model to evaluation mode
        ave_loss = 0.
        total_frames = 0
        for t, (x1, y, x2, mask, max_z) in enumerate(loader):
            x1 = x1.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            x2 = x2.to(device=device)
            mask = mask.to(device=device)
            max_z = max_z.to(device=device)
            
            if t==0:
                batch_size = x1.shape[0]

            y_hat = model((x1, x2))
            loss = loss_fn((y, y_hat, mask, max_z))
            
            # Ensure equal weighting between batches of different size
            ave_loss += loss * x1.shape[0] / batch_size / len(loader)
            total_frames += x1.shape[0]
        print("Validation loss %.4f over %d frames" % (ave_loss, total_frames))           
    return


def train(model, optimizer, train_loader, val_loader, loss_fn, device,
          epochs=1, print_every=100, print_level=1, save_every=0, lr_decay=1):
    """
        Print levels:
            1. only every print_every
            2. Each iteration, print the iteration number (if mod 10 = 0) or '.' (if mod 10 !=0) & epoch time
            3. On every print_every, print the norm, gradnorm, and update/norm ratios
    """
    
    
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    init_lr = optimizer.param_groups[0]['lr']
    t = 0
    for e in range(epochs):
        tnew = time()
        if e!= 0 and print_level >= 2:
            print("Epoch time: %.2f minutes"%((tnew-t)/60))
        for t, (x1, y, x2, mask, max_z) in enumerate(train_loader):
            
            model.train()  # put model to training mode
            x1 = x1.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            x2 = x2.to(device=device)
            mask = mask.to(device=device)
            max_z = max_z.to(device=device)

            y_hat = model((x1, x2))
            loss = loss_fn((y, y_hat, mask, max_z))

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            
            # Print the params and grads
            # Store old params
            if t % print_every == 0 and print_level >= 3:
                sd_copy = {}
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        sd_copy[n] = torch.tensor(p)

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.param_groups[0]['lr'] = init_lr * lr_decay ** (e + t/len(train_loader))
            optimizer.step()
            
            if print_level >= 1:
                if t % print_every == 0:
                    print('\n' + ('****Epoch %d ' % e) * (t==0) + 'Iteration %d, loss = %.4f' % (t, loss.item()))
                    test(model, val_loader, loss_fn, device)
                    if print_level >= 3:
                        with torch.no_grad():
                            for name, param in model.named_parameters():  
                                if param.requires_grad:   
                                    pnorm = sd_copy[name].norm().item()
                                    update_norm = (param - sd_copy[name]).norm().item()
                                    print("%s,   \tnorm: %.4e, \tupdate norm: %.4e \tUpdate/norm: %.4e"%
                                          (name, pnorm, update_norm, update_norm/pnorm))
                    printw("\nIter %d"%t)
                elif print_level >= 2:
                    if t%10 == 0:
                        printw("\nIter %d"%t)
                    elif (t%10)%3 == 0:
                        printw(". ")
                    else:
                        printw(".")
                