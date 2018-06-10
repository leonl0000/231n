import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import sys
from time import time

import os
from os import listdir, mkdir
from os.path import join, isdir

save_dir = "model_checkpoints"

def save(model_name, iteration,
             model, optimizer, loss_history=None):
    if not isdir(join(save_dir, model_name)):
        mkdir(join(save_dir, model_name))
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    save_to = join(save_dir, model_name, "%s-%d"%(model_name, iteration))
    torch.save(state, save_to)
    if loss_history:
        torch.save(loss_history, join(save_dir, model_name, "loss"))
    print('model saved to %s' % save_to)

def load(model_name, iteration,
             model, optimizer, map_location=None):
    save_to = join(save_dir, model_name, "%s-%d"%(model_name, iteration))
    if map_location != None:
        state = torch.load(save_to, map_location=map_location)
    else:
        state = torch.load(save_to)
#     if device != None:
#         for key, v in state['state_dict'].items():
#             state['state_dict'][key] = v.to(device=device)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % save_to)
    
def get_loss_history(model_name):
    return torch.load(join(save_dir, model_name, "loss"))
    
    
    

def printw(s):
    sys.stdout.write(s)
    sys.stdout.flush()


# def test(model, post_proc, loader, loss_fn, device):
#     with torch.no_grad():
#         model.eval()  # set model to evaluation mode
#         ave_loss = 0.
#         total_frames = 0
#         for t, (x1, y, x2, mask, max_z) in enumerate(loader):
#             x1 = x1.to(device=device)  # move to device, e.g. GPU
#             y = post_proc(y.to(device=device))
#             x2 = x2.to(device=device)
#             mask = mask.to(device=device)
#             max_z = max_z.to(device=device)
            
#             if t==0:
#                 batch_size = x1.shape[0]
#             y_hat = model((x1, x2))
#             y_hat = post_proc(y_hat)
#             loss = loss_fn((y, y_hat, mask, max_z))
            
#             # Ensure equal weighting between batches of different size
#             ave_loss += loss * x1.shape[0] / batch_size / len(loader)
#             total_frames += x1.shape[0]
#         print("Validation loss %.4f over %d frames" % (ave_loss, total_frames))           
#     return ave_loss

def test(model, post_proc, loader, loss_fn, device):
    with torch.no_grad():
        model.eval()  # set model to evaluation mode
        losses = []
        total_frames = 0
        for t, (x1, y, x2, mask, max_z) in enumerate(loader):
            x1 = x1.to(device=device)  # move to device, e.g. GPU
            y = post_proc(y.to(device=device))
            x2 = x2.to(device=device)
            mask = mask.to(device=device)
            max_z = max_z.to(device=device)
            
            y_hat = post_proc(model((x1, x2)))
            losses.append(loss_fn.f1((y, y_hat, mask, max_z)))
            total_frames += x1.shape[0]
            
        ave_loss = loss_fn.f2(losses)
        print("Validation loss %.4f over %d frames" % (ave_loss, total_frames))           
    return ave_loss


def train(model, post_proc, optimizer, train_loader, val_loader, loss_fn, device, 
          model_name, loss_history=None,
          epochs=1, print_every=30, print_level=1, lr_decay=1):
    """
        Print levels:
            1. only every print_every
            2. Each iteration, print the iteration number (if mod 10 = 0) or '.' (if mod 10 !=0) & epoch time
            3. On every print_every, print the norm, gradnorm, and update/norm ratios
            4. Every 3 iterations, print the loss inline
    """
    
    
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    init_lr = optimizer.param_groups[0]['lr']
    if loss_history is None:
        loss_history = {
            'train': [],
            'train_c': [],
            'valid': [],
            'epoch': 0,
            'iteration': 0,
            'print_every': print_every}   
    if loss_history['print_every'] != print_every:
        print("Warning: print_every changed at iteration %d, %dth in the list"%(loss_history['iteration'], len(loss_history['valid'])))
        loss_history['print_every'] = print_every
    original_e = loss_history['epoch']
    
    for e in range(epochs):
        loss_history['epoch'] = original_e + e
        toc = time()
        if e!= 0 and print_level >= 2:
            print("(Epoch time: %.2f minutes. Total epochs: %d)"%((toc-tic)/60, loss_history['epoch']))
        tic = toc
        
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
            with torch.no_grad():
                if loss_history['iteration'] % print_every == 0 and print_level >= 3:
                    sd_copy = {}
                    for n, p in model.named_parameters():
                        sd_copy[n] = torch.tensor(p)

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.param_groups[0]['lr'] = init_lr * lr_decay ** (e + t/len(train_loader))
            optimizer.step()
            
            with torch.no_grad():
                loss_history['train'].append(loss.item())
                if loss_history['iteration'] % print_every == 0:
                    model.eval()
                    c_y = post_proc(y)
                    c_y_hat = post_proc(y_hat)
                    c_loss = loss_fn((c_y, c_y_hat, mask, max_z))
                    print('\nIteration %d, loss = %.4f, corrected loss = %.4f' %\
                          (loss_history['iteration'], loss.item(), c_loss.item()))
                    loss_history['train_c'].append(c_loss.item())
                    valid_loss = test(model, post_proc, val_loader, loss_fn, device)
                    loss_history['valid'].append(valid_loss.item())
                    if str(type(model))[-10:-2] == "Parallel":
                        save(model_name, loss_history['iteration'], model.module, optimizer, loss_history)
                    else:
                        save(model_name, loss_history['iteration'], model, optimizer, loss_history)
                    if print_level >= 3:
                        for name, param in model.named_parameters():  
                            if param.requires_grad:   
                                pnorm = sd_copy[name].norm().item()
                                update_norm = (param - sd_copy[name]).norm().item()
                                print("%s,   \tnorm: %.4e, \tupdate norm: %.4e \tUpdate/norm: %.4e"%
                                      (name, pnorm, update_norm, update_norm/pnorm))
                    print()
                elif print_level >= 2:
                    if loss_history['iteration']%10 == 0:
                        printw("\nIter %d"%loss_history['iteration'])
                    elif (loss_history['iteration']%10)%3 == 0:
                        printw(". ")
                        if print_level >=4:
                            printw("%.4f"%loss.item())
                    else:
                        printw(".")
                loss_history['iteration'] += 1
        # Save at the end
        print()
        if str(type(model))[-10:-2] == "Parallel":
            save(model_name, loss_history['iteration'], model.module, optimizer, loss_history)
        else:
            save(model_name, loss_history['iteration'], model, optimizer, loss_history)
                