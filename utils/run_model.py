import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

#TODO: Implement model saving

def test(model, loader, loss_fn, device):
    model.eval()  # set model to evaluation mode
    total_loss = 0.
    total_frames = 0
    with torch.no_grad():
        for t, (x1, y, x2, mask, max_z) in enumerate(loader):
            x1 = x1.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)
            x2 = x2.to(device=device)
            mask = mask.to(device=device)
            max_z = max_z.to(device=device)

            y_hat = model((x1, x2))
            loss = loss_fn((y, y_hat, mask, max_z))
            
            # Ensure equal weighting between batches of different size
            total_loss += loss * x1.shape[0]
            total_frames += x1.shape[0]
        ave_loss = total_loss/total_frames
        print("Validation loss %.4f over %d frames" % (ave_loss, total_frames))           
    return


def train(model, optimizer, train_loader, val_loader, loss_fn, device,
          epochs=1, print_every=100, save_every=0):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x1, y, x2, mask, max_z) in enumerate(train_loader):
            model.train()  # put model to training mode
            print("Time %d"%t)
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

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print(('Epoch %d ' % e) * (t==0) + 'Iteration %d, loss = %.4f' % (t, loss.item()))
                test(model, val_loader, loss_fn, device)
                print()