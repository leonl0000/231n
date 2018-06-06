import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from skimage.transform import resize
import numpy as np
import torch

def set_figsize(H, W):
    plt.rcParams['figure.figsize'] = (H, W)

def display_grid(H, W, pics):
    for i, pic in enumerate(pics):
        plt.subplot(H, W, i+1)
        plt.imshow(pic)
        plt.axis('off')
    plt.show()

def z_stretch(pic):
    return resize(pic.type(torch.int), (pic.shape[0]*2.5//1, pic.shape[1]))

def visualize_frame(frame):
    H, L, W = frame.shape
    z1 = frame[H//4,:,:]
    z2 = frame[2*H//4,:,:]
    z3 = frame[3*H//4,:,:]
    y1 = z_stretch(frame[:,L//4,:])
    y2 = z_stretch(frame[:,2*L//4,:])
    y3 = z_stretch(frame[:,3*L//4,:])
    x1 = z_stretch(frame[:,:,W//4])
    x2 = z_stretch(frame[:,:,2*W//4])
    x3 = z_stretch(frame[:,:,3*W//4])
    slices = [z1, z2, z3, y1, y2, y3, x1, x2, x3]
    display_grid(3,3,slices)
    
def get_central_slices(frame):
    H, L, W = frame.shape
    z = frame[H//2,:,:]
    y = z_stretch(frame[:,L//2,:])
    x = z_stretch(frame[:,:,W//2])
    return z,x,y
    