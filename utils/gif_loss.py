import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio #for writing the gifs
import numpy as np

# matplotlib.use('Agg') #Prevents windows from immediately forming, will this be an issue for the rest of the notebook?

def make_gif(sequence, filename = 'ani', all_axes = False, fps = 3):
    N, H, L, W = sequence[0].shape
    if not all_axes:
        images = []
        for i in range(N):
            name = 'vis_trash/fig' + str(i)
            frame = sequence[i]
            plt.imshow(frame[:, L//2, :]) #plot the image
            plt.savefig(name) #save the image
            plt.clf() #clear figure for next iteration
            
            images.append(imageio.imread(name + '.PNG')) #append image onto sequence for gif
        imageio.mimsave(filename + '.gif', images, fps = fps)
    else:
        images = []
        names = ['x', 'y', 'z']
        for i in range(N):
            
            name = 'vis_trash/fig' + str(i)
            selection = sequence[i]
            frames = [
                selection[L//2, :, :],
                selection[:, L//2, :],
                selection[:, :, L//2]
            ]
            
            for i in range(len(frames)):
                plt.subplot(1, 3, i)
                plt.imshow(frames[i])
                plt.axis('off')
                plt.title(names[i])

            plt.savefig(name) #save the image
            plt.clf() #clear figure for next iteration
            images.append(imageio.imread(name + '.PNG')) #append image onto sequence for gif
        imageio.mimsave(filename + '.gif', images, fps = fps)

def plot_loss(data, labels, marker = True, x_axis = 'Epoch', y_axis = 'Loss', title = 'Training Loss'):
    patches = []
    for i in range(len(data)):
        if marker:
            temp, = plt.plot(data[i], linestyle = '-', marker = 'o', label = labels[i])
        else :
            temp, = plt.plot(data[i], linestyle = '-', label = labels[i])
        patches.append(temp)
    plt.legend(handles = patches)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()