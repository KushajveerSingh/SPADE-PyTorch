# When I was testing this function, I was encountering a bug
# where the script would exit and no plot would be shown. So
# if you also encounter such a problem when running the script
# from the terminal use -i with the python command as 
# pyhton -i my_script.py

import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torchvision.utils import make_grid

# Plot the original images
def imshow(image):
    # Here img is the output of the CityScapesDataset dataloader
    image = image.numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.pause(0.001)

def show_plots(img, seg, data_loader):
    # Here img and seg are output of dataloader created by CityScapesDataset
    segmap = torch.zeros(seg.size(0), 3, seg.size(2), seg.size(3))
    for i, a in enumerate(seg):
        image = a.squeeze()
        image = image.numpy()
        image = data_loader.decode_segmap(image)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        segmap[i] = image
    
    img_seg = torch.cat((img, segmap), dim=0)
    grid_img = make_grid(img_seg, nrow=4)
    imshow(grid_img)

# Show plots of G_losses and D_losses
def show_gen_dis_plots(G_losses, D_losses):
    # These losses are same as that contained in the train script
    g_losses = [x.item() for x in G_losses]
    d_losses = [x.item() for x in D_losses]

    plt.figure(figsize=(20,10))
    plt.title('Generator and Discriminator loss during training')
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def show_generated_images(img_list, args):
    # img_list is same as the one contained in the train_sript
    h, w = args.img_size
    b = args.batch_size
    temp = np.zeros((len(img_list)*b, 3, h, w))
    j = 0
    for i in range(len(img_list)):
        a = img_list[i]
        temp[j:j+b] = a 
        j += b 
    
    temp = torch.from_numpy(temp)
    grid_img = make_grid(temp, nrow=b)
    imshow(grid_img)