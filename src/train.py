import numpy as np 
from tqdm import tqdm

import torch 
import torch.nn as nn

from args import get_parser
from dataloader.cityscapes import CityScapesDataset
from models.generator import SPADEGenerator
from models.discriminator import SPADEDiscriminator
from models.ganloss import GANLoss
from models.weights_init import weights_init

def train(args):
    # Get the data
    path = args.path 
    dataset = {
        x: CityScapesDataset(path, split=x, is_transform=True, img_size=args.img_size) for x in ['train', 'val']
    }
    data = {
        x: torch.utils.data.DataLoader(dataset[x],
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     drop_last=True) for x in ['train', 'val']
    }

    epochs = args.epochs
    lr_gen = args.lr_gen
    lr_dis = args.lr_dis
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        raise Exception('GPU not available')
    torch.backends.cudnn.benchmark = True

    gen = SPADEGenerator(args)
    dis = SPADEDiscriminator()

    gen = gen.to(device)
    dis = dis.to(device)

    noise = torch.rand(4, 256)
    noise = noise.to(device)

    criterion = GANLoss()

    gen.apply(weights_init)
    dis.apply(weights_init)

    optim_gen = torch.optim.Adam(gen.parameters(), lr=lr_gen, betas=(0, 0.999))
    optim_dis = torch.optim.Adam(dis.parameters(), lr=lr_dis, betas=(0, 0.999))

    img_lists = []
    G_losses = []
    D_losses = []

    # The training loop
    for epoch in tqdm(range(epochs)):
        print(f'Epoch {epoch+1}/{epochs}')
        for i, (img, seg) in enumerate(data['train']):
            img = img.to(device)
            seg = seg.to(device)
            
            fake_img = gen(noise, seg)
            
            # Fake Detection and Loss
            pred_fake = dis(fake_img, seg)
            loss_D_fake = criterion(pred_fake, False)
            
            # Real Detection and Loss
            pred_real = dis(img, seg)
            loss_D_real = criterion(pred_real, True)
            
            loss_G = criterion(pred_fake, True)
            loss_D = loss_D_fake + loss_D_real*0.5
            
            # Backprop
            optim_gen.zero_grad()
            loss_G.backward(retain_graph=True)
            optim_gen.step()
            
            optim_dis.zero_grad()
            loss_D.backward()
            optim_dis.step()
            
            G_losses.append(loss_G.detach().cpu())
            D_losses.append(loss_D.detach().cpu())
            
            if i%200 == 0:
                print("Iteration {}/{} started".format(i+1, len(data['train'])))
        
        print()
        if epoch%20 == 0:
            with torch.no_grad():
                img_lists.append(fake_img.detach().cpu().numpy())

    torch.save(gen, 'gen.pth')
    torch.save(gen, 'dis.pth')

if __name__ == "__main__":
    # Parse the arguments
    parser = get_parser()
    args = parser.parse_args()

    if args.gen_hidden_size%16 != 0:
        print('hidden-size should be multiple of 16. It is based on paper where first input', end=" ") 
        print('to SPADE is (4,4) in height and width. You can change this defualt in args.py')
        exit()

    args.img_size = tuple(args.img_size)

    train(args)