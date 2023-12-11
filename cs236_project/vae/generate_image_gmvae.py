# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.gmvae import GMVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=128,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=40000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
parser.add_argument('--overwrite', type=int, default=0,     help="Flag for overwriting")
parser.add_argument('--k', type=int, default=2,     help="k parameter")
args = parser.parse_args()
layout = [
    ('model={:s}',  'gmvae'),
    ('z={:02d}',  args.z),
    ('k={:03d}', args.k),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)
######################
character = "_Kirby"
#######################
model_name += character
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = GMVAE(z_dim=args.z, k=args.k, name=model_name).to(device)
pixels = vae.get_num_pixels()
print("numb pixels: ", pixels)

#ut.load_model_by_name(vae, global_step=args.iter_max, device=device)
ut.load_model_by_name(vae, global_step=args.iter_save, device=device)
for i in range(20):
    samps = vae.sample_x(1)
    #print(samps)

    #grid_samples = samps.reshape(3, 128, 128)
    grid_samples = samps.reshape(3, pixels, pixels)
    grid_samples = grid_samples.permute(1, 2, 0)
    grid_samples_np = grid_samples.detach().cpu().numpy()
    #grid_samples_np = np.clip(grid_samples_np, 0, 1) 


    plt.imshow(grid_samples_np)  # 'gray' colormap for grayscale images
    plt.title('Samples')
    plt.show()

