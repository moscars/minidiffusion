import sys
sys.path.extend(['..', "../.."])

import numpy as np
from diffusion import Diffusion
from dataset import TorchDataset
from vae import VariationalAutoEncoder
from utils import *
from torch import optim, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import time
from tmp import Autoencoder

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_div

def train(vae, lr, num_epochs, dataset, batch_size):
    longLoss = None

    print(f"Number of parameters: {vae.get_num_params()}")
    # load from state
    vae.to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=lr)

    # diffusion.model.load_state_dict(torch.load('model_160.pt', map_location=device))
    # optimizer.load_state_dict(torch.load('optimizer_160.pt', map_location=device))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    vae.train()

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        # start by training on one at a time
        num_batches = len(dataset) // batch_size

        for i, image in enumerate(train_loader):
            image = image.to(device)
            decoded, distribution = vae(image)

            mu = distribution.mean
            logvar = distribution.log_var

            loss = vae_loss(decoded, image, mu, logvar)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            normLoss = loss.item() / (batch_size * 3 * 96 * 96)

            if longLoss is None:
                longLoss = normLoss
            else:
                longLoss = 0.995 * longLoss + 0.005 * normLoss
            
            if i % 2 == 0:
                print(f"Step {i} of {num_batches} Long: {round(longLoss, 6)}, Current: {round(normLoss, 6)}")

        if epoch % 5 == 0:
            show_4_images(image[:4].cpu().detach().numpy(), save=True, name=f"{epoch}_original")
            show_4_images(decoded[:4].cpu().detach().numpy(), save=True, name=f"{epoch}_decoded")
            if epoch > 30:
                torch.save(vae.state_dict(), f"VAE_model_{epoch}.pt")
                torch.save(optimizer.state_dict(), f"Optimizer_{epoch}.pt")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    start = time.time()
    classes = set([2, 3, 5, 9])

    images = []
    labels = []
    read_binary_data('../../stl10_binary/train_X.bin', '../../stl10_binary/train_Y.bin', images, labels, classes)
    images = np.array([normalize(squeeze01(x)) for x in images])

    images = torch.tensor(images, dtype=torch.float32)
    dataset = TorchDataset(images)

    print(f"Length of dataset: {len(dataset)}")

    vae = Autoencoder()
    train(vae, 1e-4, 500, dataset, batch_size=32)

    end = time.time()
    print(f"Total time: {end - start}")