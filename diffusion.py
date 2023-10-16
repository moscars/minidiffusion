import numpy as np
from unet import UNet
import torch
from torch import optim, nn

class Diffusion:
    def __init__(self):
        # prepare for forward noising
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.noising_steps = 1000

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.noising_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

        self.model = UNet()

    def noise_image(self, image, target_t):
        ''' 
        image: input image to be noised (3072,)
        '''
        orignal_weight = torch.sqrt(self.alpha_hats[target_t])
        noise_weight = torch.sqrt(1 - self.alphas[target_t])
        noise = torch.randn(image.shape)
        image_t = orignal_weight * image + noise_weight * noise
        return image_t, noise
    
    def generate(self, batch_size):
        self.model.eval()
        with torch.no_grad():
            randomNoise = torch.randn(batch_size, 3, 32, 32)


    
    