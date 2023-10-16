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
            image = torch.randn(batch_size, 3, 32, 32)
            for step in range(999, -1, -1):
                time = step
                pred_noise = self.model(image, time)

                alpha = self.alphas[time]
                alpha_hat = self.alpha_hats[time]
                beta = self.betas[time]
                if time > 0:
                    new_noise = torch.randn_like(image)
                else:
                    new_noise = torch.zeros_like(image)
                
                image = 1 / torch.sqrt(alpha) * (image - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * new_noise
            
        self.model.train()
        image = image[0].cpu().numpy()
        return image



    
    