import numpy as np
from unet import UNet
import torch
from torch import optim, nn

class Diffusion:
    def __init__(self, device, num_classes):
        # prepare for forward noising
        self.device = device
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.noising_steps = 1000

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.noising_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        self.alpha_hats = self.alpha_hats.to(device)

        self.model = UNet(device=device, num_classes=num_classes)
        
    def noise_image(self, image, target_t):
        '''
        image: (batch_size, 3, 32, 32)
        target_t: (batch_size,)
        '''

        orignal_weight = torch.sqrt(self.alpha_hats[target_t])
        noise_weight = torch.sqrt(1 - self.alpha_hats[target_t])
        noise = torch.randn(image.shape)
        noise = noise.to(self.device)

        orignal_weight = orignal_weight.view(-1, 1, 1, 1)
        noise_weight = noise_weight.view(-1, 1, 1, 1)

        image_t = orignal_weight * image + noise_weight * noise
        return image_t, noise
    
    def generate(self, batch_size, class_to_gen=None):
        self.model.eval()
        with torch.no_grad():
            image = torch.randn(batch_size, 3, 32, 32).to(self.device)
            for step in range(999, -1, -1):
                time = step
                uncond_pred_noise = self.model(image, time)

                if class_to_gen is not None:
                    cond_pred_noise = self.model(image, time, class_to_gen)
                    # CFG formula with scale factor 3
                    pred_noise = uncond_pred_noise + (cond_pred_noise - uncond_pred_noise) * 3
                else:
                    pred_noise = uncond_pred_noise

                alpha = self.alphas[time]
                alpha_hat = self.alpha_hats[time]
                beta = self.betas[time]
                if time > 0:
                    new_noise = torch.randn_like(image)
                else:
                    new_noise = torch.zeros_like(image)

                image = 1 / torch.sqrt(alpha) * (image - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * new_noise
            
        self.model.train()

        return image.cpu().numpy()



    
    