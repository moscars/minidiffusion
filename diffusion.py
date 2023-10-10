import numpy as np

class Diffusion:
    def __init__(self):
        # prepare for forward noising
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.noising_steps = 1000

        self.betas = np.linspace(self.beta_start, self.beta_end, self.noising_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = np.cumprod(self.alphas)

    def noise_image(self, image, target_t):
        ''' 
        image: input image to be noised (3072,)
        '''
        orignal_weight = np.sqrt(self.alpha_hats[target_t])
        noise_weight = np.sqrt(1 - self.alphas[target_t])
        noise = np.random.normal(0, 1, image.shape)
        image_t = orignal_weight * image + noise_weight * noise
        return image_t
    
    