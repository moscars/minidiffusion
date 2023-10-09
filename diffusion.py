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

    def forward(self, image, target_t):
        ''' 
        image: input image to be noised (3072,)
        '''
        cumulative_mean = image * np.sqrt(self.alpha_hats[target_t])
        cumulative_variance = 1 - self.alphas[target_t]
        image_t = np.random.normal(cumulative_mean, np.sqrt(cumulative_variance))
        return image_t