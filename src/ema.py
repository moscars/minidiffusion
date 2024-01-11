import copy
import torch
import os

class ExponentialMovingAverage:
    def __init__(self, alpha, model):
        self.alpha = alpha
        self.model = model # pointer to the model
        self.shadow = {}
        self.iter = 0
    
    def update(self):
        if self.iter < 1000:
            self.iter += 1
            self.copyModel()
            return
    
        if self.iter == 1000:
            print("EMA Model is starting to track!")
            self.iter += 1
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                assert name in self.shadow
                new_average = self.alpha * self.shadow[name] + (1.0 - self.alpha) * param.data
                self.shadow[name] = new_average.clone()

    def saveEMAModel(self, path):
        torch.save(self.shadow, path)

    def loadModel(self, path):
        if os.path.isfile(path):
            self.shadow = torch.load(path)
            print(f"EMA model state loaded from {path}")
        else:
            raise FileNotFoundError(f"No file found at {path}")

    def copyModel(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.shadow[name] = param.data.clone()
    
    def getEMAModel(self):
        with torch.no_grad():
            fullModel = copy.deepcopy(self.model)
            for name, param in fullModel.named_parameters():
                param.data = self.shadow[name]
            return fullModel
    