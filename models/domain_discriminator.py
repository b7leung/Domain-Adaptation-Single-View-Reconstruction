
import torch
import torch.nn.functional as F
from utils.gradient_reversal_module import GradientReversal

class Classifier(torch.nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        #super(Classifier, self).__init__()
        self.num_classes = num_classes
        # Layer Definitions
        #self.fc1 = torch.nn.Linear(256*8*8, self.num_classes)
        self.fc1 = torch.nn.Linear(256*8*8, 1024)
        self.fc2 = torch.nn.Linear(1024,512)
        self.fc3 = torch.nn.Linear(512,512)
        self.fc4 = torch.nn.Linear(512, self.num_classes)