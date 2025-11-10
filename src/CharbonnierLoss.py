import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return torch.mean(loss)