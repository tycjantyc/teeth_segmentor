import torch.nn as nn

class L1L2Loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + self.beta * self.l2(pred, target)
