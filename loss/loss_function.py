import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=2.0, smooth=1.0):
        """
        :param alpha: Weight for Dice loss
        :param beta: Weight for Focal loss
        :param gamma: Focusing parameter for Focal loss
        :param smooth: Smoothing for Dice loss
        """
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets):
        # BCE loss
        bce_loss = self.bce(inputs, targets)

        # Dice loss
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth)

        # Focal loss
        bce_exp = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_exp)
        focal_loss = ((1 - pt) ** self.gamma * bce_exp).mean()

        # Combine all
        loss = bce_loss + self.alpha * dice_loss + self.beta * focal_loss
        return loss
