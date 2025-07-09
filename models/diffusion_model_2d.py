import torch.nn as nn
from diffusers import UNet2DModel


class UNetWithInputPadding(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial layer to pad image from 60x60 to 64x64
        self.expand = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=5,
            stride=1,
            padding=4  
        )

        self.contract = nn.Conv2d(
            in_channels=4,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=0  
        )

        self.activ = nn.GELU()
      
        self.unet = UNet2DModel(
            sample_size=64,  
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(8, 16, 32, 64), 
            down_block_types=(
                "DownBlock2D",  
                "DownBlock2D",
                "AttnDownBlock2D",  
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  
                "AttnUpBlock2D",  
                "UpBlock2D",
                "UpBlock2D",
            ),
            norm_num_groups=8
        )

    def forward(self, x, timestep):
        x = self.activ(self.expand(x))
        x = self.unet(x, timestep)
        return self.contract(x.sample)

def build_diffusion_model_2d():

    return UNetWithInputPadding()
