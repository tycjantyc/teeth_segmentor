import torch as tc
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers import UNet3DConditionModel

def prepare_diffusion_model(path_to_weights = None):

    model = UNet3DConditionModel(
        sample_size=64,
        in_channels=1,   
        out_channels=1,  
        down_block_types=(
            "DownBlock3D", "DownBlock3D", "CrossAttnDownBlock3D"
        ),
        up_block_types=(
            "CrossAttnUpBlock3D", "UpBlock3D", "UpBlock3D"
        ),
        block_out_channels=(32, 64, 96),
        attention_head_dim=8,
        cross_attention_dim=1  # Dimension of conditional embedding (e.g., text or CLIP)
    )

    if path_to_weights is not None:
        model.load_state_dict(tc.load(path_to_weights, weights_only=True))
    
    return model