from hyspark.models.network.hyspark_model import build_hybird, UnetOutBlock
import torch
import torch.nn as nn
import os

def build_hyspark_tooth_fairy_1(in_channel = 1, n_classes = 77, img_size = 96, freeze = True):

    #C:/Users/Jan/Desktop/teeth_segmentor
    wPATH = "/app/hyspark/ckpt/hybird_ct_pretrained_timm_style_mask75.pth"

    model = build_hybird(in_channel=1, n_classes=14, img_size=96)
    model.load_state_dict(torch.load(wPATH, weights_only=True), strict = False)

    if freeze:
        for param1, param2 in zip(model.mae.parameters(), model.embeddings.parameters()):
            param1.requires_grad = False
            param2.requires_grad = False

    model.decoder.out = UnetOutBlock(in_channels=model.decoder.out.in_channels, n_classes=n_classes)
    
    if in_channel != 1:
        model.embeddings.stem = nn.Conv3d(in_channels=in_channel, out_channels=model.embeddings.channels[0], kernel_size=3, stride=1, padding=1)
        for param in model.embeddings.stem.parameters():
            param.requires_grad = True

    return model


