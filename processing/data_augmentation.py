import torchio as tio
import numpy as np
import torch

class AugmentationFactory():
    def __init__(self):
        self.transform = tio.Compose([
                                        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.3)
                                        # tio.RandomAffine(
                                        #     scales=(0.95, 1.05),
                                        #     degrees=5,
                                        #     translation=0,
                                        #     center='image',
                                        #     default_pad_value='mean',
                                        #     p=0.5
                                        # ), 
                                        #tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.3)
                                    ])

    def create(self, img_np, mask_np):

        # Convert to TorchIO Subject (add channel dim)
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(img_np[None, ...])),
            mask=tio.LabelMap(tensor=torch.from_numpy(mask_np[None,...]))
        )

        # Apply transform
        transformed = self.transform(subject)

        # Convert back to NumPy (remove channel dim and convert to float32)
        img_aug = transformed.image.numpy().squeeze().astype(np.float32)
        mask_aug = transformed.mask.numpy().squeeze().astype(np.float32)

        return img_aug, mask_aug
