from models.diffusion_model_2d import build_diffusion_model_2d
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets.diffusion_dataset_2d import Diffusion_Dataset_2D
from training.train_diffusion import train_loop_diffusion

if __name__ == '__main__':
    
    BATCH_SIZE = 64

    dataset = Diffusion_Dataset_2D()
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False, prefetch_factor=None)

    learning_rate = 1e-04
    weight_decay = 1e-06

    num_epochs = 10
    lr_warmup_steps = 500

    model = build_diffusion_model_2d()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    train_loop_diffusion(model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, num_epochs, noise_type='gaussian')





