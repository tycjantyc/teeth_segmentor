from accelerate import Accelerator
from tqdm import tqdm
import torch as tc
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from monai.losses import DiceFocalLoss

def train_tooth_fairy(model, optimizer, train_dataloader, val_dataloader, lr_scheduler, num_epochs):

    CRITERION = DiceFocalLoss(include_background=False, sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)

    DEVICE = 'cuda:0'

    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    model = model.to(DEVICE)

    global_step = 0

    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        loss_full = 0

        #Training phase
        for step, (image, label) in enumerate(train_dataloader):
            
            clean_label = F.one_hot(label.long(), num_classes=3)
            clean_label = clean_label.permute(0, 4, 1, 2, 3).float()

            clean_images = image

            bs = clean_images.shape[0]
            clean_images = clean_images.to(DEVICE)
            clean_label = clean_label.to(DEVICE)

            with accelerator.accumulate(model):
                label_pred = model(clean_images)

                loss = CRITERION(label_pred, clean_label)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            loss_full += loss.item()

        loss_full = loss_full/len(train_dataloader)
        print(f'Epoch: {epoch + 1}, Loss: {loss_full:.7f}')

        loss_val_full = 0

        #Validation phase
        for step, (image, label) in enumerate(val_dataloader):
            
            clean_label = F.one_hot(label.long(), num_classes=3)
            clean_label = clean_label.permute(0, 4, 1, 2, 3).float()

            clean_images = image

            bs = clean_images.shape[0]
            clean_images = clean_images.to(DEVICE)
            clean_label = clean_label.to(DEVICE)

            with accelerator.accumulate(model):
                label_pred = model(clean_images)

                if tc.isnan(label_pred):
                    print("Model reutrned NaN! Abort!")
                    break

                loss = CRITERION(label_pred, clean_label)
                
                if tc.isnan(loss.item()):
                    print("Loss reutrned NaN! Abort!")
                    break

            loss_val_full += loss.item()

        if len(val_dataloader) > 0:
            loss_val_full = loss_val_full/len(val_dataloader)
            print(f'Epoch: {epoch + 1}, Validation Loss: {loss_val_full:.7f}')


