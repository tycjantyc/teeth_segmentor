from accelerate import Accelerator
import torch
from tqdm import tqdm
import torch.nn.functional as F
from loss.l1l2loss import L1L2Loss


def train_loop_diffusion(model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, num_epochs, noise_type = 'gaussian'):
    
    DEVICE = 'cuda:0'
    CRITERION = L1L2Loss()

    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    #model.load_state_dict(torch.load('models/weights/weights_diffusion_2d_2.pt', weights_only=True))

    model = model.to(DEVICE)

    loss_global = 10
    global_step = 0

    # Now you train the model
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        loss_full = 0

        for step, batch in enumerate(train_dataloader):
            clean_images = batch.unsqueeze(1)
            
            if noise_type == 'gaussian':
                noise = torch.randn(clean_images.shape, device=clean_images.device)

            bs = clean_images.shape[0]
            clean_images = clean_images.to(DEVICE)

            timesteps = torch.randint(
                0, 1000, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps)
                loss = CRITERION(noise_pred, noise)
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

        if loss_full < loss_global:
            loss_global = loss_full
            torch.save(model.state_dict(), f'models/weights/weights_diffusion_2d_bigger_{epoch}.pt')


