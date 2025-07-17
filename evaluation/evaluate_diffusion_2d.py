from torch import nn
import numpy as np
from skimage.util import view_as_windows
from processing.utils import load_itk, norm_standard
import torch as tc
from tqdm import tqdm
from models.diffusion_model_2d import build_diffusion_model_2d
from diffusers import DDPMScheduler, DDPMPipeline
from torchinfo import summary

patch_size = (60, 60)
stride = (30, 30)  

def get_patches_image(image, patch_size, stride):
    padded = np.pad(image, [(max(0, (patch_size[i] - image.shape[i] % patch_size[i])//2), max(0, (patch_size[i] - image.shape[i] % patch_size[i]))//2) for i in range(2)])
    windows = view_as_windows(padded, patch_size, step=stride)
    indices = [(i, j) for i in range(windows.shape[0])
                         for j in range(windows.shape[1])]
    patches = [windows[i, j] for i, j in indices]
    return patches, indices, padded.shape

def norm_diffusion(image, min=-1000, max=3000):
        return (image - min)/(max-min)

def evaluate_diffusion_2d(weights_file:str,
             file:str,
             layer_num:int, 
             noise_level:int, 
             model_input_shape: tuple[int, int], 
             stride: tuple[int, int]):
    
    volume = load_itk(file)
    image = volume[layer_num, :, :]
    
    image = image.astype(np.float32)    
    image = norm_diffusion(image)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    model = build_diffusion_model_2d()
    model.load_state_dict(tc.load(weights_file, weights_only=True))
      
    print(model)

    model = model.cuda()
    
    #noise_scheduler.set_timesteps(num_inference_steps=100)
    # pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

    h, w = model_input_shape

    patches, indices, padded_shape = get_patches_image(image, model_input_shape, stride)
    new_image = np.zeros(padded_shape)

    for patch, (i, j) in tqdm(zip(patches, indices)):
        
        input_patch = patch[np.newaxis, np.newaxis, ...]  # Adjust to your model's input shape
        input_patch_ten = tc.tensor(input_patch, dtype = tc.float32).cuda()
        noise_level = tc.tensor(noise_level, dtype = tc.int64).cuda()

        start_level = noise_level
        end_level = 0
        denoise_steps = noise_level

        input_patch = patch[np.newaxis, np.newaxis, ...]
        input_patch_ten = tc.tensor(input_patch, dtype=tc.float32).cuda()

        # print((noise_scheduler.timesteps == start_level).nonzero(as_tuple=True)[0][0])

        for step in range(denoise_steps):
            current_level = int(start_level - (start_level - end_level) * step / (denoise_steps))
            noise_level_ten = tc.tensor(current_level, dtype=tc.int64).cuda()
            
            noise_pred = model(input_patch_ten, noise_level_ten)
            input_patch_ten = noise_scheduler.step(
                noise_pred.cpu(), noise_level_ten.cpu(), input_patch_ten.cpu()
            ).prev_sample

            input_patch_ten = input_patch_ten.cuda()

        
        noise_pred = noise_pred.squeeze().detach().cpu()[0][0]  

        # Place back into volume (accounting for overlap)
        x, y = i * stride[0], j * stride[1]
        
        new_image[x:x+h, y:y+h] = input_patch_ten.detach().cpu().numpy()
        
    #Crop to original
    start = []
    end = []
    for i in range(2):
        start.append((padded_shape[i] - image.shape[i]) // 2)
        end.append(start[i] + image.shape[i])
    new_image = new_image[start[0]:end[0], start[1]:end[1]]

    return image, new_image

