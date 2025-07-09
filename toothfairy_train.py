import os
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets.tooth_fairy_3_dataset import ToothFairy3_Dataset
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
from training.train_toothfairy import train_tooth_fairy
from processing.tooth_fairy_classes import compression_factory
from processing.utils import norm_factory
import argparse

def main(data_path = 'D:/ToothFairy3', 
         batch_size = 1, 
         min_clamp = -1000, 
         max_clamp = 3000, 
         compression = 'big',  #'big' - 3 classes, 'medium' - 8 classes, 'none' - 77 classes
         normalization = '01', #'standard', '01', 'none'
         augmentation = True, 
         learning_rate = 1e-2, 
         weight_decay = 1e-5, 
         num_epochs = 10,
         lr_warmup_steps = 40,
         freeze = False,        # for the pretrained encoder to be frozen
         channels = ['sobel'],  # Choose from: 'sobel', 'clahe', 'median'
         validation_split = 0.05): 

    comp_func, num_classes = compression_factory(compression)
    norm_func = norm_factory(normalization)

    dataset_train = ToothFairy3_Dataset(data_directory=data_path, input_size=(96, 96, 96), remove_ct_rings=False, clamp = (min_clamp, max_clamp), compression_function = comp_func, normalization=norm_func,augmentation=augmentation, channels=channels, validation_split=validation_split)
    dataset_val = ToothFairy3_Dataset(data_directory=data_path, input_size=(96, 96, 96), remove_ct_rings=False, clamp = (min_clamp, max_clamp), compression_function = comp_func, normalization=norm_func, augmentation=False, channels=channels, validation_split=validation_split, validation_mode=True)
    
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False, prefetch_factor=None)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False, prefetch_factor=None)

    inchannels = 1 + len(channels)

    model = build_hyspark_tooth_fairy_1(in_channel=inchannels, n_classes=num_classes, img_size=96, freeze=freeze)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    train_tooth_fairy(model, optimizer, train_dataloader, val_dataloader, lr_scheduler, num_epochs)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, required=False, default='D:/ToothFairy3')
    arg_parser.add_argument('--num_epochs', type=int, required=False, default=10)
    arg_parser.add_argument('--learning_rate', type=float, required=False, default=1e-3)
    arg_parser.add_argument('--weight_decay', type=float, required=False, default=1e-6)
    arg_parser.add_argument('--batch_size', type=int, required=False, default=1)
    arg_parser.add_argument('--compression', type=str, required=False, default='big', choices=['big', 'medium', 'none'])
    arg_parser.add_argument('--normalization', type=str, required=False, default='01', choices=['standard', '01', 'none'])
    arg_parser.add_argument('--channels', type=list, required=False, default=['sobel'])
    arg_parser.add_argument('--augmentation', type=bool, required=False, default=True)
    arg_parser.add_argument('--freeze', type=bool, required=False, default=False)
    args = arg_parser.parse_args()

    main(data_path=args.data_dir,
         num_epochs=args.num_epochs,
         learning_rate=args.learning_rate,
         weight_decay=args.weight_decay,
         batch_size=args.batch_size,
         compression=args.compression,
         normalization=args.normalization,
         channels=args.channels,
         augmentation=args.augmentation, 
         freeze=args.freeze
         )


