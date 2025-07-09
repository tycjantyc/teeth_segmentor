import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import load_itk, norm_to_0_1
from processing.tooth_fairy_classes import fix_tooth_fairy_classes
from processing.prepare_tooth_fairy_data import prepare_toothfairy


class ToothFairy3_Dataset(Dataset):
    
    def __init__(self, data_directory:str = 'D:/ToothFairy3',
                        input_size = (96, 96, 96), 
                        remove_ct_rings:bool = False, 
                        clamp:tuple[int, int] = None, 
                        compression_function = fix_tooth_fairy_classes,
                        normalization = norm_to_0_1,
                        augmentation = False,
                        channels = [], 
                        num_samples:int = 4, 
                        validation_split = 0.0, 
                        validation_mode = False
                    ):
        
        self.data_directory = data_directory
        self.remove_ct_rings = remove_ct_rings
        self.clamp = clamp
        self.input_size = input_size
        self.compression_function = compression_function
        self.normalization = normalization
        self.augmentation = augmentation
        self.channels = channels
        self.num_samples = num_samples
        self.validation_split = validation_split
        self.validation_mode = validation_mode

        self.lista = []

        self.path_cbct = os.path.join(self.data_directory, 'imagesTr')
        self.path_labels = os.path.join(self.data_directory, 'labelsTr')
       
        files_cbct = os.listdir(self.path_cbct)
        files_labels = os.listdir(self.path_labels)
         
        for file_cbct, file_label in zip(files_cbct, files_labels):
            
            file_cbct = os.path.join(self.path_cbct, file_cbct)
            file_label = os.path.join(self.path_labels, file_label)
            for i in range(self.num_samples):
                self.lista.append((file_cbct, file_label))

        split_index = int(len(self.lista) * self.validation_split)
        
        if validation_mode:
            self.lista = self.lista[:split_index]
        else:
            self.lista = self.lista[split_index:]
             
                    
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_label = paths

        image = load_itk(path_cbct, clamp = self.clamp)
        mask = load_itk(path_label)

        image = self.normalization(image)
        image, mask = prepare_toothfairy(image, mask, self.remove_ct_rings, self.input_size, self.compression_function, self.normalization, self.augmentation, self.channels)

        return image, mask
    

