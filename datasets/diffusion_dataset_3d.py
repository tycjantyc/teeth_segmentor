import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import remove_rings_artifacts, load_itk, norm_standard, norm_to_0_1, crop
from datasets.datasize_factory import DataSize_Factory

class Diffusion_Dataset_3D(Dataset):
    
    def __init__(self, name_of_dataset = 'agh', data_directory:str = '/../data/raw_data', remove_ct_rings:bool = False, clamp:tuple([int, int]) = None):
        
        self.name_of_dataset = name_of_dataset
        self.data_directory = data_directory
        self.remove_ct_rings = remove_ct_rings
        self.clamp = clamp

        self.lista = []
       
        folders = os.listdir(self.data_directory) 
        for num in folders:
            path = os.path.join(data_directory, num, 'image.nii.gz')
            self.lista.append(path)

        print("Calculating maximal size of data . . .")
        self.max_size = DataSize_Factory(dataset_name=self.name_of_dataset).calucalte_size(self.lista)
        self.h, self.w, self.d = self.max_size    

        print(f"Dimensions: {self.h}, {self.w}, {self.d}")        
                    
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        path = self.lista[index]
        image = load_itk(path)

        if self.remove_ct_rings:
            image = remove_rings_artifacts(image)

        image = image.astype(np.float32)
        image, (mean, std) = norm_standard(image)
        image = crop(image, self.max_size)

        return image