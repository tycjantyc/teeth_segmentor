import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import remove_rings_artifacts, load_itk, norm_standard, norm_to_0_1, crop

class Diffusion_Dataset_2D(Dataset):
    
    def __init__(self, data_directory:str = 'C:/Users/Jan/Desktop/SuperZebySegmentacja/data/diffusion_2d/images', remove_ct_rings:bool = False):
        
        self.data_directory = data_directory
        self.remove_ct_rings = remove_ct_rings
        
        self.lista = []
       
        folders = os.listdir(self.data_directory) 
        for num in folders:
            path = os.path.join(data_directory, num)
            self.lista.append(path)

                             
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        path = self.lista[index]
        image = np.load(path)['image']

        if self.remove_ct_rings:
            image = remove_rings_artifacts(image)

        image = image.astype(np.float32)
        image = norm_standard(image)

        return image