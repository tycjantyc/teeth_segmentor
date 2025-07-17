import os
import numpy as np
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing.utils import load_itk # , create_random_snippet

def create_random_snippet(image: np.ndarray, mask:np.ndarray, input_size: tuple[int, int, int]):
        h, w, d = image.shape
        h1, w1, d1 = input_size

        if h < h1 or w < w1 or d < d1:
            return ValueError('Input size is too big!')
        
        while True:
            h_new = np.random.randint(0, h)
            w_new = np.random.randint(0, w-w1)
            d_new = np.random.randint(0, d-d1)

            temp_mask = mask[h_new, w_new:w_new+w1, d_new:d_new+d1]

            if np.all(temp_mask == 0):   
                #print('ok')
                continue

            image = image[h_new, w_new:w_new+w1, d_new:d_new+d1]
            return image, temp_mask

def create_diffusion_data(path_from:str = 'D:/ToothFairy3', path_to:str = '../diffusion_2d', resolution:tuple[int, int] = (60, 60), images_per_volume:int = 400):

    PATH = path_from + '/imagesTr'
    PATH_MASK = path_from + '/labelsTr'

    for idx, (p_image, p_mask) in enumerate(zip(os.listdir(PATH), os.listdir(PATH_MASK))):
        
        p_image = os.path.join(PATH, p_image)
        p_mask = os.path.join(PATH_MASK, p_mask)

        volume = load_itk(filename=p_image)
        mask = load_itk(filename=p_mask)

        print(p_image)
        print(np.unique(mask))

        h, w, d = volume.shape

        for i in tqdm(range(images_per_volume)):

            temp_image, temp_mask = create_random_snippet(volume, mask, input_size=(1, resolution[0], resolution[1]))
            full_path_to = os.path.join(path_to, f"image_{idx}_{i}.npz")
            np.savez(full_path_to, image=temp_image, spacing=[1.0, 1.0, 1.0])

if __name__ == "__main__":
    create_diffusion_data()