import numpy as np
import cv2
from scipy.ndimage import sobel
from scipy.ndimage import median_filter

def norm(image):
    image = (image - image.min())/(image.max()- image.min())
    return image

def choose_filter(volume: np.ndarray, channels) -> np.ndarray:
    
    # Channel 1: 3D Sobel Filter
    if 'sobel' in channels:
        dx = sobel(volume, axis=2)  
        dy = sobel(volume, axis=0)  
        dz = sobel(volume, axis=1)  
        sobel_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        return norm(sobel_3d).astype(np.float32)

    # Channel 2: CLAHE (per depth slice)
    elif 'clahe' in channels:
        clahe_input = cv2.normalize(volume, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_volume = np.zeros_like(volume)
        for h in range(volume.shape[0]):
            slice_2d = clahe_input[h, :, :]
            clahe_slice = clahe.apply(slice_2d)
            clahe_volume[h, :, :] = clahe_slice
        return norm(clahe_volume).astype(np.float32)

    # Channel 3: Median filter
    elif 'median' in channels:
        filtered = median_filter(volume, size=3)
        return norm(filtered).astype(np.float32)
    
    else:
        raise ValueError('There is no such channel!')


class MultiInputFactory():
    
    def __init__(self, channels = []):

        self.channels = channels

    def create(self, volume):

        dim = len(self.channels) + 1
        output = np.zeros((dim, *volume.shape), dtype=np.float32)
        
        output[0, ...] = norm(volume)

        for idx, channel in enumerate(self.channels):
            output[idx+1, ...] = choose_filter(volume, channel)

        return output




        





