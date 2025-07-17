#import tomopy      <- problem with this library!!!
import numpy as np
import SimpleITK as sitk

def load_itk(filename:str, clamp:tuple[int, int] = None):
    """
    For teeth and canals often clamp is (500, 3000)
    """
    itkimage = sitk.ReadImage(filename)
    if clamp is not None:
        itkimage = sitk.Clamp(itkimage, lowerBound=clamp[0], upperBound=clamp[1])

    ct_scan = sitk.GetArrayFromImage(itkimage)
    return ct_scan

def remove_rings_artifacts(image: np.ndarray):  #slightly changes the image but worth testing
    image_without_rings = tomopy.misc.corr.remove_ring(image,             
                                        center_x=image.shape[1]/2,
                                        center_y=image.shape[2]/2,
                                        thresh=100)
    
    return image_without_rings


def norm_to_0_1(image: np.ndarray, min = -1000, max = 3000):
    return (image - min)/(max - min)

def norm_standard(image: np.ndarray):
    mean = image.mean()
    standard_deviation = image.std() + 1e-5
    normal_image = (image - mean)/standard_deviation

    return normal_image

def norm_factory(mode_name = 'none'):
    if mode_name == 'none':
        return lambda x:x
    
    elif mode_name == 'standard':
        return norm_standard
    
    elif mode_name == '01':
        return norm_to_0_1
    
    else:
        raise ValueError('There is no such mode!')


def crop(image: np.ndarray, new_dims: list[int, int, int]):

    h, w, d = image.shape
    h1, w1, d1 = new_dims

    x1 = int((h-h1)/2)
    y1 = h - x1

    x2 = int((w-w1)/2)
    y2 = w - x2

    x3 = int((d-d1)/2)
    y3 = d - x3

    return image[x1:y1, x2:y2, x3:y3]

def create_random_snippet(image: np.ndarray, mask:np.ndarray, input_size: tuple[int, int, int]):
        h, w, d = image.shape
        h1, w1, d1 = input_size

        if h < h1 or w < w1 or d < d1:
            return ValueError('Input size is too big!')
        
        while True:
            h_new = np.random.randint(0, h-h1)
            w_new = np.random.randint(0, w-w1)
            d_new = np.random.randint(0, d-d1)

            temp_mask = mask[h_new:h_new+h1, w_new:w_new+w1, d_new:d_new+d1]

            if np.all(mask == 0):   #prevents from choosing meaningless masks
                continue

            temp_image = image[h_new:h_new+h1, w_new:w_new+w1, d_new:d_new+d1]
            return temp_image, temp_mask
            

        