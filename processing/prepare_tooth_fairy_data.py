import numpy as np
from processing.utils import remove_rings_artifacts, create_random_snippet
from processing.data_augmentation import AugmentationFactory
from processing.multi_input import MultiInputFactory

def prepare_toothfairy(image: np.ndarray, 
                       mask: np.ndarray, 
                       remove_ct_rings:bool, 
                       input_size: tuple[int, int, int], 
                       compression_function, 
                       normalization_function, 
                       augmentation:bool, 
                       channels:list):
    
        if remove_ct_rings:
            image = remove_rings_artifacts(image)

        image = image.astype(np.float32)

        assert image.shape == mask.shape

        image, mask = create_random_snippet(image, mask, input_size)

        mask = compression_function(mask)
        image = normalization_function(image)

        if augmentation:
            aug_factory = AugmentationFactory()
            image, mask = aug_factory.create(image, mask)

        multi_input = MultiInputFactory(channels)
        image = multi_input.create(image)

        return image, mask