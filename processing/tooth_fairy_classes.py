classes = {
    "background": 0,
    "Lower Jawbone": 1,
    "Upper Jawbone": 2,
    "Left Inferior Alveolar Canal": 3,
    "Right Inferior Alveolar Canal": 4,
    "Left Maxillary Sinus": 5,
    "Right Maxillary Sinus": 6,
    "Pharynx": 7,
    "Bridge": 8,
    "Crown": 9,
    "Implant": 10,
    "Upper Right Central Incisor": 11,
    "Upper Right Lateral Incisor": 12,
    "Upper Right Canine": 13,
    "Upper Right First Premolar": 14,
    "Upper Right Second Premolar": 15,
    "Upper Right First Molar": 16,
    "Upper Right Second Molar": 17,
    "Upper Right Third Molar (Wisdom Tooth)": 18,
    "Upper Left Central Incisor": 21,
    "Upper Left Lateral Incisor": 22,
    "Upper Left Canine": 23,
    "Upper Left First Premolar": 24,
    "Upper Left Second Premolar": 25,
    "Upper Left First Molar": 26,
    "Upper Left Second Molar": 27,
    "Upper Left Third Molar (Wisdom Tooth)": 28,
    "Lower Left Central Incisor": 31,
    "Lower Left Lateral Incisor": 32,
    "Lower Left Canine": 33,
    "Lower Left First Premolar": 34,
    "Lower Left Second Premolar": 35,
    "Lower Left First Molar": 36,
    "Lower Left Second Molar": 37,
    "Lower Left Third Molar (Wisdom Tooth)": 38,
    "Lower Right Central Incisor": 41,
    "Lower Right Lateral Incisor": 42,
    "Lower Right Canine": 43,
    "Lower Right First Premolar": 44,
    "Lower Right Second Premolar": 45,
    "Lower Right First Molar": 46,
    "Lower Right Second Molar": 47,
    "Lower Right Third Molar (Wisdom Tooth)": 48,
    "Left Mandibular Incisive Canal": 103,
    "Right Mandibular Incisive Canal": 104,
    "Lingual Canal": 105,
    "Upper Right Central Incisor Pulp": 111,
    "Upper Right Lateral Incisor Pulp": 112,
    "Upper Right Canine Pulp": 113,
    "Upper Right First Premolar Pulp": 114,
    "Upper Right Second Premolar Pulp": 115,
    "Upper Right First Molar Pulp": 116,
    "Upper Right Second Molar Pulp": 117,
    "Upper Right Third Molar (Wisdom Tooth) Pulp": 118,
    "Upper Left Central Incisor Pulp": 121,
    "Upper Left Lateral Incisor Pulp": 122,
    "Upper Left Canine Pulp": 123,
    "Upper Left First Premolar Pulp": 124,
    "Upper Left Second Premolar Pulp": 125,
    "Upper Left First Molar Pulp": 126,
    "Upper Left Second Molar Pulp": 127,
    "Upper Left Third Molar (Wisdom Tooth) Pulp": 128,
    "Lower Left Central Incisor Pulp": 131,
    "Lower Left Lateral Incisor Pulp": 132,
    "Lower Left Canine Pulp": 133,
    "Lower Left First Premolar Pulp": 134,
    "Lower Left Second Premolar Pulp": 135,
    "Lower Left First Molar Pulp": 136,
    "Lower Left Second Molar Pulp": 137,
    "Lower Left Third Molar (Wisdom Tooth) Pulp": 138,
    "Lower Right Central Incisor Pulp": 141,
    "Lower Right Lateral Incisor Pulp": 142,
    "Lower Right Canine Pulp": 143,
    "Lower Right First Premolar Pulp": 144,
    "Lower Right Second Premolar Pulp": 145,
    "Lower Right First Molar Pulp": 146,
    "Lower Right Second Molar Pulp": 147,
    "Lower Right Third Molar (Wisdom Tooth) Pulp": 148
}

new_classes = dict()

for num, key in enumerate(classes.keys()):
    new_classes[key] = num


def fix_tooth_fairy_classes(volume):
    volume_new = volume.copy()  

    for label_name, old_class in classes.items():
        new_class = new_classes[label_name]
        volume_new[volume == old_class] = new_class

    return volume_new



def compress_tooth_fairy_classes(volume):
    """
    Kompresja:
    Klasa 0 - background ([0])
    Klasa 1 - Kości szczęki górne dolne([1, 2])
    Klasa 2 - Gardło ([7])
    Klasa 3 - Jamy nosowe ([5, 6])
    Klasa 4 - Nerwy w szczęce ([3, 4, 103:105)
    Klasa 5 - Ciała obce ([8:10])
    Klasa 6 - Zęby ([11:48])
    Klasa 7 - Kanały zębowe ([111:148])
    """
    volume_new = volume.copy()  

    volume_new[(volume > 0) & (volume < 3)] = 1
    volume_new[volume == 7] = 2
    volume_new[(volume > 4) & (volume < 7)] = 3
    volume_new[(volume > 2) & (volume < 5)] = 4
    volume_new[(volume > 102) & (volume < 106)] = 4
    volume_new[(volume > 7) & (volume < 11)] = 5
    volume_new[(volume > 10) & (volume < 49)] = 6
    volume_new[(volume > 110) & (volume < 149)] = 7

    return volume_new

def bigger_compress_tooth_fairy_classes(volume):
    """
    Kompresja:
    Klasa 0 - background ([0])
    Klasa 1 - Kości szczęki górne dolne([1, 2])
    Klasa 2 - Gardło i Jamy nosowe ([5, 6, 7])
    Klasa 3 - Nerwy w szczęce ([3, 4, 103:105)
    Klasa 4 - Zęby, kanały i ciała obce ([8:48],[111:148])
    """
    volume_new = volume.copy()  

    volume_new[(volume > 0) & (volume < 3)] = 1
    volume_new[(volume > 4) & (volume < 8)] = 2
    volume_new[((volume > 102) & (volume < 106)) | ((volume > 2) & (volume < 5))] = 3
    volume_new[(volume > 7) & (volume < 49)] = 4
    volume_new[(volume > 110) & (volume < 149)] = 4

    return volume_new

def only_teeth_and_canal_classes(volume):
    """
    Kompresja:
    Klasa 0 - background ([0:10, 49:110])
    Klasa 1 - Zęby ([11:48])
    Klasa 2 - Kanały zębowe ([111:148])
    """
    volume_new = volume.copy()  

    volume_new[(volume > -1) & (volume < 11)] = 0
    volume_new[(volume > 48) & (volume < 111)] = 0
    
    volume_new[(volume > 10) & (volume < 49)] = 1
    volume_new[(volume > 110) & (volume < 149)] = 2

    return volume_new

def compression_factory(mode_name = 'big'):

    if mode_name == 'big':
        return only_teeth_and_canal_classes, 3
    
    if mode_name == 'medium_rare':
        return bigger_compress_tooth_fairy_classes, 5

    if mode_name == 'medium':
        return compress_tooth_fairy_classes, 8
    
    if mode_name == 'none':
        return fix_tooth_fairy_classes, 77





    
    


