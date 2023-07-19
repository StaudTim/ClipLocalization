import albumentations as A

from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms

"""
This file provides different functions you could use to transform your data. 
"""


def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


# Define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_train_aug():
    return A.Compose([
        A.MotionBlur(blur_limit=3, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, p=0.5
        ),
        A.ColorJitter(p=0.5),
        A.RandomGamma(p=0.2),
        A.RandomFog(p=0.2),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)
