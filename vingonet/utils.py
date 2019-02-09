import cv2
from albumentations import CenterCrop
from torchvision.transforms import ToTensor

to_tensor = ToTensor()

def load_image(image_path, augmentations=None, image_size=(224, 224), zoom=1):
    image = cv2.imread(image_path)

    crop_size = int(min(image.shape[:-1]) / zoom)
    image = CenterCrop(crop_size, crop_size)(image=image)["image"]
    image = cv2.resize(image, (600, 600))

    if augmentations is not None:
        image = augmentations(image=image)["image"]

    image = cv2.resize(image, image_size)
    image = to_tensor(image)

    return image
