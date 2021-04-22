import numpy as np
import torch
import random
from data import l2abands as bands

#from torchvision import transforms
"""
def center_crop(image,mask):
    size_crop = 56
    R = transforms.CenterCrop(size_crop)
    image = R(image)
    mask = R(mask)
    return image, mask
"""

def get_transform(mode, intensity=0, add_fdi_ndvi=False):
    assert mode in ["train", "test"]
    if mode in ["train"]:
        def train_transform(image, mask):

            if add_fdi_ndvi:
                fdi = np.expand_dims(calculate_fdi(image),0)
                ndvi = np.expand_dims(calculate_ndvi(image),0)
                image = np.vstack([image,ndvi,fdi])

            image *= 1e-4
            # return image, mask
            data_augmentation = get_data_augmentation(intensity=intensity)
            return data_augmentation(image, mask)
        return train_transform
    else:
        def test_transform(image, mask):
            if add_fdi_ndvi:
                fdi = np.expand_dims(calculate_fdi(image),0)
                ndvi = np.expand_dims(calculate_ndvi(image),0)
                image = np.vstack([image,ndvi,fdi])

            image *= 1e-4
            image = torch.Tensor(image)
            mask = torch.Tensor(mask)
            return image, mask
        return test_transform


def calculate_fdi(scene):
    # scene values [0,1e4]

    NIR = scene[bands.index("B8")] * 1e-4
    RED2 = scene[bands.index("B6")] * 1e-4
#    RED2 = cv2.resize(RED2, NIR.shape)

    SWIR1 = scene[bands.index("B11")] * 1e-4
    #SWIR1 = cv2.resize(SWIR1, NIR.shape)

    lambda_NIR = 832.9
    lambda_RED = 664.8
    lambda_SWIR1 = 1612.05
    NIR_prime = RED2 + (SWIR1 - RED2) * 10 * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED)

    return NIR - NIR_prime

def calculate_ndvi(scene):
    NIR = scene[bands.index("B8")].astype(np.float)
    RED = scene[bands.index("B4")].astype(np.float)
    return (NIR - RED) / (NIR + RED + 1e-12)

def get_data_augmentation(intensity):
    """
    do data augmentation:
    model
    """
    def data_augmentation(image, mask):
        image = torch.Tensor(image)
        mask = torch.Tensor(mask)
        mask = mask.unsqueeze(0)

        if random.random() < 0.5:
            # flip left right
            image = torch.fliplr(image)
            mask = torch.fliplr(mask)

        rot = np.random.choice([0,1,2,3])
        image = torch.rot90(image, rot, [1, 2])
        mask = torch.rot90(mask, rot, [1, 2])

        if random.random() < 0.5:
            # flip up-down
            image = torch.flipud(image)
            mask = torch.flipud(mask)

        if intensity >= 1:

            # random crop
            cropsize = image.shape[2] // 2
            image, mask = random_crop(image, mask, cropsize=cropsize)

            std_noise = 1 * image.std()
            if random.random() < 0.5:
                # add noise per pixel and per channel
                pixel_noise = torch.rand(image.shape[1], image.shape[2])
                pixel_noise = torch.repeat_interleave(pixel_noise.unsqueeze(0), image.size(0), dim=0)
                image = image + pixel_noise*std_noise

            if random.random() < 0.5:
                channel_noise = torch.rand(image.shape[0]).unsqueeze(1).unsqueeze(2)
                channel_noise = torch.repeat_interleave(torch.repeat_interleave(channel_noise, image.shape[1], 1),
                                                        image.shape[2], 2)
                image = image + channel_noise*std_noise

            if random.random() < 0.5:
                # add noise
                noise = torch.rand(image.shape[0], image.shape[1], image.shape[2]) * std_noise
                image = image + noise

        if intensity >= 2:
            # channel shuffle
            if random.random() < 0.5:
                idxs = np.arange(image.shape[0])
                np.random.shuffle(idxs) # random band indixes
                image = image[idxs]

        mask = mask.squeeze(0)
        return image, mask
    return data_augmentation

def random_crop(image, mask, cropsize):
    C, W, H = image.shape
    w, h = cropsize, cropsize

    # distance from image border
    dh, dw = h // 2, w // 2

    # sample some point inside the valid square
    x = np.random.randint(dw, W - dw)
    y = np.random.randint(dh, H - dh)

    # crop image
    image = image[:, x - dw:x + dw, y - dh:y + dh]
    mask = mask[:, x - dw:x + dw, y - dh:y + dh]

    return image, mask
