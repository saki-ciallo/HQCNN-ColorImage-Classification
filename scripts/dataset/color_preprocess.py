import cv2
import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from pennylane import numpy as np


def scaling_channel(image_np, color_space=None):

    image_np = image_np.astype(np.float32)
    if color_space == 'Lab':
        # cv2.COLOR_RGB2Lab [255,128,128]
        # L: 0-255 -> 0-1
        image_np[:, :, 0] /= 255.0
        # a: 0-255 -> approximately -1 to 1
        image_np[:, :, 1] = (image_np[:, :, 1] - 128) / 127.5
        # b: same as a
        image_np[:, :, 2] = (image_np[:, :, 2] - 128) / 127.5
    elif color_space == 'YCrCb':
        # cv2.COLOR_RGB2YCrCb [255,128,128]
        # Y: 0-255 -> 0-1
        image_np[:, :, 0] /= 255.0
        # Cr: 0-255 -> -1 to 1
        image_np[:, :, 1] = (image_np[:, :, 1] - 128) / 127.5
        # Cb: same as Cr
        image_np[:, :, 2] = (image_np[:, :, 2] - 128) / 127.5
    elif color_space == 'HSV':
        # cv2.COLOR_RGB2HSV [179,255,255]
        # H: 0-179 -> 0-1
        image_np[:, :, 0] /= 179.0
        # S: 0-255 -> 0-1
        image_np[:, :, 1] /= 255.0
        # V: 0-255 -> 0-1
        image_np[:, :, 2] /= 255.0
    else:  # Assume RGB or other
        # Scale all channels to 0-1
        image_np /= 255.0

    return image_np

def process_images(dataset, indices, size=(16, 16), color_space=None, interpolation='bilinear'):

    processed_images = []
    # todo 实际处理图像的地方
    for idx in indices:

        image = dataset.data[idx] # train_dataset (32,32,3) 一张张处理
        image_tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0).float()  # N,C,H,W
        
        # input dimensions: mini-batch x channels x [optional depth] x [optional height] x width
        # if interpolation in ['bilinear', 'bicubic']:
        #     resized_image = interpolate(image_tensor, size=size, mode=interpolation, align_corners=False)
        # else:
        #     resized_image = interpolate(image_tensor, size=size, mode=interpolation)
        resized_image = interpolate(image_tensor, size=size, mode=interpolation, align_corners=False)

        image_np = resized_image.squeeze(0).permute(1, 2, 0).numpy()

        if color_space == 'Lab':
            image_np = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2Lab)
        elif color_space == 'YCrCb':
            image_np = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        elif color_space == 'HSV':
            image_np = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2HSV)
        # If color_space is None or other, remain as RGB

        image_np = scaling_channel(image_np, color_space=color_space)
        processed_images.append(image_np)

    return processed_images
    
class preprocessed_Dataset(Dataset):
    def __init__(self, dataset, indices, labels, size=(16, 16), color_space=None, scaling_method=None):
        self.dataset = dataset # train_dataset
        self.indices = indices
        self.labels = labels
        self.size = size
        self.color_space = color_space
        self.scaling_method = scaling_method
        self.images = process_images(self.dataset, self.indices, size=self.size, color_space=self.color_space)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32, requires_grad=False)
        label = torch.tensor(self.labels[idx], dtype=torch.long, requires_grad=False)
        # one_hot_label = one_hot(label, num_classes=2).float()
        return image, label

    def __len__(self):
        return len(self.labels)
    