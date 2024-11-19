import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, name, list_file, preprocess=False):
        """
        Args:
            name (string) : The name of the dataset.
            list_file (string): Path to the txt file with image filenames.
            preprocess (bool): If true, scale and crop images at load time.
        """
        self.name = name
        self.preprocess = preprocess
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # # Get the image filename
        img_name = self.image_filenames[idx]
        image = Image.open(img_name).convert('RGB')

        # split image into image_rgb and image_semantic
        w, h = image.size
        w2 = int(w / 2)
        image_rgb = image.crop((0, 0, w2, h))
        image_semantic = image.crop((w2, 0, w, h))

        transform_params = get_params(image_rgb.size)
        A_transform = get_transform(self.preprocess, transform_params)
        B_transform = get_transform(self.preprocess, transform_params)

        image_rgb = A_transform(image_rgb)[[2, 1, 0], ...]
        image_semantic = B_transform(image_semantic)[[2, 1, 0], ...]

        return image_rgb, image_semantic
    
def get_params(size):
    w, h = size
    new_h = h
    new_w = w

    new_h = new_w = 286

    x = random.randint(0, np.maximum(0, new_w - 256))
    y = random.randint(0, np.maximum(0, new_h - 256))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(preprocess, params=None, method=transforms.InterpolationMode.BICUBIC):
    transform_list = []

    if preprocess:
        osize = [286, 286]
        transform_list.append(transforms.Resize(osize, method))
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], 256)))
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
