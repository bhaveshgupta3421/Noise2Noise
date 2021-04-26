from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image
from torchvision.transforms import Resize, InterpolationMode
import torch
import random
resize = Resize((256,256), interpolation = InterpolationMode.BICUBIC)

def add_noise(x):

    (minval,maxval) = (0.0, 50.0)
    std = random.uniform(minval/255.0, maxval/255.0)
    return x + torch.normal(0.0, std, size=(1,256,256))

def resize_noise_image(x):
    resized_image = resize.forward(x)/ 255.0 - 0.5
    return (add_noise(resized_image), add_noise(resized_image))

class LoadImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.filepaths = os.listdir(self.img_dir)
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filepaths[idx])
        orig_img = read_image(img_path)
        noised_inp, noised_mask = self.transform(orig_img)
        #sample = {"image": noised_inp, "label": noised_mask}
        sample = (noised_inp,noised_mask)

        return sample