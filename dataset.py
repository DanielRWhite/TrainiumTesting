from typing import Union
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from functools import partial
import multiprocessing as mp
import numpy as np
import torch
import re

def glob_re(pattern, strings):
    return list(filter(re.compile(pattern).match, strings))

def list_files(key, root_dir = None):
        if root_dir is None:
                files = Path(key).glob('*')
        else:
                files = root_dir.joinpath(key).glob('*')
        
        return (key, glob_re(r'.+(\.)(jpg|jpeg|png)', [ str(file) for file in files ]))

class DigiFaceDataset(Dataset):
        
        def __init__(self, root_dir, image_size, pre_transforms = None, **kwargs):
                super(DigiFaceDataset, self).__init__(**kwargs)
                
                self.root_dir = Path(root_dir)
                self.load()
                
                self.pre_transforms = pre_transforms
                self.transforms = transforms.Compose([
                        transforms.RandomApply(transforms = [
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness = 0.5, hue = 0.5),
                                transforms.RandomPosterize(bits = 2),
                                transforms.RandomAdjustSharpness(sharpness_factor = 8),
                                transforms.RandomAutocontrast(),
                                transforms.RandomEqualize()
                        ], p = 0.5),
                        transforms.Resize(image_size),
                        transforms.ToTensor()
                ])
                
        def load(self):
                self.total = 0
                people = [ x for x in self.root_dir.iterdir() if x.is_dir() ]
                
                with mp.Pool(8) as p:
                        results = p.map(partial(list_files, root_dir = self.root_dir), people)
                        
                for (_, fi) in list(results):
                        self.total = self.total + len(fi)
                        
                self.config = dict(list(results))
                self.key_mappings = people

        def idx_mapping(self, idx) -> Union[str, int]:
                total = 0
                for person in list(self.config.keys()):
                        if len(self.config[person]) + total < idx:
                                total = total + len(self.config[person])
                                continue
                        
                        index = (len(self.config[person]) + total) - idx - 1
                        return person, index
                
        def preprocess(self, image):
                return transforms.ToPILImage()(self.pre_transforms(image)) if self.pre_transforms is not None else image
                        
        def __len__(self):
                return self.total
                        
        def __getitem__(self, idx):
                person, index = self.idx_mapping(idx)
                
                anchor_image = Image.open(self.root_dir.joinpath(
                        self.config[person][index]
                )).convert("RGB")
                
                pos_choices = [ p_index for p_index in range(len(self.config[person])) if p_index != index ]
                pos_choice = np.random.choice(pos_choices)
                pos_image = Image.open(self.root_dir.joinpath(
                        self.config[person][pos_choice])
                ).convert("RGB")
                
                neg_key = np.random.choice([ key for key in self.key_mappings if key != person ])
                neg_image = Image.open(self.root_dir.joinpath(self.config[neg_key][
                        np.random.choice(range(len(list(self.config[neg_key]))))
                ])).convert("RGB")
                
                
                a_img = self.transforms(self.preprocess(anchor_image))
                p_img = self.transforms(self.preprocess(pos_image))
                n_img = self.transforms(self.preprocess(neg_image))
                
                return a_img, p_img, n_img
                