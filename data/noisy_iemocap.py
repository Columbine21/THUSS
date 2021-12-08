# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

from torchvision import transforms
from sklearn.model_selection import train_test_split


class NoisyIemocap(data.Dataset):
    def __init__(self, data_dir=r'data/ref',
                 class_num=4,
                 train=True,
                 no_augment=True,
                 aug_prob=0.5):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment

        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass