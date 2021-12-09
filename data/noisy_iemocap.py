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

import json
from pathlib import Path
import numpy as np
import torch
import pickle as pkl
from tqdm import tqdm
import soundfile as sf
import random
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class NoisyIemocap(Dataset):
    def __init__(self, data_dir=r'source/noise_iemocap',
                 noise_type='NPARK', 
                 noise_level=2,
                 maxseqlen=160000,
                 class_num=4,
                 label_sess_no=1,
                 mode='train'
                 ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.label2idx = {'anger': 0, 'happy': 1, 'neutral': 2, 'sad': 3}
        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        with open(Path(self.data_dir).joinpath('labels_sess', f'label_{self.label_sess_no}.json'), 'r') as f:
            metainfo = json.load(f) #{wavname: {emotion: number}} or {wavname: emotion}

        if self.mode == 'train':
            t_dataset = [(str(Path(self.data_dir).joinpath(f'{self.noise_type}-{self.noise_level}', Path(n).stem+'_n.wav')), self.label2idx[l])
                                    for n,l in metainfo['Train'].items()]
            self.dataset = t_dataset[:int(0.9*len(t_dataset))]
        elif self.mode == 'valid':
            t_dataset = [(str(Path(self.data_dir).joinpath(f'{self.noise_type}-{self.noise_level}', Path(n).stem+'_n.wav')), self.label2idx[l])
                                    for n,l in metainfo['Train'].items()]
            self.dataset = t_dataset[int(0.9*len(t_dataset)):]
        else:
            self.dataset = [(str(Path(self.data_dir).joinpath(f'{self.noise_type}-{self.noise_level}', Path(n).stem+'_n.wav')), self.label2idx[l])
                                    for n,l in metainfo['Test'].items()]
        #Print statistics:
        print(f'Total {len(self.dataset)} examples')

        self.lengths = []
        print ("Loading over the dataset once...")
        for dataname, _ in tqdm(self.dataset):
            wav, _sr = sf.read(dataname)
            self.lengths.append(len(wav))

        avglen = float(sum(self.lengths)) / len(self.lengths) / 16000
        print (f"Average duration of audio: {np.round(avglen, decimals=4)} sec")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataname, label = self.dataset[idx]
        wav, _sr = sf.read(dataname)
        return wav.astype(np.float32), label

    def seqCollate(self, batch):
        getlen = lambda x: x[0].shape[0]
        max_seqlen = max(map(getlen, batch))
        target_seqlen = min(self.maxseqlen, max_seqlen)
        def trunc(x):
            x = list(x)
            if x[0].shape[0] >= target_seqlen:
                x[0] = x[0][:target_seqlen]
                output_length = target_seqlen
            else:
                output_length = x[0].shape[0]
                over = target_seqlen - x[0].shape[0]
                x[0] = np.pad(x[0], [0, over])
            ret = (x[0], output_length, x[1])
            return ret
        batch = list(map(trunc, batch))
        return default_collate(batch)
