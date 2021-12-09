# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
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

""" This main entrance of the whole project.
    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import MInterface
from data import DInterface
from utils import load_model_path_by_args


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=8,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    args.labeldir = str(Path(args.data_dir).joinpath('labels_sess'))
    nfolds = len(os.listdir(args.labeldir))
    for foldlabel in os.listdir(args.labeldir):
        assert foldlabel[-5:] == '.json'

    metrics, confusion = np.zeros((4, len(args.seeds), nfolds)), 0.
    for exp, seed in enumerate(args.seeds):
        pl.seed_everything(seed)
        for ifold, foldlabel in enumerate(os.listdir(args.labeldir)):
            print (f"Running experiment {exp+1} / {len(args.seeds)}, fold {ifold+1} / {nfolds}...")
            args.label_sess_no = ifold + 1
            load_path = load_model_path_by_args(args)
            data_module = DInterface(**vars(args))

            if load_path is None:
                model = MInterface(**vars(args))
            else:
                model = MInterface(**vars(args))
                args.resume_from_checkpoint = load_path
        
            args.callbacks = load_callbacks()

            # # If you want to change the logger's saving folder
            # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
            # args.logger = logger

            trainer = Trainer.from_argparse_args(args)
            trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seeds', default=[111,1111,11111], type=list)

    parser.add_argument('--weight_decay_pretrain', default=2e-5, type=float)
    parser.add_argument('--learning_rate_pretrain', default=4e-6, type=float)
    parser.add_argument('--weight_decay_other', default=2e-5, type=float)
    parser.add_argument('--learning_rate_other', default=2e-4, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', default=None, choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='noisy_iemocap', type=str)
    parser.add_argument('--data_dir', default='source/noise_iemocap', type=str)
    parser.add_argument('--noise_type', default='NPARK', type=str)
    parser.add_argument('--noise_level', default=2, type=int)
    parser.add_argument('--maxseqlen', default=128000, type=int)
    parser.add_argument('--class_num', default=4, type=int)
    parser.add_argument('--label_sess_no', default=4, type=int)
    parser.add_argument('--model_name', default='wav2vec2_baseline', type=str)
    parser.add_argument('--log_dir', default='lightning_logs', type=str)

    
    # Model Hyperparameters
    parser.add_argument('--use_weighted_layer_sum', default=True, type=bool)
    parser.add_argument('--model_type', default='dense', type=str)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--num_labels', default=4, type=int)
    parser.add_argument('--freeze', default=True, type=bool)

    # Other
    parser.add_argument_group(title="pl.Trainer args")
    parser = Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)
    parser.set_defaults(gpus=1)

    args = parser.parse_args()

    main(args)