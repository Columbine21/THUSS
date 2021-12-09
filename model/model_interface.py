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

import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from transformers import AdamW
from utils.metrics import ConfusionMetrics

import pytorch_lightning as pl


class MInterface(pl.LightningModule):
    def __init__(self, model_name, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.valid_met = ConfusionMetrics(self.hparams.num_labels)
        self.test_met = ConfusionMetrics(self.hparams.num_labels)

    def forward(self, audio, audio_length):
        return self.model(audio, audio_length)

    def training_step(self, batch, batch_idx):
        audio, input_length, labels = batch
        out = self(audio, input_length)
        loss = self.loss_function(out, labels)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio, input_length, labels = batch
        out = self(audio, input_length)
        loss = self.loss_function(out, labels)
        out_digit = out.argmax(axis=1)

        correct_num = sum(labels == out_digit).cpu().item()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num/len(out_digit),
                 on_step=False, on_epoch=True, prog_bar=True)

        return (correct_num, len(out_digit))

    def test_step(self, batch, batch_idx):
        audio, input_length, labels = batch
        out = self(audio, input_length)
        loss = self.loss_function(out, labels)
        out_digit = out.argmax(axis=1)

        correct_num = sum(labels == out_digit).cpu().item()

        for l, p in zip(labels, out):
            # add to global validation metrics.
            self.test_met.fit(int(l), int(p.argmax()))

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num/len(out_digit),
                 on_step=False, on_epoch=True, prog_bar=True)

        return (correct_num, len(out_digit))

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        print('')

    def on_test_epoch_end(self):
        """Report metrics."""

        self.test_met.WriteConfusionSeaborn(labels=['neg', 'neu', 'pos'], outpath='conf_m.png')
        self.log('test_UAR', self.test_met.uar, logger=True)
        self.log('test_WAR', self.test_met.war, logger=True)
        self.log('test_macroF1', self.test_met.macroF1, logger=True)
        self.log('test_microF1', self.test_met.microF1, logger=True)

        print(f"""++++ Classification Metrics ++++
                  UAR: {self.test_met.uar:.4f}
                  WAR: {self.test_met.war:.4f}
                  macroF1: {self.test_met.macroF1:.4f}
                  microF1: {self.test_met.microF1:.4f}""")

    def configure_optimizers(self):
        # OPTIMIZER: finetune Bert Parameters.
        pretrain_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        pretrain_params = list(self.model.wav2vec2.named_parameters())

        pretrain_params_decay = [p for n, p in pretrain_params if not any(nd in n for nd in pretrain_no_decay)]
        pretrain_params_no_decay = [p for n, p in pretrain_params if any(nd in n for nd in pretrain_no_decay)]
        model_params_other = [p for n, p in list(self.model.named_parameters()) if 'wav2vec2' not in n]

        optimizer_grouped_parameters = [
                {'params': pretrain_params_decay, 'weight_decay': self.hparams.weight_decay_pretrain, 'lr': self.hparams.learning_rate_pretrain},
                {'params': pretrain_params_no_decay, 'weight_decay': 0.0, 'lr': self.hparams.learning_rate_pretrain},
                {'params': model_params_other, 'weight_decay': self.hparams.weight_decay_other, 'lr': self.hparams.learning_rate_other}
            ]
        optimizer = AdamW(optimizer_grouped_parameters)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        self.loss_function = F.cross_entropy

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)