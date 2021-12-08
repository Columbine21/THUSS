import torch
from torch import nn

from transformers import BertForSequenceClassification, AdamW, BertConfig

class Wav2vec2Baseline(nn.Module):
    """ If you want to use pretrained model, or simply the standard structure implemented
        by Pytorch official, please use this template. It enable you to easily control whether
        use or not the pretrained weights, and whether to freeze the internal layers or not,
        and the in/out channel numbers, resnet version. This is made for resnet, but you can
        also adapt it to other structures by changing the `torch.hub.load` content.
    """
    def __init__(self, pretrained_dir, num_labels, freeze):
        super().__init__()
        
        # 加载 BertForSequenceClassification, 预训练 BERT 模型 + 顶层的线性分类层 
        self.clf = BertForSequenceClassification.from_pretrained(
            pretrained_dir, 
            num_labels = num_labels, 
            output_attentions = False, 
            output_hidden_states = False, 
        )
        
        if freeze:  
            for param in self.clf.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, input_mask, segment_ids):
        return self.clf(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
