from unicodedata import bidirectional
import torch
from torch import nn

from transformers import Wav2Vec2Model

_HIDDEN_STATES_START_POSITION = 2

class Wav2vec2Baseline(nn.Module):
    """ If you want to use pretrained model, or simply the standard structure implemented
        by Pytorch official, please use this template. It enable you to easily control whether
        use or not the pretrained weights, and whether to freeze the internal layers or not,
        This is made for wav2vec, but you can also adapt it to other structures by changing
        the `torch.hub.load` content.
    """
    def __init__(
        self, 
        use_weighted_layer_sum=True,
        model_type='dense',
        hidden_size=128,
        num_labels=4,
        freeze=True,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        num_layers = self.wav2vec2.config.num_hidden_layers + 1  # transformer layers + input embeddings
        if use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        if model_type == 'dense':
            self.projector = nn.Sequential (
                nn.Linear(self.wav2vec2.config.hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=0.2),
            )
        elif model_type == 'lstm':
            self.projector = nn.Sequential (
                nn.Linear(hidden_size, hidden_size), 
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)
            )
        elif model_type == 'fusion':
            pass
        else:
            raise ValueError(
                f'Invalid Module Type {model_type}!')

        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze:
            self.freeze_base_model()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameters
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False        

    def forward(
        self, 
        input_values,
        input_length,
        attention_mask=None,
        output_hidden_states=None
        ):
        output_hidden_states = True if self.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
        hidden_states = torch.stack(hidden_states, dim=1)
        norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
        hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        return self.classifier(pooled_output)
