"""
Copyright 2019 Pingpong AI Research, ScatterLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

from .components.char_conv import CharConvolution
from .components.char_lstm import CharLSTM
from .components.embed import CharEmbedding
from .components.projection import Projection
from .components.seq_fnn import SequentialFNN


class ChatSpaceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = CharEmbedding(config)
        self.conv = CharConvolution(config)
        self.lstm = CharLSTM(config)
        self.projection = Projection(config)
        self.fnn = SequentialFNN(config)
        self.batch_normalization = nn.BatchNorm1d(4 * config["cnn_features"])
        self.layer_normalization = nn.LayerNorm(config["cnn_features"])

    def forward(self, input_seq, length) -> torch.Tensor:
        x = self.embed.forward(input_seq)
        x = self.conv.forward(x)
        x = self.batch_normalization.forward(x)
        x = self.fnn.forward(x)
        x = self.lstm.forward(x, length)
        x = self.layer_normalization(x)
        x = self.projection.forward(x)
        return x
