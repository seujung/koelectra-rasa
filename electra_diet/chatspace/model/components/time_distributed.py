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

import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, layer, activation="relu"):
        super().__init__()
        self.layer = layer
        self.activation = self.select_activation(activation)

    def forward(self, x):
        x_reshaped = x.contiguous().view(-1, x.size(-1))

        y = self.layer(x_reshaped)
        y = self.activation(y)

        y = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y

    def select_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "tanh":
            return nn.Tanh()
        raise KeyError
