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

from .time_distributed import TimeDistributed


class Projection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 0: PAD_TARGET, 1: NONE_SPACE_TARGET, 2: SPACE_TARGET
        self.seq_fnn = TimeDistributed(nn.Linear(config["cnn_features"], 3))
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # feed into projection layer
        x = torch.transpose(x, 1, 0)
        x = self.seq_fnn(x)

        # log-softmax output
        x = torch.transpose(x, 1, 0)
        x = self.softmax(x)
        return x
