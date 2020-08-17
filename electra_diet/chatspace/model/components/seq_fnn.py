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


class SequentialFNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_distributed_1 = TimeDistributed(
            nn.Linear(in_features=config["cnn_features"] * 4, out_features=config["cnn_features"])
        )
        self.time_distributed_2 = TimeDistributed(
            nn.Linear(in_features=config["cnn_features"], out_features=config["cnn_features"] // 2)
        )

    def forward(self, conv_embed):
        x = torch.transpose(conv_embed, 2, 1)
        x = self.time_distributed_1(x)
        x = self.time_distributed_2(x)
        return x
