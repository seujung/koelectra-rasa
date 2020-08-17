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


class CharLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config["cnn_features"] // 2,
            hidden_size=config["cnn_features"] // 2,
            num_layers=config["lstm_layers"],
            bidirectional=config["lstm_bidirectional"],
            batch_first=True,
        )

    def forward(self, x, length):
        return self.lstm(x)[0]
