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

import random
from typing import List, Union

import torch
from torch.utils.data import Dataset

from .corpus import DynamicCorpus
from .indexer import Indexer
from .vocab import Vocab


class ChatSpaceDataset(Dataset):
    def __init__(
        self,
        config,
        texts: Union[DynamicCorpus, List[str]],
        vocab: Vocab,
        with_random_space: bool = False,
    ):
        self.texts = texts
        self.indexer = Indexer(vocab)
        self.space_prob = config["space_prob"]
        self.with_random_space = with_random_space
        self.config = config
        self.lines = []

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.with_random_space:
            model_input = self.get_train_input(self.texts[idx], idx)
        else:
            model_input = {"input": list(self.texts[idx])}

        model_input["input"] = self.indexer.encode(
            model_input["input"], min_seq_len=self.config["min_seq_len"], unk_word="[UNK]"
        )
        model_input["length"] = len(model_input["input"])
        return model_input

    def get_train_input(self, input_text, idx):
        input_char, label = [], []
        word_list = input_text.split()

        for word in word_list:
            word_label = [1] * (len(word) - 1) + [2]
            char_list = list(word)

            if random.random() < self.space_prob[idx % len(self.space_prob)]:
                char_list.append(" ")
                word_label.append(1)

            input_char.extend(char_list)
            label.extend(word_label)

        return {"input": "".join(input_char), "label": label}

    @staticmethod
    def train_collect_fn(batch):
        max_seq_len = max([model_input["length"] for model_input in batch])
        for example in batch:
            example["input"].extend([0] * (max_seq_len - len(example["input"])))
            example["label"].extend([0] * (max_seq_len - len(example["label"])))

        batch = {key: torch.tensor([example[key] for example in batch]) for key in batch[0].keys()}
        return batch

    @staticmethod
    def eval_collect_fn(batch):
        max_seq_len = max([model_input["length"] for model_input in batch])
        for example in batch:
            example["input"].extend([0] * (max_seq_len - len(example["input"])))

        batch = {key: torch.tensor([example[key] for example in batch]) for key in batch[0].keys()}
        return batch
