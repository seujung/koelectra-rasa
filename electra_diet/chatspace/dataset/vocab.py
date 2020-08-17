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

from collections import Counter
from typing import Iterable, List, Optional, Tuple, Union

DEFAULT_FORWARD_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[SOS]", "[EOS]", " ")


class Vocab(dict):
    def __init__(
        self,
        forward_special_tokens: Optional[
            Union[List[str], Tuple[str, ...]]
        ] = DEFAULT_FORWARD_SPECIAL_TOKENS,
        tokens: Optional[Union[List[str], Tuple[str, ...]]] = None,
        backward_special_tokens: Optional[Union[List[str], Tuple[str, ...]]] = None,
    ):
        """
        default forward special tokens will be allocated at forward index (0...4)
        then tokens (list or tuple or iterable of str) will be allocated as next
        at last, special_tokens will be allocated at last position.

        :param tokens: list or tuple of tokens(str)
        :param forward_special_tokens: tuple of string,
                which indicate the special tokens appended in forward
        :param backward_special_tokens: tuple of string,
                which indicate the special tokens appended in backward
        """
        super().__init__()

        self.idx_to_token = []
        self.forward_special_token = forward_special_tokens
        self.tokens = tokens
        self.backward_special_tokens = backward_special_tokens

        # add special tokens on back size
        if forward_special_tokens:
            for special_token in forward_special_tokens:
                self.add(special_token)

        if tokens is not None:
            for token in tokens:
                self.add(token)

        if backward_special_tokens is not None:
            for special_token in backward_special_tokens:
                self.add(special_token)

    def add(self, token: str, index: int = None) -> int:
        """
        add token on this vocab.

        :param token: token which you want to add on this vocab
        :param index: optional, index where you want to add.
            if another token is exist in index, override it into this token.
        :return: index of the token in vocab
        """
        if token in self:
            return self[int]

        token_index = len(self) if index is None else index

        if token_index == len(self) or token_index == 0:
            self.idx_to_token.append(token)
        else:
            self.idx_to_token[token_index] = token

        self[token] = token_index

        return token_index

    def build(
        self,
        lines: Union[Iterable, List[str]],
        min_count: Optional[int] = None,
        max_vocab_size: Optional[int] = None,
        sep_token: str = "",
    ):
        """
        build vocab with multiple string lines.
        vocab will be created as ascending order of token counter.
        however index will be allocated after current fixed special tokens or others.

        :param lines: texts or string iterable
        :param min_count: optional, tokens need to be occurred then min_count
        :param max_vocab_size: optional, vocab size will be limited with max_vocab_size
        :param sep_token: line will be separated by sep_token (default: space)
        :return: vocab instance
        """

        counter = Counter()
        for line in lines:
            for token in line.strip().split(sep_token):
                counter.update(token)

        return self.build_with_counter(counter, min_count, max_vocab_size)

    def build_with_counter(
        self,
        token_counter: Counter,
        min_count: Optional[int] = None,
        max_vocab_size: Optional[int] = None,
    ) -> "Vocab":
        """
        build vocab with token counter.
        vocab will be created as ascending order of counter.
        however index will be allocated after current fixed special tokens or others.

        :param token_counter: counter class instance of token count
        :param min_count: optional, tokens need to be occurred then min_count
        :param max_vocab_size: optional, vocab size will be limited with max_vocab_size
        :return vocab instance
        """

        max_vocab_size = len(token_counter) if max_vocab_size is None else max_vocab_size
        max_vocab_size -= len(self)

        for token, count in token_counter.most_common(max_vocab_size):
            if min_count is None or count >= min_count:
                self.add(token)

        return self

    def get_token(self, index: int) -> str:
        """
        return token of given index.

        :param index: query index to get the token
        :return: token string
        """
        return self.idx_to_token[index]

    @staticmethod
    def load(
        path: str, with_forward_special_tokens: bool = False, encoding: str = "utf-8"
    ) -> "Vocab":
        """
        load vocab file from txt file.

        :param path: vocab txt file path
        :param with_forward_special_tokens:
            if true, the forward special tokens(PAD, EOS ..) will be added
            before txt vocab loading. and txt vocabs will be assigned in
            backward position (e.x apple 4, banana 5 ...)
        :param encoding: encoding string. default: utf-8
        :return:
        """

        with open(path, encoding=encoding) as f:
            tokens = [line.strip() for line in f]

        if with_forward_special_tokens:
            return Vocab(tokens=tokens)
        return Vocab(forward_special_tokens=None, tokens=tokens)

    def dump(
        self,
        path: Optional[str] = None,
        with_forward_special_tokens: bool = True,
        with_backward_special_tokens: bool = True,
        encoding: str = "utf-8",
    ) -> List[str]:
        """
        dump and save vocab into file

        :param path: path to save vocab
        :param with_forward_special_tokens:
            if true, dumped tokens include the forward_special_token
        :param with_backward_special_tokens:
            if true, dumped tokens include the backward_special_token
        :param encoding: encoding string. default: utf-8
        :return: dumped tokens as list of string
        """
        dump_tokens = []

        if with_forward_special_tokens and self.forward_special_token:
            dump_tokens.extend(self.forward_special_token)

        if self.tokens:
            dump_tokens.extend(self.tokens)

        if with_backward_special_tokens and self.backward_special_tokens:
            dump_tokens.extend(self.backward_special_tokens)

        if path is not None:
            with open(path, "w", encoding=encoding) as f:
                for token in dump_tokens:
                    f.write(f"{token}\n")

        return dump_tokens

    def keys(self):
        return self.idx_to_token
