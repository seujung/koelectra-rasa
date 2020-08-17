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

from typing import Iterable, List, Optional, Union

from .vocab import Vocab


class Indexer:
    """Indexer for word."""

    def __init__(self, vocab: Vocab):
        """
        Init WordIndexer with inherited Indexer initializer

        :param vocab: word vocab object
        """
        self.vocab = vocab

    def encode(
        self,
        text: str,
        max_seq_len: Optional[int] = None,
        min_seq_len: Optional[int] = None,
        pad_idx: int = 0,
        unk_word: str = "[UNK]",
    ) -> Union[List[int], str]:
        """
        Convert tokenized tokens to corresponding ids.

        When `max_seq_len` is not None, encoded tokens will be
        padded up by padding idx to `max_seq_len`.

        :param text: single text for encoding
        :param min_seq_len: minimum sequence length of encoded tokens
            if encoded tokens are shorter then min_seq_len add padding
            tokens (min_seq_len - encoded_token_length) times.
        :param max_seq_len : maximum sequence length of encoded tokens.
            if encoded tokens are longer then max_seq_len then slice it.
        :param pad_idx: padding token index(int) for padding
        :param unk_word: get unk index with unk_word word as key
        :return: list of token ids (int)
        """
        unk_token_id = self.vocab.get(unk_word)
        encoded_text = [self.vocab.get(token, unk_token_id) for token in text]
        if min_seq_len is not None and len(encoded_text) < min_seq_len:
            encoded_text.extend((pad_idx,) * (min_seq_len - len(encoded_text)))
        if max_seq_len:
            encoded_text = (
                encoded_text if len(encoded_text) < max_seq_len else encoded_text[:max_seq_len]
            )
        return encoded_text

    def decode(self, token_ids: Iterable[int], pad_idx: int = 0, as_str=False) -> List[str]:
        """
        Convert token ids to corresponding tokens.

        :param token_ids: token ids(list of int) for decoding
        :param pad_idx: padding token index for padding
        :param as_str: if true, return as concatenated string
            if false, return as list of token string
        :return: return decoded result. return type changed depends on as_str
        """
        decoded_token_ids = [self.vocab.get_token(token_id) for token_id in token_ids]
        return "".join(decoded_token_ids) if as_str else decoded_token_ids
