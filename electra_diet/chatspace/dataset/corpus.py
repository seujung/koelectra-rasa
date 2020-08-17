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


class DynamicCorpus:
    def __init__(self, corpus_path, corpus_size=None, repeat=False, encoding="utf-8"):
        self.corpus_path = corpus_path
        self.corpus_file = self.open()
        self.corpus_size = corpus_size if corpus_size else self._get_corpus_size()
        self.line_pointer = 0
        self.repeat = repeat
        self.encoding = encoding

    def __getitem__(self, item):
        if self.line_pointer >= self.corpus_size:
            if self.repeat:
                self.reload()
            else:
                raise IndexError
        return self.read_line()

    def __iter__(self):
        self.reload()
        for _ in range(self.corpus_size):
            yield self.read_line()

    def __len__(self):
        return self.corpus_size

    def read_line(self):
        line = self.corpus_file.readline().strip()
        self.line_pointer += 1
        return line

    def open(self):
        self.corpus_file = open(self.corpus_path, encoding=self.encoding)
        return self.corpus_file

    def reload(self):
        self.corpus_file.close()
        self.open()
        self.line_pointer = 0

    def _get_corpus_size(self):
        line_count = 0
        for _ in self.corpus_file:
            line_count += 1
        self.reload()
        return line_count
