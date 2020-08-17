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

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculated_metric(batch_input, batch_output, batch_label):
    binary_batch_output = [
        max(0, batch_output[batch_index][seq_index] - 1)
        for batch_index in range(len(batch_input))
        for seq_index in range(len(batch_input[batch_index]))
        if batch_input[batch_index][seq_index] > 0
    ]

    batch_label = [
        batch_label[batch_index][seq_index] - 1
        for batch_index in range(len(batch_input))
        for seq_index in range(len(batch_input[batch_index]))
        if batch_input[batch_index][seq_index] > 0
    ]

    accuracy = accuracy_score(batch_label, binary_batch_output)
    precision = precision_score(batch_label, binary_batch_output)
    recall = recall_score(batch_label, binary_batch_output)
    f1score = f1_score(batch_label, binary_batch_output)

    return {"acc": accuracy, "precision": precision, "recall": recall, "f1": f1score}
