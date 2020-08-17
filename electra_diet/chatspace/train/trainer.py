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

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from chatspace.data import ChatSpaceDataset, DynamicCorpus, Vocab
from chatspace.model import ChatSpaceModel

from .metric import calculated_metric


class ChatSpaceTrainer:
    def __init__(
        self,
        config,
        model: ChatSpaceModel,
        vocab: Vocab,
        device: torch.device,
        train_corpus_path,
        eval_corpus_path=None,
        encoding="utf-8",
    ):
        self.config = config
        self.device = device
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=config["learning_rate"])
        self.criterion = nn.NLLLoss()
        self.vocab = vocab
        self.encoding = encoding

        self.train_corpus = DynamicCorpus(train_corpus_path, repeat=True, encoding=self.encoding)
        self.train_dataset = ChatSpaceDataset(
            config, self.train_corpus, self.vocab, with_random_space=True
        )

        if eval_corpus_path is not None:
            self.eval_corpus = DynamicCorpus(eval_corpus_path, encoding=self.encoding)
            self.eval_dataset = ChatSpaceDataset(
                self.config, eval_corpus_path, self.vocab, with_random_space=True
            )

        self.global_epochs = 0
        self.global_steps = 0

    def eval(self, batch_size=64):
        self.model.eval()

        with torch.no_grad():
            eval_output = self.run_epoch(self.eval_dataset, batch_size=batch_size, is_train=False)

        self.model.train()
        return eval_output

    def train(self, epochs=10, batch_size=64):
        for epoch_id in range(epochs):
            self.run_epoch(
                self.train_dataset,
                batch_size=batch_size,
                epoch_id=epoch_id,
                is_train=True,
                log_freq=self.config["logging_step"],
            )
            self.save_checkpoint(f"outputs/checkpoints/checkpoint_ep{epoch_id}.cpt")
            self.save_model(f"outputs/models/chatspace_ep{epoch_id}.pt")
            self.save_model(f"outputs/jit_models/chatspace_ep{epoch_id}.pt", as_jit=False)

    def run_epoch(self, dataset, batch_size=64, epoch_id=0, is_train=True, log_freq=100):
        step_outputs, step_metrics, step_inputs = [], [], []
        collect_fn = (
            ChatSpaceDataset.train_collect_fn if is_train else ChatSpaceDataset.eval_collect_fn
        )
        data_loader = DataLoader(dataset, batch_size, collate_fn=collect_fn)
        for step_num, batch in enumerate(data_loader):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            output = self.step(step_num, batch)

            if is_train:
                self.update(output["loss"])

            if not is_train or step_num % log_freq == 0:
                batch = {key: value.cpu().numpy() for key, value in batch.items()}
                output = {key: value.detach().cpu().numpy() for key, value in output.items()}

                metric = self.step_metric(output["output"], batch, output["loss"])

                if is_train:
                    print(
                        f"EPOCH:{epoch_id}",
                        f"STEP:{step_num}/{len(data_loader)}",
                        [(key + ":" + "%.3f" % metric[key]) for key in metric],
                    )
                else:
                    step_outputs.append(output)
                    step_metrics.append(metric)
                    step_inputs.append(batch)

        if not is_train:
            return self.epoch_metric(step_inputs, step_outputs, step_metrics)

        if is_train:
            self.global_epochs += 1

    def epoch_metric(self, step_inputs, step_outputs, step_metrics):
        average_loss = np.mean([metric["loss"] for metric in step_metrics])

        epoch_inputs = [
            example for step_input in step_inputs for example in step_input["input"].tolist()
        ]
        epoch_outputs = [
            example
            for output in step_outputs
            for example in output["output"].argmax(axis=-1).tolist()
        ]
        epoch_labels = [
            example for step_input in step_inputs for example in step_input["label"].tolist()
        ]

        epoch_metric = calculated_metric(
            batch_input=epoch_inputs, batch_output=epoch_outputs, batch_label=epoch_labels
        )

        epoch_metric["loss"] = average_loss
        return epoch_metric

    def step_metric(self, output, batch, loss=None):
        metric = calculated_metric(
            batch_input=batch["input"].tolist(),
            batch_output=output.argmax(axis=-1).tolist(),
            batch_label=batch["label"].tolist(),
        )

        if loss is not None:
            metric["loss"] = loss
        return metric

    def step(self, step_num, batch, with_loss=True, is_train=True):
        output = self.model.forward(batch["input"], batch["length"])
        if is_train:
            self.global_steps += 1

        if not with_loss:
            return {"output": output}

        loss = self.criterion(output.transpose(1, 2), batch["label"])
        return {"loss": loss, "output": output}

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path, as_jit=False):
        self.optimizer.zero_grad()
        params = [
            {"param": param, "require_grad": param.requires_grad}
            for param in self.model.parameters()
        ]

        for param in params:
            param["param"].require_grad = False

        with torch.no_grad():
            if not as_jit:
                torch.save(self.model.state_dict(), path)
            else:
                self.model.cpu().eval()

                sample_texts = ["오늘 너무 재밌지 않았어?", "너랑 하루종일 놀아서 기분이 좋았어!"]
                dataset = ChatSpaceDataset(
                    self.config, sample_texts, self.vocab, with_random_space=False
                )
                data_loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.eval_collect_fn)

                for batch in data_loader:
                    model_input = (batch["input"].detach(), batch["length"].detach())
                    traced_model = torch.jit.trace(self.model, model_input)
                    torch.jit.save(traced_model, path)
                    break

                self.model.to(self.device).train()

        print(f"Model Saved on {path}{' as_jit' if as_jit else ''}")

        for param in params:
            if param["require_grad"]:
                param["param"].require_grad = True

    def save_checkpoint(self, path):
        torch.save(
            {
                "epoch": self.global_epochs,
                "steps": self.global_steps,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_epochs = checkpoint["epoch"]
        self.global_steps = checkpoint["steps"]

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
