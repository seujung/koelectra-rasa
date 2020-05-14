from pytorch_lightning import Trainer
from transformers import ElectraTokenizer
from argparse import Namespace

from electra_diet.pl_model import KoELECTRAClassifier

import os, sys
import torch

def train(
    file_path,
    # training args
    train_ratio=0.8,
    batch_size=32,
    intent_class_num=None,
    entity_class_num=None,
    optimizer="AdamW",
    intent_optimizer_lr=1e-5,
    entity_optimizer_lr=2e-5,
    checkpoint_path=os.getcwd(),
    max_epochs=20,
    #tokenizer=None,
    **kwargs
):
    gpu_num = torch.cuda.device_count()

    trainer = Trainer(
        default_root_dir=checkpoint_path, max_epochs=max_epochs, gpus=gpu_num
    )

    model_args = {}

    # training args
    model_args["max_epochs"] = max_epochs
    model_args["file_path"] = file_path
    model_args["train_ratio"] = train_ratio
    model_args["batch_size"] = batch_size
    model_args["intent_class_num"] = intent_class_num
    model_args["entity_class_num"] = entity_class_num
    model_args["optimizer"] = optimizer
    model_args["intent_optimizer_lr"] = intent_optimizer_lr
    model_args["entity_optimizer_lr"] = entity_optimizer_lr

    for key, value in kwargs.items():
        model_args[key] = value

    hparams = Namespace(**model_args)

    model = KoELECTRAClassifier(hparams)

    trainer.fit(model)
