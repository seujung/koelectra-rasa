from pytorch_lightning import Trainer
from transformers import ElectraTokenizer
from argparse import Namespace

from electra_diet.pl_model import KoELECTRAClassifier
from electra_diet.dataset.electra_dataset import ElectraDataset

import os, sys
import glob
import torch
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.base import Callback
from electra_diet.metrics import show_intent_report

class PerfCallback(Callback):
    def __init__(self, file_path=None, gpu_num=0, report_nm=None, output_dir=None):
        self.file_path = file_path
        if gpu_num > 0:
            self.cuda = True
        else:
            self.cuda = False
        self.report_nm = report_nm
        self.output_dir = output_dir

    def on_train_end(self, trainer, pl_module):
        print("train finished")
        if self.file_path is None:
            dataset = pl_module.val_dataset
        else:
            dataset = ElectraDataset(file_path=self.file_path, tokenizer=None)
                
        if self.output_dir is None:
            path = 'lightning_logs/'
            folder_path = [f for f in glob.glob(path + "**/", recursive=False)]
            folder_path.sort()
            self.output_dir  = folder_path[-1]
        self.output_dir = os.path.join(self.output_dir, 'results')
        show_intent_report(dataset, pl_module, file_name=self.report_nm, output_dir=self.output_dir, cuda=self.cuda)


def train(
    file_path,
    # training args
    train_ratio=0.8,
    batch_size=32,
    seq_len=128,
    intent_class_num=None,
    entity_class_num=None,
    optimizer="AdamW",
    intent_optimizer_lr=1e-5,
    entity_optimizer_lr=2e-5,
    checkpoint_path=os.getcwd(),
    max_epochs=10,
    report_nm=None,
    #tokenizer=None,
    **kwargs
):
    gpu_num = torch.cuda.device_count()

    trainer = Trainer(
        default_root_dir=checkpoint_path, max_epochs=max_epochs, gpus=gpu_num, callbacks=[PerfCallback(file_path = file_path, gpu_num=gpu_num, report_nm=report_nm)]
    )

    model_args = {}

    # training args
    model_args["max_epochs"] = max_epochs
    model_args["file_path"] = file_path
    model_args["train_ratio"] = train_ratio
    model_args["batch_size"] = batch_size
    model_args["seq_len"] = seq_len
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
