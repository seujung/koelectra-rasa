from pytorch_lightning import Trainer
from transformers import ElectraTokenizer
from argparse import Namespace

from electra_diet.pl_model import KoELECTRAClassifier
from electra_diet.dataset.electra_dataset import ElectraDataset

import os, sys
import glob
import torch
from torch.utils.data import DataLoader

from electra_diet.eval import PerfCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateLogger

early_stop_callback = EarlyStopping(
   monitor='val_accuracy',
   min_delta=0.00,
   patience=3,
   verbose=False,
   mode='max'
)

lr_logger = LearningRateLogger()

def train(
    file_path,
    # training args
    train_ratio=0.8,
    batch_size=32,
    seq_len=128,
    warmup_ratio = 0.2,
    intent_class_num=None,
    entity_class_num=None,
    optimizer="AdamW",
    intent_optimizer_lr=1e-5,
    entity_optimizer_lr=2e-5,
    checkpoint_path=os.getcwd(),
    max_epochs=10,
    report_nm=None,
    lower_text=True,
    early_stop=True,
    **kwargs
):
    gpu_num = torch.cuda.device_count()
    if early_stop:
        early_stop_callback = EarlyStopping(
                               monitor='val_loss',
                               min_delta=0.00,
                               patience=3,
                               verbose=False,
                               mode='min'
                            )
        
        trainer = Trainer(
            default_root_dir=checkpoint_path, max_epochs=max_epochs, gpus=gpu_num,
            callbacks=[PerfCallback(gpu_num=gpu_num, report_nm=report_nm, root_path=checkpoint_path), lr_logger],
            early_stop_callback=early_stop_callback
        )
        
    else:
        trainer = Trainer(
            default_root_dir=checkpoint_path, max_epochs=max_epochs, gpus=gpu_num,
            callbacks=[PerfCallback(gpu_num=gpu_num, report_nm=report_nm, root_path=checkpoint_path), lr_logger]
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
    model_args['lower_text'] = lower_text
    model_args['warmup_ratio'] = warmup_ratio

    for key, value in kwargs.items():
        model_args[key] = value

    hparams = Namespace(**model_args)
    print(hparams)

    model = KoELECTRAClassifier(hparams)

    trainer.fit(model)
