#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
from glob import glob
from datetime import datetime

def display(text):
    print("="*50)
    print("==> {}".format(text))
    print("="*50)

def install_package():
    os.system("rm -rf appliednl-mlp")
#     os.system("rm -rf appliednl-rasalt")
    os.system("rm -rf koelectra-rasa")
    os.system("rm -rf electra_diet_log")
#     os.system("pip uninstall -y mlp")
    os.system("pip uninstall -y electra_diet")
#     os.system("pip uninstall -y rasalt")
#     os.system('git clone -b master --single-branch https://SKT-AIDevOps:5ukojedx2dpxrghc63uyyzp75pddbe5jmrk53xdbnlydaap2p6va@dev.azure.com/SKT-AIDevOps/hera/_git/appliednl-mlp && cd appliednl-mlp && pip install -r requirements.txt --user && python setup.py install --user')
    os.system("git clone https://github.com/seujung/koelectra-rasa.git && cd koelectra-rasa && pip install -r requirements.txt && pip install .")


# dt = datetime.today().strftime('%Y%m%d')
    
# display("install packages")
# install_package()

from rasalt.utils import DataStoreUtil

display("Download nlu.md")
DataStoreUtil.downloadData('raw/rasa/data/latest/nlu.md','nlu.md')

display("Download nlu_goldenset.md")
DataStoreUtil.downloadData('raw/rasa/goldenset/latest/nlu_goldenset.md','nlu_goldenset.md')

display("Download nlu_adversarial.md")
DataStoreUtil.downloadData('raw/rasa/goldenset/latest/nlu_adversarial.md','nlu_adversarial.md')

display("Download nlu_regression.md")
DataStoreUtil.downloadData('raw/rasa/goldenset/latest/nlu_regression.md','nlu_regression.md')

display("Download labels.json")
DataStoreUtil.downloadData('raw/rasa/data/latest/labels.json','labels.json')

display("Get intent/entity label count")
with open('labels.json') as f:
    labels = json.load(f)

intent_class_num = len(labels['intent'])
entity_class_num = len(labels['entity']) * 2 + 1  ## consider BIO type

display("# intent :{}".format(intent_class_num))
display("# entity :{}".format(entity_class_num))

display("Train Model")

from electra_diet import trainer
trainer.train(
    file_path='./nlu.md',
    #training args
    train_ratio=0.8,
    batch_size=256,
    intent_class_num = intent_class_num,
    entity_class_num = entity_class_num,
    optimizer="AdamW",
    intent_optimizer_lr=3e-5,
    entity_optimizer_lr=4e-5,
    checkpoint_path='electra_diet_log',
    max_epochs=20,
    gpu_num=1,
    lower_text=True,
    early_stop=True,
    report_nm ="electra_report.json"
)
