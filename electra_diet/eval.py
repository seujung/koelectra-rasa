import os
import glob
from pytorch_lightning.callbacks.base import Callback
from electra_diet.metrics import show_intent_report, show_entity_report, show_intent_generation_report
from electra_diet.dataset.electra_dataset import ElectraDataset

class PerfCallback(Callback):
    def __init__(self, file_path=None, gpu_num=0, report_nm=None, output_dir=None, root_path=None):
        self.file_path = file_path
        if gpu_num > 0:
            self.cuda = True
        else:
            self.cuda = False
        self.report_nm = report_nm
        self.output_dir = output_dir
        if root_path is None:
            self.root_path = 'lightning_logs'
        else:
            self.root_path = os.path.join(root_path, 'lightning_logs')

    def on_train_end(self, trainer, pl_module):
        print("train finished")
        if self.file_path is None:
            dataset = pl_module.val_dataset
        else:
            dataset = ElectraDataset(file_path=self.file_path, tokenizer=None)
                
        if self.output_dir is None:
            folder_path = [f for f in glob.glob(os.path.join(self.root_path, "**/"), recursive=False)]
            folder_path.sort()
            self.output_dir  = folder_path[-1]
        self.output_dir = os.path.join(self.output_dir, 'results')
        intent_report_nm = self.report_nm.replace('.', '_intent.')
        entity_report_nm = self.report_nm.replace('.', '_entity.')

        if pl_module.hparams.use_generator:
            show_intent_generation_report(dataset, pl_module, file_name=intent_report_nm, output_dir=self.output_dir, cuda=self.cuda)
        else:
            show_intent_report(dataset, pl_module, file_name=intent_report_nm, output_dir=self.output_dir, cuda=self.cuda)
        show_entity_report(dataset, pl_module, file_name=entity_report_nm, output_dir=self.output_dir, cuda=self.cuda)