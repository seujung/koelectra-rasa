import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_electra import ElectraModel, ElectraConfig, ElectraPreTrainedModel


class KoElectraModel(nn.Module):
    def __init__(self, intent_class_num, entity_class_num):
        super(KoElectraModel, self).__init__()

        # config = ElectraConfig.from_dict(config)
        # self.bert = ElectraModel(config)
        self.bert = ElectraModel.from_pretrained("monologg/koelectra-small-discriminator")
        config = self.bert.config
        self.pad_idx = config.pad_token_id
        ##For intent part
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_cls = nn.Linear(config.hidden_size, intent_class_num)
        ##For entity part
        self.entity_cls = nn.Linear(config.hidden_size, entity_class_num)


    def forward(self, input_ids, token_type_ids):
        attention_mask = input_ids.ne(self.pad_idx).float()
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        outputs = outputs[0]

        cls_outout = self.intent_cls(self.dropout(outputs[:,0,:]))
        entity_output = self.entity_cls(self.dropout(outputs))


        return cls_outout, entity_output
