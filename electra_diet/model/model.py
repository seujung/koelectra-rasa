import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_electra import ElectraModel, ElectraConfig, ElectraPreTrainedModel


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=24):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class KoElectraModel(nn.Module):
    def __init__(self, intent_class_num, entity_class_num, intent_word_len):
        super(KoElectraModel, self).__init__()

        # config = ElectraConfig.from_dict(config)
        # self.bert = ElectraModel(config)
        self.bert = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")
        config = self.bert.config
        self.pad_idx = config.pad_token_id
        ##For intent part
        self.intent_class_num = intent_class_num
        if intent_class_num is not None:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.intent_cls = nn.Linear(config.hidden_size, intent_class_num)
        else:
            self.attn_decoder = AttnDecoderRNN(config.hidden_size, config.vocab_size, dropout_p=0.1, max_length=intent_word_len)
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

        entity_output = self.entity_cls(self.dropout(outputs))

        if self.intent_class_num is not None:
            cls_outout = self.intent_cls(self.dropout(outputs[:,0,:]))
            return cls_outout, entity_output
        else:
            decoder = self.attn_decoder()
            return decoder, hidden_state, entity_output
