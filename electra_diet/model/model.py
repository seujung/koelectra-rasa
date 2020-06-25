import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_electra import ElectraModel, ElectraConfig, ElectraPreTrainedModel


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, share_emb=None, dropout_p=0.1, max_length=128):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        if share_emb is not None:
            self.share_emb = True
        else:
            self.share_emb = False

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if share_emb is not None:
            out_dim = share_emb.shape[-1]
            self.embedding = nn.Embedding(self.output_size, out_dim)
            self.embedding.weight.data = share_emb
            self.dense = nn.Linear(out_dim, self.hidden_size, bias=False)
        else:
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        if self.share_emb:
            embedded = self.embedding(input.transpose(0, 1))
            embedded = self.dense(embedded)
        else:
            embedded = self.embedding(input.transpose(0, 1))
        embedded = self.dropout(embedded)
        embedded = embedded.transpose(0, 1)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)

        output = torch.cat((embedded, attn_applied.transpose(0, 1)), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, device, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)


class KoElectraModel(nn.Module):
    def __init__(self, intent_class_num, entity_class_num):
        super(KoElectraModel, self).__init__()

        # config = ElectraConfig.from_dict(config)
        # self.bert = ElectraModel(config)
        
        self.bert = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")
        config = self.bert.config
        self.pad_idx = config.pad_token_id
        ##For intent part
        self.intent_class_num = intent_class_num
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

        entity_output = self.entity_cls(self.dropout(outputs))

        cls_outout = self.intent_cls(self.dropout(outputs[:,0,:]))
        return cls_outout, entity_output


class KoElectraGenerationModel(nn.Module):
    def __init__(self, entity_class_num, share_emb=True):
        super(KoElectraGenerationModel, self).__init__()

        # config = ElectraConfig.from_dict(config)
        # self.bert = ElectraModel(config)
        
        self.bert = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")
        config = self.bert.config
        self.pad_idx = config.pad_token_id
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        ##For intent part
        if share_emb:
            self.attn_decoder = AttnDecoderRNN(config.hidden_size, 
                config.vocab_size, dropout_p=0.1, max_length=config.embedding_size) 
        else:
            emb_weight = self.bert.embeddings.word_embeddings.weight.data
            self.attn_decoder = AttnDecoderRNN(config.hidden_size, 
                config.vocab_size, dropout_p=0.1, max_length=config.embedding_size, share_emb=emb_weight) 


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
        hidden_state = outputs

        entity_output = self.entity_cls(self.dropout(outputs))

        decoder = self.attn_decoder
        return decoder, hidden_state, entity_output
