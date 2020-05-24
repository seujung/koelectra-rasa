import torch
import torch.nn as nn
from electra_diet.pl_model import KoELECTRAClassifier
from electra_diet.tokenizer import tokenize, get_tokenizer
from konlpy.tag import Komoran
import re

import logging

model = None
intent_dict = {}
entity_dict = {}
komoran = Komoran()

def delete_josa(text):
    josa_list = ['JC', 'JKB', 'JKC', 'JKG', 'JKO', 'JKQ', ' JKS', 'JKV', 'JX']
    pos = komoran.pos(text)
    for (k, v) in pos:
        if v in josa_list:
            text = text.replace(k, '')
    return text


class Inferencer:
    def __init__(self, checkpoint_path: str):
        self.model = KoELECTRAClassifier.load_from_checkpoint(checkpoint_path)
        self.model.model.eval()

        self.intent_dict = {}
        for k, v in self.model.hparams.intent_label.items():
            self.intent_dict[int(k)] = v

        self.entity_dict = {}
        for k, v in self.model.hparams.entity_label.items():
            self.entity_dict[int(k)] = v

        logging.info('intent dictionary')
        logging.info(self.intent_dict)

        logging.info('entity dictionary')
        logging.info(self.entity_dict)

    def inference(self, text: str, intent_topk=5):
        if self.model is None:
            raise ValueError(
                "model is not loaded, first call load_model(checkpoint_path)"
            )
        tokenizer = get_tokenizer()
        tokens_tmp = tokenize(text, self.model.hparams.seq_len)
        tokens = []
        for t in tokens_tmp:
            tokens.append(t.unsqueeze(0))

        tokens = tuple(tokens)
        
        intent_result, entity_result = self.model.forward(*tokens)

        # mapping intent result
        rank_values, rank_indicies = torch.topk(
            nn.Softmax(dim=1)(intent_result)[0], k=intent_topk
        )
        intent = {}
        intent_ranking = []
        for i, (value, index) in enumerate(
            list(zip(rank_values.tolist(), rank_indicies.tolist()))
        ):
            intent_ranking.append({"confidence": value, "name": self.intent_dict[index]})

            if i == 0:
                intent["name"] = self.intent_dict[index]
                intent["confidence"] = value

        # mapping entity result
        entities = []

        # except first sequnce token whcih indicate BOS token
        _, entity_indices = torch.max((entity_result)[0][1:,:], dim=1)
        entity_indices = entity_indices.tolist()

        input_token, _ = tokens_tmp
        input_token = input_token.numpy()

        entity_val = []
        entity_typ = ''
        entity_pos = dict()

        for i, e in enumerate(entity_indices):
            e = int(e)

            if e > 0:
                ##get index info
                entity_label = self.entity_dict[e]
                pos, typ = entity_label.split('-')
                if pos == 'B':
                    entity_val = []
                    entity_val.append(input_token[i])
                    entity_typ = typ

                elif typ == entity_typ:
                    entity_val.append(input_token[i])
            else:
                if len(entity_val) > 0:
                    value = tokenizer.decode(entity_val)
                    value = delete_josa(value).replace('#', '').replace(' ','')
                    entity_pos[value] = entity_typ
        for value, typ in entity_pos.items():
            m = re.search(value, text)
            start_idx, end_idx = m.span()
            entities.append(
                            {
                                "start": start_idx,
                                "end": end_idx,
                                "value": value,
                                "entity": typ
                            }
                        )

        # start_idx = -1
        # for i, char_idx in enumerate(entity_indices):
        #     if char_idx != 0 and start_idx == -1:
        #         start_idx = i
        #     elif i > 0 and entity_indices[i-1] != entity_indices[i]:
        #         end_idx = i - 1
        #         entities.append(
        #             {
        #                 "start": max(start_idx,0),
        #                 "end": end_idx,
        #                 "value": text[max(start_idx, 0) : end_idx + 1],
        #                 "entity": self.entity_dict[entity_indices[i - 1]],
        #             }
        #         )
        #         start_idx = -1


        return {
            "text": text,
            "intent": intent,
            "intent_ranking": intent_ranking,
            "entities": entities,
        }

        # rasa NLU entire result format
        """
        {
            "text": "Hello!",
            "intent": {
                "confidence": 0.6323,
                "name": "greet"
            },
            "intent_ranking": [
                {
                    "confidence": 0.6323,
                    "name": "greet"
                }
            ],
            "entities": [
                {
                    "start": 0,
                    "end": 0,
                    "value": "string",
                    "entity": "string"
                }
            ]
        }
        """
