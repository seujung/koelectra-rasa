import torch
import torch.nn as nn
from electra_diet.pl_model import KoELECTRAClassifier, KoELECTRAGenClassifier
from electra_diet.postprocessor.intent_decoder import IntentDecoder, convert_intent_to_id
from electra_diet.tokenizer import tokenize, get_tokenizer, delete_josa
import re

import logging

model = None
intent_dict = {}
entity_dict = {}

class Inferencer:
    def __init__(self, checkpoint_path: str):
        try:
            self.model = KoELECTRAGenClassifier.load_from_checkpoint(checkpoint_path)
        except:
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

        self.use_generator = self.model.hparams.use_generator

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
        if self.use_generator:
            target_length = self.model.hparams.intent_label_len
            intent_decoder, encoder_outputs, entity_result = self.model.forward(*tokens)
            decoder = IntentDecoder(target_length, intent_decoder, encoder_outputs)
            intent_results = decoder.process()
            
            index = convert_intent_to_id(intent_results, self.intent_dict, fallback_intent='intent_미지원')
            intent = {}
            intent_ranking = []
            intent['name'] = self.intent_dict[index[0].item()]
            intent['confidence'] = 1.0
            intent_ranking.append(intent)

        else:
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
        _, entity_indices = torch.max(entity_result[0], dim=1)
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
#                 print("pos:{} typ:{}".format(pos, typ))
                if pos == 'B':
                    ##최초로 B- entity가 발생한 경우
                    if len(entity_val) == 0:
                        entity_val = []
                        entity_val.append(input_token[i])
                        entity_typ = typ
                        
                    ##이전에 B- entity가 존재한 경우
                    else:
                        ##update previous entity
                        value = tokenizer.decode(entity_val)
                        value = delete_josa(value).replace('#', '')
                        # value = value.replace('#', '')
                        entity_pos[value] = entity_typ
                        entity_val = []
                        ##add current entity
                        entity_val.append(input_token[i])
                        entity_typ = typ
                        
                ## 동일한 Entity의 I- label인 경우
                elif pos == 'I' and typ == entity_typ:
                    entity_val.append(input_token[i])
            
            ## O token인 경우
            else:
                if len(entity_val) > 0:
                    value = tokenizer.decode(entity_val)
                    value = delete_josa(value).replace('#', '')
                    # value = value.replace('#', '')
                    entity_pos[value] = entity_typ
                    entity_val = []

        # ## For debug type
        # print(entity_pos)
        for value, typ in entity_pos.items():
            m = re.search(value, text)
            try:
                start_idx, end_idx = m.span()
                entities.append(
                                {
                                    "start": start_idx,
                                    "end": end_idx,
                                    "value": value,
                                    "entity": typ
                                }
                            )
            except:
                pass




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
