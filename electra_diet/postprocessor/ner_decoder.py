
import re
from electra_diet.tokenizer import get_tokenizer, delete_josa

class NERDecoder(object):

    def __init__(self, entity_dict:dict, tokenizer):
        self.entity_dict = entity_dict
        self.tokenizer = get_tokenizer()

    def process(self, input_token, entity_indices):
        # mapping entity result
        entities = []

        entity_val = []
        entity_typ = ''
        entity_pos = dict()
        text = self.tokenizer.decode(input_token, skip_special_tokens=True)

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
                        value = self.tokenizer.decode(entity_val)
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
                    value = self.tokenizer.decode(entity_val, skip_special_tokens=True)
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
        
        return entities
                

