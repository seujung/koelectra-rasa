import torch
import numpy as np
import re

from collections import OrderedDict
from tqdm import tqdm
from typing import List
from transformers import ElectraTokenizer

def find_sub_list(sub_list,this_list):
    if set(sub_list).issubset(set(this_list)):
        start_pos = this_list.index(sub_list[0])
        end_pos = start_pos + len(sub_list)
        return (start_pos, end_pos)
    else:
        return (-1, -1)


class ElectraDataset(torch.utils.data.Dataset):
    """
    RASA NLU markdown file lines based Custom Dataset Class

    Dataset Example in nlu.md

    ## intent:intent_데이터_자동_선물하기_멀티턴                <- intent name
    - T끼리 데이터 주기적으로 보내기                            <- utterance without entity
    - 인터넷 데이터 [달마다](Every_Month)마다 보내줄 수 있어?    <- utterance with entity
    
    """

    def __init__(
        self,
        file_path: str,
        seq_len=128,
        tokenizer=None,
        intent_dict=None,
        entity_dict=None,
        tag_type='bio'
    ):
        self.intent_dict = {}
        self.entity_dict_bio = {}
        self.entity_dict_bio[
            "O"
        ] = 0  # based on XO tagging(one entity_type has assigned to one class)
        
        self.entity_dict = {}
        self.entity_dict[
            "O"
        ] = 0  # based on XO tagging(one entity_type has assigned to one class)

        self.dataset = []
        self.seq_len = seq_len
        self.tag_type = tag_type.lower()

        if tokenizer is None:
            self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-discriminator")
        else:
            self.tokenizer = tokenizer
        
        self.pad_token_id = self.tokenizer.pad_token_id
        
        markdown_lines = open(file_path, encoding="utf-8").readlines()

        intent_value_list = []
        entity_type_list = []
        current_intent_focus = ""

        for line in tqdm(
            markdown_lines,
            desc="Organizing Intent & Entity dictionary in NLU markdown file ...",
        ):
            if len(line.strip()) < 2:
                current_intent_focus = ""
                continue

            if "## " in line:
                if "intent:" in line:
                    intent_value_list.append(line.split(":")[1].strip())
                    current_intent_focus = line.split(":")[1].strip()
                else:
                    current_intent_focus = ""

            else:
                if current_intent_focus != "":
                    text = line[2:].strip()

                    for type_str in re.finditer(r"\([a-zA-Z_1-2]+\)", text):
                        entity_type = (
                            text[type_str.start() + 1 : type_str.end() - 1]
                            .replace("(", "")
                            .replace(")", "")
                        )
                        entity_type_list.append(entity_type)

        intent_value_list = sorted(intent_value_list)
        for intent_value in intent_value_list:
            if intent_value not in self.intent_dict.keys():
                self.intent_dict[intent_value] = len(self.intent_dict)

        entity_type_list = sorted(list(set(entity_type_list)))
        for entity_type in entity_type_list:
            self.entity_dict_bio['B-'+entity_type] = len(self.entity_dict_bio)
            self.entity_dict_bio['I-'+entity_type] = len(self.entity_dict_bio)
            self.entity_dict[entity_type] = len(self.entity_dict)

        current_intent_focus = ""

        for line in tqdm(
            markdown_lines, desc="Extracting Intent & Entity in NLU markdown files...",
        ):
            if len(line.strip()) < 2:
                current_intent_focus = ""
                continue

            if "## " in line:
                if "intent:" in line:
                    current_intent_focus = line.split(":")[1].strip()
                else:
                    current_intent_focus = ""
            else:
                if current_intent_focus != "":  # intent & entity sentence occur case
                    text = line[2:]

                    entity_value_list = []
                    for value in re.finditer(r"\[(.*?)\]", text):
                        entity_value_list.append(
                            text[value.start() + 1 : value.end() - 1]
                            .replace("[", "")
                            .replace("]", "")
                        )

                    entity_type_list = []
                    for type_str in re.finditer(r"\([a-zA-Z_1-2]+\)", text):
                        entity_type = (
                            text[type_str.start() + 1 : type_str.end() - 1]
                            .replace("(", "")
                            .replace(")", "")
                        )
                        entity_type_list.append(entity_type)

                    text = re.sub(r"\([a-zA-Z_1-2]+\)", "", text)
                    text = text.replace("[", "").replace("]", "")

                    each_data_dict = {}
                    each_data_dict["text"] = text.strip()
                    each_data_dict["intent"] = current_intent_focus
                    each_data_dict["intent_idx"] = self.intent_dict[
                        current_intent_focus
                    ]
                    each_data_dict["entities"] = []

                    for value, type_str in zip(entity_value_list, entity_type_list):
                        try:
                            for entity in re.finditer(value, text):
                                each_data_dict["entities"].append(
                                    {
                                        "start": entity.start(),
                                        "end": entity.end(),
                                        "entity": type_str,
                                        # "entity_idx": self.entity_dict[type_str],
                                    }
                                )
                        except Exception as ex:
                            print(f"error occured : {ex}")
                            print(f"value: {value}")
                            print(f"text: {text}")

                    self.dataset.append(each_data_dict)

        print(f"Intents: {self.intent_dict}")
        if self.tag_type == 'bio':
            print(f"Entities: {self.entity_dict_bio}")
        else:
            print(f"Entities: {self.entity_dict}")


    def tokenize(self, text: str, padding: bool = True, return_tensor: bool = True):
        tokens = self.tokenizer.encode(text)
        ##consider single token only
        segment_ids = [0] * len(tokens)

        if type(tokens) == list:
            tokens = torch.tensor(tokens)
            
        if padding:
            if len(tokens) > self.seq_len:
                tokens = torch.tensor(tokens[:self.seq_len])
                segment_ids = torch.tensor(segment_ids[:self.seq_len])
            else:
                pad_tensor = torch.tensor(
                    [self.pad_token_id] * (self.seq_len - len(tokens))
                )
                tokens = torch.cat((tokens, pad_tensor), 0)
                segment_ids = torch.tensor([0] * self.seq_len)

        if return_tensor:
            return (tokens, segment_ids)
        else:
            return (tokens.numpy(), segment_ids.numpy())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (tokens, segment_ids) = self.tokenize(self.dataset[idx]["text"])

        intent_idx = torch.tensor([self.dataset[idx]["intent_idx"]])

        text = self.dataset[idx]["text"]
        text_token = self.tokenizer.encode(text)
        entity_idx = np.zeros(self.seq_len)
        for entity_info in self.dataset[idx]["entities"]:
            add_token = 0
            flag = -1
            ## [데이터를] 이 하나의 토큰으로 인식됨 --> 조사 추가
            while flag < 0:
                base_cur_entity_value = text[entity_info['start']:entity_info['end']]
                cur_entity_value = text[entity_info['start']:entity_info['end'] + add_token] ##원래 설정된 entity value
                cur_entityt_idx = self.tokenizer.encode(cur_entity_value)[1:-1]
                (start_pos, end_pos) = find_sub_list(cur_entityt_idx, text_token)
                flag = start_pos + end_pos
                add_token += 1
                if add_token > len(text_token):
                    ## Case 3: token이 분리된 경우
                    isin_token = ''
                    for t in text.split(' '):
                        if base_cur_entity_value in t:
                            isin_token = t
                    partial_idx = self.tokenizer.encode(isin_token)[1:-1]
                    ##token이 검출되지 않은 경우 에러 발생
                    if len(partial_idx) == 0:
                        raise Exception('please check the entity value.! current text is {}'.format(cur_entity_value)) 
                    
                    cur_entity_idx = []
                    # partial token 검색
                    for i in partial_idx:
                        partial_token = self.tokenizer.ids_to_tokens[i].replace('#','')
                        if partial_token in base_cur_entity_value:
                            # print(partial_token)
                            cur_entity_idx.append(i)
                    if len(cur_entity_idx) == 0:
                        raise Exception('please check the entity value.! current text is {}'.format(cur_entity_value)) 
                    
                    (start_pos, end_pos) = find_sub_list(cur_entity_idx, text_token)
                    flag = start_pos + end_pos

            if self.tag_type == 'bio':
                begin = -1
                begin_tag = 'B-'
                mid_tag = 'I-'
                cur_entity = entity_info['entity']
                for i in range(start_pos, end_pos):
                    if begin < 0:
                        entity_idx[i] = self.entity_dict_bio[begin_tag + cur_entity]
                        begin += 1
                    else:
                        entity_idx[i] = self.entity_dict_bio[mid_tag + cur_entity]
            else:
                begin = -1
                cur_entity = entity_info['entity']
                for i in range(start_pos, end_pos):
                    if begin < 0:
                        entity_idx[i] = self.entity_dict[cur_entity]
                        begin += 1
                    else:
                        entity_idx[i] = self.entity_dict[cur_entity]
        entity_idx = torch.from_numpy(entity_idx)
        # for entity_info in self.dataset[idx]["entities"]:
        #     ##consider [CLS] token
        #     for i in range(entity_info["start"] + 1, entity_info["end"] + 2):
        #         entity_idx[i] = entity_info["entity_idx"]
        # entity_idx = torch.from_numpy(entity_idx)

        return (tokens, segment_ids), intent_idx, entity_idx

    def get_intent_idx(self):
        return self.intent_dict

    def get_entity_idx(self):
        # return self.entity_dict
        if self.tag_type == 'bio':
            return self.entity_dict_bio
        else:
            return self.entity_dict


    def get_vocab_size(self):
        if self.tokenizer is not None:
            return len(self.tokenizer)

        return len(self.encoder.vocab)

    def get_seq_len(self):
        return self.seq_len
