import numpy as np
from electra_diet.dataset import ElectraDataset
# file_path ='/Users/digit82_mac/git_repo/nlu_dataset/nlu_goldenset.md'
file_path ='/Users/digit82_mac/git_repo/nlu_dataset/nlu.md'

data = ElectraDataset(file_path)

from tqdm import trange
for idx in trange(len(data)):
    o = data.__getitem__(idx)


from electra_diet.tokenizer import get_tokenizer

tokenizer = get_tokenizer()

# idx =30
idx = 24325
dataset = data.dataset
text = dataset[idx]['text']
print(text)

data.__getitem__(idx)

text_token = data.tokenizer.encode(text)
for i, t in enumerate(text_token):
    print("i:{} token:{}".format(i, data.tokenizer.ids_to_tokens[t]))

entity_dict_bio = data.entity_dict_bio

def find_sub_list(sub_list,this_list):
    if set(sub_list).issubset(set(this_list)):
        start_pos = this_list.index(sub_list[0])
        end_pos = start_pos + len(sub_list)
        return (start_pos, end_pos)
    else:
        ##position이 없는 경우
        return (-1, -1)


entity_idx = np.zeros(128)
for entity_info in dataset[idx]["entities"]:
    add_token = 0
    flag = -1
    ## [데이터를] 이 하나의 토큰으로 인식됨 --> 조사 추가
    while flag < 0:
        base_cur_entity_value = text[entity_info['start']:entity_info['end']]
        cur_entity_value = text[entity_info['start']:entity_info['end'] + add_token]
        cur_entity_idx = data.tokenizer.encode(cur_entity_value)[1:-1]

        (start_pos, end_pos) = find_sub_list(cur_entity_idx, text_token)
        flag = start_pos + end_pos
        ## Case 2: token에 조사가 포함된 경우
        add_token += 1
        if add_token > len(text_token):
            ## Case 3: token이 분리된 경우
            isin_token = ''
            for t in text.split(' '):
                if base_cur_entity_value in t:
                    isin_token = t
            partial_idx = data.tokenizer.encode(isin_token)[1:-1]
            ##token이 검출되지 않은 경우
            if len(partial_idx) == 0:
                raise Exception('please check the entity value.! current text is {}'.format(cur_entity_value)) 

            cur_entity_idx = []
            for i in partial_idx:
                partial_token = data.tokenizer.ids_to_tokens[i].replace('#','')
                if partial_token in base_cur_entity_value:
                    print(partial_token)
                    cur_entity_idx.append(i)
            if len(cur_entity_idx) == 0:
                raise Exception('please check the entity value.! current text is {}'.format(cur_entity_value)) 

            (start_pos, end_pos) = find_sub_list(cur_entity_idx, text_token)
            flag = start_pos + end_pos
            

    begin = -1
    begin_tag = 'B-'
    mid_tag = 'I-'
    cur_entity = entity_info['entity']
    for i in range(start_pos, end_pos):
        if begin < 0:
            entity_idx[i] = entity_dict_bio[begin_tag + cur_entity]
            begin += 1
        else:
            entity_idx[i] = entity_dict_bio[mid_tag + cur_entity]

    

## konlpy 확인

from konlpy.tag import Twitter
from konlpy.tag import Kkma, Komoran
twitter = Twitter()
twitter.pos(text)

kkma = Kkma()
kkma.pos(text)

komoran = Komoran()
##josa index
josa_list = ['JC', 'JKB', 'JKC', 'JKG', 'JKO', 'JKQ', ' JKS', 'JKV', 'JX']
text = "데이터를"
pos = komoran.pos(text)
for (k, v) in pos:
    if v in josa_list:
        text = text.replace(k, '')

entity_dict = dict()
for k, v in data.entity_dict_bio.items():
    entity_dict[int(v)] = k


######################################
## test infer
import re
from konlpy.tag import Komoran
komoran = Komoran()
def delete_josa(text):
    josa_list = ['JC', 'JKB', 'JKC', 'JKG', 'JKO', 'JKQ', ' JKS', 'JKV', 'JX']
    pos = komoran.pos(text)
    for (k, v) in pos:
        if v in josa_list:
            text = text.replace(k, '')
    return text

# idx =30
idx = 24325
dataset = data.dataset
out = data.__getitem__(idx)

text = "010-1234-5678에다가 1월부터1GB를 매달 주기로 보내줘"


# mapping entity result
entities = []

entity_out = out[2].numpy()
entity_output = dict()

input_token, _ = data.tokenize(text)
input_token = input_token.numpy()

for i in input_token:
    print(data.tokenizer.ids_to_tokens[i])

entity_val = []
entity_typ = ''
entity_pos = dict()

for i, e in enumerate(entity_out):
    e = int(e)

    if e > 0:
        ##get index info
        entity_label = entity_dict[e]
        pos, typ = entity_label.split('-')
        if pos == 'B':
            entity_val = []
            entity_val.append(input_token[i])
            entity_typ = typ
        
        elif typ == entity_typ:
            entity_val.append(input_token[i])
    else:
        if len(entity_val) > 0:
            value = data.tokenizer.decode(entity_val)
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
    

            
## load vocab file
import pandas as pd
from konlpy.tag import Komoran
from tqdm import tqdm
komoran = Komoran()
josa_list = ['JC', 'JKB', 'JKC', 'JKG', 'JKO', 'JKQ', ' JKS', 'JKV', 'JX']

f = open("/Users/digit82_mac/vocab.txt", 'r')
pos_dict = dict()

lines = f.readlines()
for line in tqdm(lines):
    line = line.replace('\n', '')
    index = 0
    pos = komoran.pos(line)
    for (k, v) in pos:
        if v in josa_list:
            index += 1
    if index > 0:
        pos_dict[line] = pos

f.close()

len(pos_dict.keys())

output = pd.DataFrame(pos_dict.items(), columns=['text', 'pos'])

output.to_csv('pos_out.tsv', sep='\t')

##filter file load
filter_df  = pd.read_csv('~/git_repo/pos_out_filter.tsv', sep='\t')
filter_df = filter_df.dropna().reset_index(drop=True)

filter_text = filter_df['text'].tolist()

komoran = Komoran()

def delete_josa(text):
    josa_list = ['JC', 'JKB', 'JKC', 'JKG', 'JKO', 'JKQ', ' JKS', 'JKV', 'JX']
    pos = komoran.pos(text)
    for (k, v) in pos:
        if v in josa_list:
            text = text.replace(k, '')
    return text

filter_dict = dict()

for text in filter_text:
    rep_text = delete_josa(text)
    filter_dict[text] = rep_text

import json
with open('token_converter.json', 'w') as fp:
    json.dump(filter_dict, fp)
    