import numpy as np
from electra_diet.dataset import ElectraDataset
file_path ='/Users/digit82_mac/git_repo/nlu_dataset/nlu_goldenset.md'


data = ElectraDataset(file_path)


dataset = data.dataset

# idx =30
idx = 54
text = dataset[idx]['text']
print(text)

data.__getitem__(idx)

input_ids = data.tokenizer.encode(text)
for i, t in enumerate(input_ids):
    print("i:{} token:{}".format(i, data.tokenizer.ids_to_tokens[t]))

entity_dict_bio = data.entity_dict_bio

def find_sub_list(sub_list,this_list):
    start_pos = this_list.index(sub_list[0])
    end_pos = start_pos + len(sub_list)
    return (start_pos, end_pos)


entity_idx = np.zeros(128)
for entity_info in dataset[idx]["entities"]:
    ##consider [CLS] token
    cur_entity_value = text[entity_info['start']:entity_info['end']]
    cur_entityt_idx = data.tokenizer.encode(cur_entity_value)[1:-1]
    
    (start_pos, end_pos) = find_sub_list(cur_entityt_idx, input_ids)

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

    

