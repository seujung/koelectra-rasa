import os
import torch
import json
from transformers import ElectraTokenizer
from pathlib import Path

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")
path = Path(__file__).parent
# print(path)
with open(os.path.join(path, 'assets/token_converter.json')) as f:
    token_conver = json.load(f)

def get_tokenizer():
    return tokenizer

def tokenize(text: str, seq_len: int, padding: bool = True, return_tensor: bool = True, lower_text=True):
    if lower_text:
        text = text.lower()
        
    pad_token_id = tokenizer.pad_token_id
    tokens = tokenizer.encode(text)
    ##consider single token only
    segment_ids = [0] * len(tokens) 
    if type(tokens) == list:
        tokens = torch.tensor(tokens)
        
    if padding:
        if len(tokens) > seq_len:
            tokens = torch.tensor(tokens[:seq_len])
            segment_ids = torch.tensor(segment_ids[:seq_len])
        else:
            pad_tensor = torch.tensor(
                [pad_token_id] * (seq_len - len(tokens))
            )
            tokens = torch.cat((tokens, pad_tensor), 0)
            segment_ids = torch.tensor([0] * seq_len)  
    if return_tensor:
        return (tokens, segment_ids)
    else:
        return (tokens.numpy(), segment_ids.numpy())


def delete_josa(token):
    if token in token_conver.keys():
        return token_conver[token]
    else:
        return token