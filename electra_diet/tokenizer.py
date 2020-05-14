import torch
from transformers import ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-discriminator")

def tokenize(self, text: str, seq_len: int, padding: bool = True, return_tensor: bool = True):

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
                [pad_token_id] * (self.seq_len - len(tokens))
            )
            tokens = torch.cat((tokens, pad_tensor), 0)
            segment_ids = torch.tensor([0] * seq_len)  
    if return_tensor:
        return (tokens, segment_ids)
    else:
        return (tokens.numpy(), segment_ids.numpy())