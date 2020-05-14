

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