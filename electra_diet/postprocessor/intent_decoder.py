import torch
import numpy as np
from electra_diet.tokenizer import get_tokenizer

class IntentDecoder(object):

    def __init__(self, target_length, intent_decoder, encoder_outputs):
        self.tokenizer = get_tokenizer()
        self.target_length = target_length
        self.intent_decoder = intent_decoder
        self.batch_size = encoder_outputs.shape[0]
        self.device = encoder_outputs.device
        self.encoder_outputs = encoder_outputs
        self.tokenizer = get_tokenizer()
        self.BOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'

    def process(self):
        intent_pred = []
        bos_index = self.tokenizer.convert_tokens_to_ids(self.BOS_TOKEN)
        decoder_input = torch.tensor([[bos_index] * self.batch_size]).to(self.device)
        decoder_hidden = self.intent_decoder.initHidden(device=self.device, batch_size=self.batch_size)
        
        for di in range(self.target_length):
            decoder_output, decoder_hidden, decoder_attention = self.intent_decoder(
                decoder_input, decoder_hidden, self.encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.transpose(0, 1).detach()  # 입력으로 사용할 부분을 히스토리에서 분리
            intent_pred.append(decoder_output.unsqueeze(1))
        
        intent_pred = torch.cat(intent_pred, 1)   
        intent_token = intent_pred.argmax(-1)

        intent_results = []
        for tok in intent_token:
            intent_result = self.tokenizer.decode(tok, skip_special_tokens=True)
            intent_result = intent_result.replace(self.EOS_TOKEN, '')
            intent_result = intent_result.replace(' ', '')
            intent_results.append(intent_result)
        
        return intent_results



def convert_intent_to_id(intent_results, intent_labels, fallback_intent='intent_미지원'):
    intent_label = []
    for intent in intent_results:
        intent_label.append(intent_labels[intent])
    
    return np.array(intent_label)

    