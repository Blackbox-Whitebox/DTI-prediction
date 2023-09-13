import json
import string

class Vocab:
    def __init__(self, tokens): 
        special_tokens = ["PAD", "SOS", "EOS"]
        self.tokens = tokens + special_tokens
        self.token_ix = {t:i for i, t in enumerate(self.tokens)}
        self.ix_token = {i:t for i,t in enumerate(self.tokens)}

    def encode(self, seq, max_len=None):
        encoded = [self.token_ix["SOS"]] + [self.token_ix[t] for t in seq] + [self.token_ix["EOS"]]
        if max_len:
            if len(encoded) < max_len:
                encoded += [self.token_ix["PAD"]]*(max_len-len(encoded))
                
        return encoded
                
    def decode(self, seq):
        return [self.ix_token[t] for t in seq]

    def __len__(self):
        return len(self.tokens)

class ProteinVocab(Vocab):
    def __init__(self):
        # 20 amino acids
        tokens = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'X']
        super().__init__(tokens)
        
class SMILEVocab(Vocab):
    def __init__(self):
        tokens = []
        tokens += list(string.digits) 
        tokens += list(string.ascii_letters)
        tokens += list(string.punctuation)
        tokens += [" "]
        
        super().__init__(tokens)