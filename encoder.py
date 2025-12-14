from collections import Counter
import torch
class Encoder():
    def __init__(self,texts,max_vocab_size=10000,max_len=50):
        self.texts = texts
        self.max_vocab_size=max_vocab_size
        self.max_len= max_len
    def build_vocab(self):
        words = " ".join(self.texts).split()
        freq = Counter(words)
        self.vocab = {'<PAD>':0,"<UNK>":1}
        for idx, (word,_) in enumerate(freq.most_common(self.max_vocab_size - 2),start = 2)  :
            self.vocab[word] = idx
        return self.vocab
    
    def encode_text(self, text, device=None):
        tokens = text.split()
        ids = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in tokens]
        if len(ids) < self.max_len:
            ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        
        if device is not None:
            return torch.tensor(ids).to(device)
            
        return ids