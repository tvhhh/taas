import os

import torch
from torch.utils.data import Dataset

from gensim.corpora import Dictionary


class BowDataset(Dataset):
    def __init__(
        self,
        documents,
        dict_path=None,
        dict_filter_no_below=5,
        dict_filter_no_above=0.5,
        max_vocab_size=200000,
    ):
        self.documents = documents
        
        dict_path = dict_path or os.path.join(os.getcwd(), "data/corpus/dict.txt")
        if not os.path.exists(dict_path):
            if not os.path.exists(os.path.dirname(dict_path)):
                os.makedirs(os.path.dirname(dict_path))
            self.dictionary = Dictionary(documents)
            self.dictionary.filter_extremes(
                no_below=dict_filter_no_below,
                no_above=dict_filter_no_above,
                keep_n=max_vocab_size,
            )
            self.dictionary.compactify()
            self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}
            self.dictionary.save_as_text(dict_path)
        else:
            self.dictionary = Dictionary.load_from_text(dict_path)
            self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}
        
        self.vocab_size = len(self.dictionary)
    
    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        bow = torch.zeros(self.vocab_size)
        token_freqs = self.dictionary.doc2bow(self.documents[index])
        ids, freq = zip(*token_freqs)
        bow[list(ids)] = torch.tensor(list(freq)).float()
        return bow
