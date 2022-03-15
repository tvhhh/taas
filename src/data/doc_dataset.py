import os
import torch

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from torch.utils.data import Dataset
from typing import List, Optional


DEFAULT_CORPUS_PATH = "data/corpus"
DICT_FILE_NAME = "dict.txt"


class DocDataset(Dataset):
    def __init__(
        self,
        documents: List[List[str]],
        corpus_path: Optional[str] = None,
        dict_filter_no_below: int = 5,
        dict_filter_no_above: float = 0.5,
        max_vocab_size: int = 100000,
        use_tfidf: bool = False,
    ):
        self.documents = documents
        self.use_tfidf = use_tfidf
        
        corpus_path = corpus_path or os.path.join(os.getcwd(), DEFAULT_CORPUS_PATH)
        
        if not os.path.exists(os.path.join(corpus_path, DICT_FILE_NAME)):
            os.makedirs(corpus_path, exist_ok=True)
            
            self.dictionary = Dictionary(documents)
            self.dictionary.filter_extremes(
                no_below=dict_filter_no_below,
                no_above=dict_filter_no_above,
                keep_n=max_vocab_size,
            )
            self.dictionary.compactify()
            self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}
            
            self.dictionary.save_as_text(os.path.join(corpus_path, DICT_FILE_NAME))
        
        else:
            self.dictionary = Dictionary.load_from_text(os.path.join(corpus_path, DICT_FILE_NAME))
            self.dictionary.id2token = {v: k for k, v in self.dictionary.token2id.items()}
        
        self.bows = [self.dictionary.doc2bow(doc) for doc in self.documents]
        if self.use_tfidf:
            tfidf_model = TfidfModel(self.bows)
            self.tfidf = [tfidf_model[bow] for bow in self.bows]
        
        self.vocab_size = len(self.dictionary)
    
    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        vec = torch.zeros(self.vocab_size)

        ids, vals = [], []
        if self.use_tfidf:
            if len(self.tfidf[index]) > 0:
                ids, vals = zip(*self.tfidf[index])
        else:
            if len(self.bows[index]) > 0:
                ids, vals = zip(*self.bows[index])
        
        vec[list(ids)] = torch.tensor(list(vals)).float()
        return vec
