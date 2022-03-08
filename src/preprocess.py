import argparse
import gensim
import re
import spacy

from datasets import load_dataset, set_caching_enabled
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class DataPreprocessor:
    
    def __init__(self, dataset, args):

        self.args = args
        self.dataset = dataset
        
        self.nlp = spacy.load("en_core_web_sm")
        
        self.tokenizer = gensim.utils.tokenize
        self.lemmatizer = WordNetLemmatizer()
        
        self.stop_words = None
        if args.stop_words_path is not None:
            with open(args.stop_words_path, "r") as reader:
                raw_text = reader.read()
                self.stop_words = set(word.strip() for word in raw_text.split("\n"))
        else:
            self.stop_words = set(stopwords.words("english"))
    
    def _preprocess(self, src, progress=None):

        # Remove all non-ASCII characters
        src = re.sub(r"[^\x00-\x7F]+", "", src)

        # Process more-than-2-word entities
        # This should be considered because it is very time-consuming
        if self.args.tune_ne_words:
            doc = self.nlp(src)
            for ent in doc.ents:
                text, label = ent.text, ent.label_
                # https://spacy.io/api/annotation#named-entities
                considered_ent_types = ["PERSON", "NORP", "FAC", "GPE", "LOC"]
                if len(text.split()) > 1 and label in considered_ent_types:
                    # e.g. Hong Kong -> Hong_Kong
                    src = src.replace(text, "_".join(text.split()))
        
        # Tokenize the source text
        tokens = self.tokenizer(src)

        # Lemmatize the texts
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Remove all punctuations and numbers
        tokens = re.sub(r"[^a-zA-Z ]", "", " ".join(tokens)).split()

        # Lowercase and remove all stop words
        tokens = [token.lower() for token in tokens if token.lower() not in self.stop_words]

        if progress is not None:
            progress.update(1)
        
        return " ".join(tokens)
    
    def preprocess(self, splits=["train","validation","test"], save_to_disk=True):
        
        for s in splits:
            data_split = self.dataset[s]
            progress_bar = tqdm(range(len(data_split)))
            corpus = list(map(
                lambda src: self._preprocess(src, progress_bar),
                data_split[self.args.input_name],
            ))
            data_split = data_split.add_column("corpus", corpus)
            self.dataset[s] = data_split
        
        if save_to_disk:
            self.dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_path", type=str, default="cnn_dailymail")
    parser.add_argument("-dataset_name", type=str, default="3.0.0")
    parser.add_argument("-input_name", type=str, default="article")
    parser.add_argument("-label_name", type=str, default="highlights")
    parser.add_argument("-stop_words_path", type=str, default=None)
    parser.add_argument("-tune_ne_words", type=str2bool, nargs="?", default=True)
    parser.add_argument("-output_dir", type=str, default="data/cnndm")

    args = parser.parse_args()

    set_caching_enabled(False)
    dataset = load_dataset(args.dataset_path, args.dataset_name)

    data_builder = DataPreprocessor(dataset, args)
    data_builder.preprocess(save_to_disk=True)
