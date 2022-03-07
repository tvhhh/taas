import argparse
import re
from collections import defaultdict
from datasets import load_dataset, set_caching_enabled

import nltk
nltk.download("averaged_perceptron_tagger")
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from tqdm.auto import tqdm


stop_words = None
def _preprocess_corpus(args, src, progress=None):
    # Remove stop words
    global stop_words
    if stop_words is None:
        if args.stop_words_path is not None:
            with open(args.stop_words_path, "r") as reader:
                raw_text = reader.read()
                stop_words = set(word.strip() for word in raw_text.split("\n"))
        else:
            stop_words = set(stopwords.words("english"))
    words = word_tokenize(src)
    words = [w.lower() for w in words if w.lower() not in stop_words]

    # Remove punctuations and numbers
    words = re.sub(r"[^a-zA-Z ]", "", " ".join(words)).split()

    # Lemmatize
    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map["J"] = wordnet.ADJ
    tag_map["V"] = wordnet.VERB
    tag_map["R"] = wordnet.ADV
    lmtz = WordNetLemmatizer()
    words = [lmtz.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(words)]

    if progress is not None:
        progress.update(1)

    return words


def preprocess(args, dataset):
    progress_bar = tqdm(range(len(dataset)))
    corpus = map(
        lambda src: _preprocess_corpus(args, src, progress_bar),
        dataset[args.input_name]
    )
    dataset = dataset.add_column("corpus", list(corpus))

    return dataset


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_path", type=str, default="cnn_dailymail")
    parser.add_argument("-dataset_name", type=str, default="3.0.0")
    parser.add_argument("-input_name", type=str, default="article")
    parser.add_argument("-label_name", type=str, default="highlights")
    parser.add_argument("-stop_words_path", type=str, default=None)
    parser.add_argument("-output_dir", type=str, default="data/cnndm")

    args = parser.parse_args()

    set_caching_enabled(False)
    dataset = load_dataset(args.dataset_path, args.dataset_name)

    for s in ("train", "validation", "test"):
        dataset[s] = preprocess(args, dataset[s])
    
    dataset.save_to_disk(args.output_dir)
