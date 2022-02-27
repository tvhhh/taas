import argparse
import numpy as np
import re

from datasets import load_dataset, set_caching_enabled
from rouge_score import rouge_scorer

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict


SUS_INPUT_PREFIX = "sus-formatted"
EXT_LABEL_PREFIX = "extractive"


def _preprocess_corpus(src):
    # Remove special characters
    src = re.sub(r"[^a-zA-Z0-9 ]", "", src)
    
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(src)
    words = [w.lower() for w in words if w.lower() not in stop_words]

    # Lemmatize
    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map['J'] = wordnet.ADJ
    tag_map['V'] = wordnet.VERB
    tag_map['R'] = wordnet.ADV

    lmtz = WordNetLemmatizer()
    words = [lmtz.lemmatize(word, tag_map[tag[0]]) for word, tag in pos_tag(words)]

    processed_txt = " ".join(words)

    return processed_txt


def _format_to_sus(src, cls_token="<cls>", sep_token="<sep>"):
    sents = nltk.sent_tokenize(src)
    joint_sents = f" {sep_token} {cls_token} ".join(sents)
    text = f"{cls_token} {joint_sents} {sep_token}"
    return text


def _greedy_select_extractive_label(src, tgt, summary_size=3, metrics=["rouge1","rouge2"]):
    def _rouge_clean(src):
        return re.sub(r"[^a-zA-Z0-9 ]", "", src)
    
    sents = nltk.sent_tokenize(src)
    sents = [_rouge_clean(s) for s in sents]
    
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=False)
    def _cal_sent_score(tgt, src):
        scores = scorer.score(tgt, src)
        return sum(scores[m].fmeasure for m in scores.keys())
    
    ext_sum = str()
    selected = []
    curr_score = 0
    for _ in range(summary_size):
        scores = [
            0 if i in selected else
            _cal_sent_score(tgt, " ".join([ext_sum, s]))
            for i, s in enumerate(sents)
        ]
        max_score, selected_idx = np.max(scores), np.argmax(scores)
        if max_score <= curr_score:
            break
        ext_sum = " ".join([ext_sum, sents[selected_idx]])
        selected.append(selected_idx)
        curr_score = max_score
    
    return selected


def preprocess(dataset, input_name, label_name): 
    corpus = map(
        lambda src: _preprocess_corpus(src),
        dataset[input_name]
    )
    dataset = dataset.add_column(f"corpus", list(corpus))

    formatted_input = map(
        lambda src: _format_to_sus(src),
        dataset[input_name]
    )
    dataset = dataset.add_column(f"{SUS_INPUT_PREFIX} {input_name}", list(formatted_input))

    ext_labels = map(
        lambda src, tgt: _greedy_select_extractive_label(src, tgt),
        dataset[input_name],
        dataset[label_name],
    )
    dataset = dataset.add_column(f"{EXT_LABEL_PREFIX} {label_name}", list(ext_labels))
    
    return dataset


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_path", type=str, default="cnn_dailymail")
    parser.add_argument("-dataset_name", type=str, default="3.0.0")
    parser.add_argument("-input_name", type=str, default="article")
    parser.add_argument("-label_name", type=str, default="highlights")
    parser.add_argument("-output_dir", type=str, default="data/cnndm")

    args = parser.parse_args()

    set_caching_enabled(False)
    dataset = load_dataset(args.dataset_path, args.dataset_name)

    for s in ["train", "validation", "test"]:
        dataset[s] = preprocess(dataset[s], args.input_name, args.label_name)
    
    dataset.save_to_disk(args.output_dir)
