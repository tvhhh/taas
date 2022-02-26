import argparse
import re

from datasets import load_dataset, set_caching_enabled
from rouge_score import rouge_scorer

import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

import numpy as np


SUS_INPUT_PREFIX = "sus-formatted"
EXT_LABEL_PREFIX = "extractive"


# stop_words = set(stopwords.words("english"))
# def _remove_stop_words(src):
#     word_tokens = word_tokenize(src)
#     processed_tokens = [w.lower() for w in word_tokens if w.lower() not in stop_words]
#     return processed_tokens


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
    
    scorer = rouge_scorer.RougeScorer([metrics], use_stemmer=False)
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
    # preprocessed_words = map(
    #     lambda src: _remove_stop_words(src),
    #     dataset[input_name]
    # )
    # dataset = dataset.add_column(f"corpus", list(preprocessed_words))

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
    parser.add_argument("-output_dir", type=str, default=".")

    args = parser.parse_args()

    set_caching_enabled(False)
    dataset = load_dataset(args.dataset_path, args.dataset_name)

    for s in ["train", "validation", "test"]:
        dataset[s] = preprocess(dataset[s], args.input_name, args.label_name)
    
    dataset.save_to_disk(args.output_dir)
