import json
import nltk
import numpy as np
import os
import re
import torch

from argparse import Namespace
from datasets import load_dataset, load_from_disk
from datasets.arrow_dataset import Dataset
from gensim.corpora import Dictionary
from models.modeling_sus import SusForConditionalGeneration
from rouge_score import rouge_scorer, scoring
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.models.pegasus.tokenization_pegasus import PegasusTokenizer
from typing import Any, Dict, List, Optional


CHECKPOINT_PREFIX = "checkpoint"
METRIC_PREFIX = "eval"

OUTPUTS_FILE = "outputs.txt"
RESULTS_FILE = "results.json"
TRAINER_STATE_FILE = "trainer_state.json"


def _get_last_checkpoint(output_dir):
    re_checkpoint = re.compile(r"^" + CHECKPOINT_PREFIX + r"\-(\d+)$")
    contents = [
        path for path in os.listdir(output_dir)
        if re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(output_dir, path))
    ]
    return os.path.join(output_dir, max(contents, key=lambda x: int(re_checkpoint.search(x).groups()[0])))


def _get_best_checkpoints(output_dir, num_checkpoints=1, metric="loss", greater_is_better=False):
    last_checkpoint = _get_last_checkpoint(output_dir)
    
    with open(os.path.join(last_checkpoint, TRAINER_STATE_FILE), "r") as reader:
        trainer_state = json.load(reader)
    
    metric_history = trainer_state["log_history"][1::2]
    metrics = np.array([x[f"{METRIC_PREFIX}_{metric}"] for x in metric_history])

    sorted_ids = np.argsort(metrics)
    if greater_is_better:
        sorted_ids = sorted_ids[::-1]
    
    best_ids = sorted_ids[:num_checkpoints]

    return tuple(
        os.path.join(
            output_dir, 
            f"{CHECKPOINT_PREFIX}-{metric_history[i]['step']}"
        ) for i in best_ids
    )


def _prepare_data(
    dataset: Dataset,
    tokenizer: PegasusTokenizer,
    input_name: str,
    label_name: str,
    max_input_length: int = 1024,
    max_target_length: int = 128,
    dictionary: Optional[Dictionary] = None,
):
    def _process_data(examples):
        model_inputs = tokenizer(
            examples[input_name],
            max_length=max_input_length,
            padding=True,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[label_name],
                max_length=max_target_length,
                padding=True,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]

        if dictionary is not None:
            gensim_bows = [dictionary.doc2bow(doc.split()) for doc in examples["corpus"]]
            bows = np.zeros((len(examples["corpus"]), len(dictionary.token2id)))
            for i, bow in enumerate(gensim_bows):
                if len(bow) > 0:
                    ids, freqs = zip(*bow)
                    bows[i][list(ids)] = list(freqs)
            model_inputs["bag_of_words"] = bows.tolist()
        
        return model_inputs

    return dataset.map(_process_data, batched=True)


def _generate(
    dataloader: DataLoader,
    model: SusForConditionalGeneration,
    beam_width: int,
    max_target_length: int,
    tokenizer: PegasusTokenizer,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    progress_bar = tqdm(range(len(dataloader)))

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}

            output_ids = model.generate(
                **inputs,
                num_beams=beam_width,
                max_length=max_target_length,
                early_stopping=True,
            )
            outputs = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
            predictions += outputs
            progress_bar.update(1)

    return predictions


def _save_output_to_file(
    output_dir: str,
    test_set: Dataset,
    input_name: str,
    label_name: str,
    predictions: List[str],
):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, OUTPUTS_FILE), "a+") as writer:
        for source, target, prediction in zip(test_set[input_name], test_set[label_name], predictions):
            fmt_output = "[SOURCE]\n" + source + "\n" + "-"*100 + \
                    "\n[REFERENCE]\n" + target + "\n" + "-"*100 + \
                    "\n[OUTPUT]\n" + "\n".join(nltk.sent_tokenize(prediction)) + "\n" + "="*100
            writer.write(fmt_output + "\n\n")


def _extract_rouge_mid_statistics(dct: Dict[str, Any]):
    returned_dct = {}
    for k, v in dct.items():
        mid = v.mid
        returned_dct[k] = {
            stat: round(getattr(mid, stat) * 100, 2)
            for stat in ["precision", "recall", "fmeasure"]
        }
    return returned_dct


def _cal_rouge_scores(
    predictions: List[str],
    labels: List[str],
    metrics: List[str] = ["rouge1", "rouge2", "rougeLsum"],
    return_precision_and_recall: bool = False,
):
    scorer = rouge_scorer.RougeScorer(metrics)
    aggregator = scoring.BootstrapAggregator()

    for pred, target in zip(predictions, labels):
        scores = scorer.score(
            "\n".join(nltk.sent_tokenize(target)),
            "\n".join(nltk.sent_tokenize(pred)),
        )
        aggregator.add_scores(scores)
    
    result = aggregator.aggregate()

    if return_precision_and_recall:
        return _extract_rouge_mid_statistics(result)
    else:
        return {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}


def _save_result_to_file(output_dir: str, result: Dict[str, float]):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, RESULTS_FILE), "w") as writer:
        writer.write(json.dumps(result, indent=2) + "\n")


def _test(
    args: Namespace,
    checkpoint_dir: str,
    dataset: Dataset,
    dictionary: Optional[Dictionary] = None,
):
    re_checkpoint = re.compile(r"^" + CHECKPOINT_PREFIX + r"\-(\d+)$")
    checkpoint_name = re_checkpoint.search(checkpoint_dir).groups()[0]

    tokenizer = PegasusTokenizer.from_pretrained(checkpoint_dir)
    sus = SusForConditionalGeneration.from_pretrained(checkpoint_dir)

    test_set = _prepare_data(
        dataset["test"],
        tokenizer,
        args.data_input_name,
        args.data_label_name,
        args.max_input_length,
        args.max_target_length,
        dictionary,
    )
    test_set.set_format(
        type="torch",
        columns=(
            ["attention_mask","input_ids"] + 
            (["bag_of_words"] if dictionary else [])
        )
    )
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)

    predictions = _generate(
        test_loader,
        sus,
        args.beam_width,
        args.max_target_length,
        tokenizer,
    )
    _save_output_to_file(
        f"{args.result_dir}-{checkpoint_name}",
        test_set,
        args.data_input_name,
        args.data_label_name,
        predictions,
    )

    rouge_scores = _cal_rouge_scores(
        predictions,
        test_set[args.data_label_name],
        return_precision_and_recall=args.output_precision_recall
    )
    _save_result_to_file(f"{args.result_dir}-{checkpoint_name}", rouge_scores)

    return rouge_scores


def _nested_collect(obj, new_obj):
    if isinstance(obj, dict) and isinstance(new_obj, dict):
        return {
            k: _nested_collect((obj[k] if k in obj else None), v)
            for k, v in new_obj.items()
        }
    elif obj is None:
        return {
            k: _nested_collect(obj, v)
            for k, v in new_obj.items()
        } if isinstance(new_obj, dict) else [new_obj]
    else:
        return obj + [new_obj]


def _nested_compute(obj):
    if isinstance(obj, dict):
        return {k: _nested_compute(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.mean(obj)
    else:
        raise ValueError(f"Unsupported type {type(obj)}")


def test(args: Namespace):
    
    nltk.download("punkt")

    # Load preprocessed dataset
    dataset = None
    if args.load_data_from_disk:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, args.dataset_name)

    # Use gensim dictionary if use neural topic model
    dictionary = None
    if args.use_ntm:
        dictionary = Dictionary.load_from_text(os.path.join(args.ntm_corpus_path, "dict.txt"))

    
    best_checkpoints = None
    if not args.test_best_checkpoints:
        best_checkpoints = (args.pretrained_model_path,)
    else:
        best_checkpoints = _get_best_checkpoints(args.output_dir, args.test_num_checkpoints)
    
    result_collector = {}
    for checkpoint in best_checkpoints:
        result = _test(args, checkpoint, dataset, dictionary)
        result_collector = _nested_collect(result_collector, result)
    
    overall_result = _nested_compute(result_collector)
    _save_result_to_file(args.result_dir, overall_result)
