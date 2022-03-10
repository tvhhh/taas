import json
import nltk
import numpy as np
import os
import torch

from datasets import load_dataset, load_from_disk, load_metric
from gensim.corpora import Dictionary
from models.modeling_sus import SusForConditionalGeneration
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.models.pegasus.tokenization_pegasus import PegasusTokenizer


OUTPUTS_FILE = "summaries.txt"
RESULTS_FILE = "results.json"


def _prepare_data(
    dataset,
    tokenizer,
    input_name,
    label_name,
    max_input_length=1024,
    max_target_length=128,
    dictionary=None,
):
    def _process_data(examples):
        model_inputs = tokenizer(
            examples[input_name],
            max_length=max_input_length,
            truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[label_name],
                max_length=max_target_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]

        if dictionary is not None:
            gensim_bows = [
                dictionary.doc2bow(doc.split())
                for doc in examples["corpus"]
            ]
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
    output_dir,
    test_set,
    input_name,
    label_name,
    predictions,
):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, OUTPUTS_FILE), "a+") as writer:
        for source, target, prediction in zip(test_set[input_name], test_set[label_name], predictions):
            fmt_output = "[SOURCE]\n" + source + "\n" + "-"*100 + \
                    "\n[REFERENCE]\n" + target + "\n" + "-"*100 + \
                    "\n[OUTPUT]\n" + prediction + "\n" + "="*100
            writer.write(fmt_output + "\n\n")


def _compute_metrics(predictions, labels, rouge):
    predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in predictions]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    results = rouge.compute(predictions=predictions, references=labels)

    return results


def _save_result_to_file(output_dir, result):
    os.makedirs(output_dir, exist_ok=True)
    fmt_result = {
        "ROUGE-1": {
            "low": {
                "precision": result["rouge1"].low.precision,
                "recall": result["rouge1"].low.recall,
                "f1": result["rouge1"].low.fmeasure,
            },
            "mid": {
                "precision": result["rouge1"].mid.precision,
                "recall": result["rouge1"].mid.recall,
                "f1": result["rouge1"].mid.fmeasure,
            },
            "high": {
                "precision": result["rouge1"].high.precision,
                "recall": result["rouge1"].high.recall,
                "f1": result["rouge1"].high.fmeasure,
            },
        },
        "ROUGE-2": {
            "low": {
                "precision": result["rouge2"].low.precision,
                "recall": result["rouge2"].low.recall,
                "f1": result["rouge2"].low.fmeasure,
            },
            "mid": {
                "precision": result["rouge2"].mid.precision,
                "recall": result["rouge2"].mid.recall,
                "f1": result["rouge2"].mid.fmeasure,
            },
            "high": {
                "precision": result["rouge2"].high.precision,
                "recall": result["rouge2"].high.recall,
                "f1": result["rouge2"].high.fmeasure,
            },
        },
        "ROUGE-L": {
            "low": {
                "precision": result["rougeLsum"].low.precision,
                "recall": result["rougeLsum"].low.recall,
                "f1": result["rougeLsum"].low.fmeasure,
            },
            "mid": {
                "precision": result["rougeLsum"].mid.precision,
                "recall": result["rougeLsum"].mid.recall,
                "f1": result["rougeLsum"].mid.fmeasure,
            },
            "high": {
                "precision": result["rougeLsum"].high.precision,
                "recall": result["rougeLsum"].high.recall,
                "f1": result["rougeLsum"].high.fmeasure,
            },
        },
    }
    with open(os.path.join(output_dir, RESULTS_FILE), "w") as writer:
        writer.write(json.dumps(fmt_result, indent=2) + "\n")


def test(args):
    
    nltk.download("punkt")

    # Load preprocessed dataset
    dataset = None
    if args.load_data_from_disk:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, args.dataset_name)
    
    # Load pretrained tokenizer for PEGASUS
    tokenizer = PegasusTokenizer.from_pretrained(args.pretrained_tokenizer_path)

    # Use gensim dictionary if use neural topic model
    dictionary = None
    if args.use_ntm:
        dictionary = Dictionary.load_from_text(os.path.join(args.ntm_corpus_path, "dict.txt"))
    
    sus = SusForConditionalGeneration.from_pretrained(args.pretrained_model_path)

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
        args.result_dir,
        test_set,
        args.input_name,
        args.label_name,
        predictions
    )

    rouge = load_metric("rouge")
    result = _compute_metrics(predictions, test_set[args.label_name], rouge)
    _save_result_to_file(args.result_dir, result)
