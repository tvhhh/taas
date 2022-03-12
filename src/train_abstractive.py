import nltk
import numpy as np
import os

from datasets import load_dataset, load_from_disk, load_metric
from datasets.arrow_dataset import Dataset
from datasets.metric import Metric
from gensim.corpora import Dictionary
from models.configuration_sus import SusConfig
from models.modeling_sus import SusForConditionalGeneration
from trainer.hf_trainer import HFTrainer
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.models.pegasus.tokenization_pegasus import PegasusTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
from typing import Optional


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
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[label_name],
                max_length=max_target_length,
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


def _compute_metrics(p: EvalPrediction, tokenizer: PegasusTokenizer, rouge: Metric):
    (predictions, _), labels = p
    
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    result = {key: value.mid.fmeasure * 100 for key, value in rouge_scores.items()}

    return result


def train_abs(args):

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

    # Load or initialize model
    sus = None
    if args.pretrained_model_path is not None:
        sus = SusForConditionalGeneration.from_pretrained(
            args.pretrained_model_path,
            use_ntm=args.use_ntm,
            corpus_size=(len(dictionary.token2id) if dictionary else 20000),
            n_topics=args.ntm_num_topics,
            ntm_activation=args.ntm_activation,
            ntm_dropout=args.ntm_dropout,
            ntm_loss_weight=args.ntm_loss_weight,
        )
    else:
        config = SusConfig(
            use_ntm=args.use_ntm,
            corpus_size=(len(dictionary.token2id) if dictionary else 20000),
            n_topics=args.ntm_num_topics,
            ntm_activation=args.ntm_activation,
            ntm_dropout=args.ntm_dropout,
            ntm_loss_weight=args.ntm_loss_weight,
        )
        sus = SusForConditionalGeneration(config)
    
    if args.pretrained_ntm_path is not None:
        sus.load_pretrained_ntm(args.pretrained_ntm_path)
    
    # Freeze encoder layers
    if args.freeze_encoder_layers is not None:
        for param in sus.get_encoder().embed_tokens.parameters():
            param.requires_grad = False
        for i in range(args.freeze_encoder_layers):
            for param in sus.get_encoder().layers[i].parameters():
                param.requires_grad = False
    
    # Use data collator for padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=sus)
    
    # Prepare data for abstractive summarization training
    train_set, eval_set = (
        _prepare_data(
            dataset[s],
            tokenizer,
            args.data_input_name,
            args.data_label_name,
            args.max_input_length,
            args.max_target_length,
            dictionary,
        ) for s in ("train", "validation")
    )

    # Use compute_metrics for validation during training
    compute_metrics = None
    if args.compute_metrics:
        rouge = load_metric("rouge")
        compute_metrics = lambda p: _compute_metrics(p, tokenizer, rouge)

    # Use HuggingFace Trainer API
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=(not args.resume_from_checkpoint),
        evaluation_strategy=args.eval_strategy,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        adam_epsilon=args.epsilon,
        num_train_epochs=args.train_epochs,
        max_steps=args.train_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        save_strategy=args.save_checkpoint_strategy,
        save_steps=args.save_checkpoint_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_at_end,
        metric_for_best_model=args.metric_load_best,
        greater_is_better=args.greater_better,
    )
    trainer = HFTrainer(
        model=sus,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.evaluate_first_step:
        trainer.evaluate()
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
