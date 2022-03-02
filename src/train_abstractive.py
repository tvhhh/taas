import numpy as np
import nltk

from datasets import load_dataset, load_from_disk, load_metric

from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.models.pegasus.tokenization_pegasus import PegasusTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from models.configuration_sus import SusConfig
from models.modeling_sus import SusForConditionalGeneration


def _prepare_data(
    dataset,
    tokenizer,
    input_name,
    label_name,
    max_input_length=1024,
    max_target_length=128,
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
        
        return model_inputs

    return dataset.map(_process_data, batched=True)


rouge = load_metric("rouge")
def _compute_metrics(p, tokenizer):
    (logits, _), labels = p
    predictions = np.argmax(logits, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

    rouge_results = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    results = {key: value.mid.fmeasure * 100 for key, value in rouge_results.items()}

    return results


def train_abs(args):

    # Load preprocessed dataset
    dataset = None
    if args.load_data_from_disk:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, args.dataset_name)

    # Load pretrained tokenizer for PEGASUS
    tokenizer = PegasusTokenizer.from_pretrained(args.pretrained_tokenizer_path)

    # Load or initialize model
    sus = None
    if args.pretrained_model_path is not None:
        sus = SusForConditionalGeneration.from_pretrained(args.pretrained_model_path)
    else:
        config = SusConfig(
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
        )
        
        sus = SusForConditionalGeneration(
            config,
            pretrained_pegasus_path=args.pretrained_pegasus_large_path,
            shrink_pegasus_large=args.shrink_pegasus_large,
        )
    
    # Use data collator for padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=sus)
    
    # Prepare data for abstractive summarization training
    train_set, eval_set = (
        _prepare_data(
            dataset[s],
            tokenizer,
            args.data_input_name,
            args.data_label_name,
            args.max_input_length,
            args.max_target_length)
        for s in ("train", "validation")
    )

    # Use compute_metrics for validation during training
    compute_metrics = None
    if args.compute_metrics:
        compute_metrics = lambda p: _compute_metrics(p, tokenizer)

    # Use HuggingFace Trainer API
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=(not args.from_checkpoint),
        evaluation_strategy=args.eval_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.beta1,
        adam_beta2=args.beta2,
        adam_epsilon=args.epsilon,
        num_train_epochs=args.train_epochs,
        max_steps=args.train_steps,
        warmup_ratio=args.warmup_ratio,
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
    trainer = Trainer(
        model=sus,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=args.from_checkpoint)
