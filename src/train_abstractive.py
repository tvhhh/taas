import numpy as np
import nltk

from datasets import load_metric

from transformers.data.data_collator import DataCollatorForSeq2Seq

from .models.configuration_sus import SusConfig
from .models.modeling_sus import SusForAbstractiveSummarization


SUS_INPUT_PREFIX = "sus-formatted"


def _prepare_layers_for_distill(teacher_layers, student_layers):
    if student_layers > teacher_layers:
        raise ValueError("Student model must be smaller than teacher model.")
    step = int(round(teacher_layers / student_layers))
    layers = list(range(0, step * student_layers, step))
    layers[-1] = teacher_layers - 1
    return tuple(layers)


def _prepare_data(
    dataset,
    tokenizer,
    input_name,
    label_name,
    max_input_length=1024,
    max_target_length=128,
):
    required_attrs = [f"{SUS_INPUT_PREFIX} {input_name}"]
    for attr in required_attrs:
        if attr not in dataset.features:
            raise ValueError(f"Column {attr} is required in dataset.")
    
    def _process_data(examples):
        model_inputs = tokenizer(
            examples[f"{SUS_INPUT_PREFIX} {input_name}"],
            max_length=max_input_length,
            truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[label_name],
                max_length=max_target_length,
                truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(_process_data, batched=True)


rouge = None
def _compute_metrics(p, tokenizer):
    if rouge is None:
        rouge = load_metric("rouge")
    
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


def train_abs(args, tokenizer, dataset, train):
    sus = None
    if args.pretrained_model_path is not None:
        sus = SusForAbstractiveSummarization.from_pretrained(args.pretrained_model_path)
    else:
        config = SusConfig(
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
        )

        copied_encoder_layers = None
        copied_decoder_layers = None
        if args.distill_pegasus:
            copied_encoder_layers = _prepare_layers_for_distill(
                16, args.copied_encoder_layers
            )
            copied_decoder_layers = _prepare_layers_for_distill(
                16, args.copied_decoder_layers
            )
        
        sus = SusForAbstractiveSummarization(
            config,
            pretrained_pegasus_path=args.pretrained_pegasus_path,
            distill_pegasus=args.distill_pegasus,
            copied_encoder_layers=copied_encoder_layers,
            copied_decoder_layers=copied_decoder_layers,
        )
        
        if args.add_special_tokens:
            sus.resize_token_embeddings(len(tokenizer))
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=sus)
    
    train_set, eval_set = (
        _prepare_data(
            dataset[s],
            tokenizer,
            args.data_input_name,
            args.data_label_name,
            args.max_sentence_length)
        for s in ["train", "validation"]
    )

    compute_metrics = None
    if args.compute_metrics:
        compute_metrics = lambda p: _compute_metrics(p, tokenizer)

    train(
        args=args,
        model=sus,
        data_collator=data_collator,
        train_set=train_set,
        eval_set=eval_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
