import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.configuration_sus import SusConfig
from models.modeling_sus import SusForExtractiveSummarization


SUS_INPUT_PREFIX = "sus-formatted"
EXT_LABEL_PREFIX = "extractive"


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
    max_target_length=128,
    samples=None,
):
    required_attrs = [f"{SUS_INPUT_PREFIX} {input_name}", f"{EXT_LABEL_PREFIX} {label_name}"]
    for attr in required_attrs:
        if attr not in dataset.features:
            raise ValueError(f"Column {attr} is required in dataset.")
    
    def _process_data(examples):
        model_inputs = tokenizer(
            examples[f"{SUS_INPUT_PREFIX} {input_name}"],
            truncation=True,
        )
        def _process_ext_label(label, src):
            tgt = [0] * src.count(tokenizer.cls_token)
            for l in label: tgt[l] = 1
            pad_size = max_target_length - len(tgt)
            tgt = tgt[:max_target_length] if pad_size < 0 else (tgt + pad_size*[-100])
            return tgt
        model_inputs["labels"] = list(map(
            _process_ext_label,
            examples[f"{EXT_LABEL_PREFIX} {label_name}"],
            examples[f"{SUS_INPUT_PREFIX} {input_name}"],
        ))
        return model_inputs

    if samples is not None:
        return dataset.shuffle(seed=0).select(range(samples)).map(_process_data, batched=True)
    else:
        return dataset.map(_process_data, batched=True)


def _compute_metrics(p, summary_size=3):
    predictions, labels = p
    sorted_scores = np.argsort(predictions, axis=-1)[:,::-1]
    
    preds = [
        [(1 if j in sorted_scores[i,:summary_size] else 0) 
            for j, l in enumerate(label) if l != -100]
        for i, label in enumerate(labels)
    ]
    preds = [p for pred in preds for p in pred]
    refs = [l for label in labels for l in label if l != -100]

    prf = precision_recall_fscore_support(refs, preds, average="weighted")
    acc = accuracy_score(refs, preds)

    return {
        "precision": prf[0],
        "recall": prf[1],
        "f1": prf[2],
        "accuracy": acc,
    }


def train_ext(args, tokenizer, dataset, train):
    sus = None
    if args.pretrained_model_path is not None:
        sus = SusForExtractiveSummarization.from_pretrained(args.pretrained_model_path)
    else:
        config = SusConfig(
            stacked_encoder_type=args.stacked_encoder_type,
            stacked_layers=args.stacked_layers,
            stacked_ffn_dim=args.stacked_ffn_fim,
            stacked_attention_heads=args.stacked_attention_heads,
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

        sus = SusForExtractiveSummarization(
            config,
            pretrained_pegasus_path=args.pretrained_pegasus_path,
            distill_pegasus=args.distill_pegasus,
            copied_encoder_layers=copied_encoder_layers,
            copied_decoder_layers=copied_decoder_layers,
        )
        
        if args.add_special_tokens:
            sus.resize_token_embeddings(len(tokenizer))
    
    train_set, eval_set = (
        _prepare_data(
            dataset[s],
            tokenizer,
            args.data_input_name,
            args.data_label_name,
            args.max_sentence_length,
            args.train_samples if s == "train" else args.eval_samples)
        for s in ["train", "validation"]
    )

    compute_metrics = None
    if args.compute_metrics:
        compute_metrics = lambda p: _compute_metrics(p, args.ext_summary_size)
    
    train(
        args=args,
        model=sus,
        data_collator=None,
        train_set=train_set,
        eval_set=eval_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
