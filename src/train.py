import argparse

from datasets import load_dataset, load_from_disk

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.models.pegasus.tokenization_pegasus import PegasusTokenizer

from train_abstractive import train_abs
from train_extractive import train_ext


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train(args, model, data_collator, train_set, eval_set, tokenizer, compute_metrics=None):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
        resume_from_checkpoint=args.from_checkpoint,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-task", type=str, choices=["ext", "abs", "test"], default="abs")

    parser.add_argument("-load_data_from_disk", type=str2bool, nargs="?", default=False)
    parser.add_argument("-dataset_path", type=str, default="cnn_dailymail")
    parser.add_argument("-dataset_name", type=str, default=None)
    parser.add_argument("-data_input_name", type=str, default="article")
    parser.add_argument("-data_label_name", type=str, default="highlights")

    parser.add_argument("-pretrained_model_path", type=str, default=None)
    parser.add_argument("-pretrained_pegasus_path", type=str, default="google/pegasus-large")
    parser.add_argument("-distill_pegasus", type=str2bool, default=False)
    parser.add_argument("-copied_encoder_layers", type=int, default=16)
    parser.add_argument("-copied_decoder_layers", type=int, default=4)
    parser.add_argument("-pretrained_tokenizer_path", type=str, default=None)
    parser.add_argument("-add_special_tokens", type=str2bool, nargs="?", default=True)

    parser.add_argument("-max_input_length", type=int, default=1024)
    parser.add_argument("-max_label_length", type=int, default=128)
    parser.add_argument("-max_sentence_length", type=int, default=128)
    parser.add_argument("-ext_summary_size", type=int, default=3)

    parser.add_argument("-stacked_encoder_type", type=str, choices=["classifier", "transformer"], default="transformer")
    parser.add_argument("-stacked_layers", type=int, default=3)
    parser.add_argument("-stacked_ffn_dim", type=int, default=4096)
    parser.add_argument("-stacked_attention_heads", type=int, default=16)
    
    parser.add_argument("-eval_strategy", type=str, choices=["no", "epoch", "steps"], default="epoch")
    parser.add_argument("-eval_steps", type=int, default=5)
    parser.add_argument("-eval_accumulation_steps", type=int, default=None)
    parser.add_argument("-logging_strategy", type=str, choices=["no", "epoch", "steps"], default="epoch")
    parser.add_argument("-logging_steps", type=int, default=5)
    parser.add_argument("-save_checkpoint_strategy", type=str, choices=["no", "epoch", "steps"], default="epoch")
    parser.add_argument("-save_checkpoint_steps", type=int, default=5)
    parser.add_argument("-batch_size", type=int, default=1024)
    parser.add_argument("-train_epochs", type=int, default=128)
    parser.add_argument("-lr", type=float, default=1e-5)
    parser.add_argument("-beta1", type=float, default=0.9)
    parser.add_argument("-beta2", type=float, default=0.999)
    parser.add_argument("-epsilon", type=float, default=1e-8)
    parser.add_argument("-weight_decay", type=float, default=0)
    parser.add_argument("-warmup_ratio", type=float, default=0)
    parser.add_argument("-load_best_at_end", type=str2bool, nargs="?", default=True)
    parser.add_argument("-metric_load_best", type=str, default="loss")
    parser.add_argument("-greater_better", type=str2bool, nargs="?", default=False)
    parser.add_argument("-from_checkpoint", type=str, default=None)
    parser.add_argument("-compute_metrics", type=str2bool, default=True)

    parser.add_argument("-output_dir", type=str, default="./checkpoints")
    parser.add_argument("-logging_dir", type=str, default="./logs")

    args = parser.parse_args()

    dataset = None
    if args.load_data_from_disk:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, args.dataset_name)

    tokenizer = PegasusTokenizer.from_pretrained(args.pretrained_tokenizer_path)
    if args.add_special_tokens:
        tokenizer.add_special_tokens({
            "cls_token": "<cls>",
            "sep_token": "<sep>",
        })
    
    if args.task == "ext":
        train_ext(args, tokenizer, dataset, train)

    elif args.task == "abs":
        train_abs(args, tokenizer, dataset, train)

    elif args.task == "test":
        pass

    else:
        raise ValueError(f"Unknown task {args.task}.")
