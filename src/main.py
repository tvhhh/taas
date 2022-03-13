import argparse

from test import test
from train_abstractive import train_abs
from train_ntm import train_ntm


def str2bool(v: str):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-task", type=str, choices=["ntm", "abs", "test"], default="abs")

    parser.add_argument("-load_data_from_disk", type=str2bool, nargs="?", default=False)
    parser.add_argument("-dataset_path", type=str, default="cnn_dailymail")
    parser.add_argument("-dataset_name", type=str, default=None)
    parser.add_argument("-data_input_name", type=str, default="article")
    parser.add_argument("-data_label_name", type=str, default="highlights")

    parser.add_argument("-pretrained_tokenizer_path", type=str, default=None)
    
    parser.add_argument("-pretrained_model_path", type=str, default=None)
    parser.add_argument("-freeze_embeddings", type=str2bool, default=False)
    parser.add_argument("-freeze_encoder_layers", type=int, default=None)
    
    parser.add_argument("-use_ntm", type=str2bool, nargs="?", default=True)
    parser.add_argument("-ntm_loss_weight", type=float, default=0.1)
    parser.add_argument("-pretrained_ntm_path", type=str, default=None)
    parser.add_argument("-ntm_corpus_path", type=str, default=None)
    parser.add_argument("-ntm_dict_filter_no_below", type=int, default=2)
    parser.add_argument("-ntm_dict_filter_no_above", type=float, default=0.5)
    parser.add_argument("-ntm_max_vocab_size", type=int, default=100000)
    parser.add_argument("-ntm_num_topics", type=int, default=100)
    parser.add_argument("-ntm_activation", type=str, choices=["softplus", "relu"], default="softplus")
    parser.add_argument("-ntm_dropout", type=float, default=0.0)
    parser.add_argument("-ntm_use_tfidf", type=str2bool, nargs="?", default=False)

    parser.add_argument("-max_input_length", type=int, default=1024)
    parser.add_argument("-max_target_length", type=int, default=128)
    
    parser.add_argument("-eval_batch_size", type=int, default=4)
    parser.add_argument("-eval_strategy", type=str, choices=["no", "epoch", "steps"], default="epoch")
    parser.add_argument("-eval_steps", type=int, default=5)
    parser.add_argument("-eval_accumulation_steps", type=int, default=None)
    parser.add_argument("-logging_strategy", type=str, choices=["no", "epoch", "steps"], default="epoch")
    parser.add_argument("-logging_steps", type=int, default=5)
    parser.add_argument("-save_checkpoint_strategy", type=str, choices=["no", "epoch", "steps"], default="epoch")
    parser.add_argument("-save_checkpoint_steps", type=int, default=5)
    parser.add_argument("-train_batch_size", type=int, default=1)
    parser.add_argument("-train_epochs", type=int, default=5)
    parser.add_argument("-train_steps", type=int, default=-1)
    parser.add_argument("-gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("-lr", type=float, default=1e-5)
    parser.add_argument("-beta1", type=float, default=0.9)
    parser.add_argument("-beta2", type=float, default=0.999)
    parser.add_argument("-epsilon", type=float, default=1e-8)
    parser.add_argument("-weight_decay", type=float, default=0)
    parser.add_argument("-warmup_ratio", type=float, default=0)
    parser.add_argument("-warmup_steps", type=int, default=0)
    parser.add_argument("-load_best_at_end", type=str2bool, nargs="?", default=True)
    parser.add_argument("-metric_load_best", type=str, default="loss")
    parser.add_argument("-greater_better", type=str2bool, nargs="?", default=False)
    parser.add_argument("-resume_from_checkpoint", type=str2bool, nargs="?", default=False)
    parser.add_argument("-compute_metrics", type=str2bool, nargs="?", default=True)
    parser.add_argument("-evaluate_first_step", type=str2bool, nargs="?", default=False)

    parser.add_argument("-test_batch_size", type=int, default=10)
    parser.add_argument("-beam_width", type=int, default=8)
    parser.add_argument("-output_precision_recall", type=str2bool, nargs="?", default=False)

    parser.add_argument("-output_dir", type=str, default=None)
    parser.add_argument("-logging_dir", type=str, default=None)
    parser.add_argument("-result_dir", type=str, default=None)

    args = parser.parse_args()

    if args.task == "abs":
        train_abs(args)
    
    elif args.task == "ntm":
        train_ntm(args)

    elif args.task == "test":
        test(args)

    else:
        raise ValueError(f"Unknown task {args.task}")
