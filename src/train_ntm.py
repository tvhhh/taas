from argparse import Namespace
from data.doc_dataset import DocDataset
from datasets import load_dataset, load_from_disk
from models.modeling_topic.batm import BATM
from models.modeling_topic.gsm import GSM
from trainer.ntm_trainer import BATMTrainer, NTMTrainer


def train_ntm(args: Namespace):
    
    # Load preprocessed dataset
    dataset = None
    if args.load_data_from_disk:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, args.dataset_name)
    
    # Prepare data for neural topic model training
    train_set, eval_set = (
        DocDataset(
            [doc.split() for doc in dataset[s]["corpus"]],
            args.ntm_corpus_path,
            args.ntm_dict_filter_no_below,
            args.ntm_dict_filter_no_above,
            args.ntm_max_vocab_size,
            args.ntm_use_tfidf,
        ) for s in ("train", "validation")
    )

    ntm_class = None
    if args.ntm == "gsm":
        ntm_class = GSM
    elif args.ntm == "batm":
        ntm_class = BATM
    else:
        raise ValueError(f"Unknown NTM {args.ntm}")

    # Load or initialize neural topic model
    ntm = None
    if args.pretrained_ntm_path:
        ntm = ntm_class.from_pretrained(args.pretrained_ntm_path)
    else:
        ntm = ntm_class(
            vocab_size=train_set.vocab_size,
            n_topics=args.ntm_num_topics,
            dropout=args.ntm_dropout,
        )

    if args.ntm == "batm":
        trainer = BATMTrainer(
            model=ntm,
            args=args,
            train_set=train_set,
            eval_set=eval_set,
        )
    else:
        trainer = NTMTrainer(
            model=ntm,
            args=args,
            train_set=train_set,
            eval_set=eval_set,
        )
    
    trainer.train()
