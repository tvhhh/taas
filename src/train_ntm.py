from data.doc_dataset import DocDataset
from datasets import load_dataset, load_from_disk
from models.modeling_topic import NeuralTopicModel
from trainer.ntm_trainer import NTMTrainer


def train_ntm(args):
    
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

    # Load or initialize neural topic model
    ntm = None
    if args.pretrained_ntm_path:
        ntm = NeuralTopicModel.from_pretrained(args.pretrained_ntm_path)
    else:
        ntm = NeuralTopicModel(
            vocab_size=train_set.vocab_size,
            n_topics=args.ntm_num_topics,
            activation=args.ntm_activation,
            dropout=args.ntm_dropout,
        )

    trainer = NTMTrainer(
        model=ntm,
        args=args,
        train_set=train_set,
        eval_set=eval_set,
    )
    trainer.train()
