import os
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk
from data import BowDataset

from models.configuration_sus import SusConfig
from models.modeling_sus import SusNeuralTopicModel

from gensim.models.coherencemodel import CoherenceModel

from tqdm.auto import tqdm


def _compute_metrics(topic_words, eval_set):
    c_v_coherence_model, c_uci_coherence_model, c_npmi_coherence_model = (
        CoherenceModel(
            topics=topic_words,
            texts=eval_set.documents,
            dictionary=eval_set.dictionary,
            coherence=coherence_metric,
        ) for coherence_metric in ("c_v", "c_uci", "c_npmi")
    )
    c_v_score = c_v_coherence_model.get_coherence()
    c_uci_score = c_uci_coherence_model.get_coherence()
    c_npmi_score = c_npmi_coherence_model.get_coherence()
    return {
        "c_v": c_v_score,
        "c_uci": c_uci_score,
        "c_npmi": c_npmi_score,
    }


eval_progress = None
def _evaluation_loop(args, model, eval_set):
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size)
    
    global eval_progress
    if eval_progress is None:
        eval_progress = tqdm(range(len(eval_loader)))
    else:
        eval_progress.refresh()
        eval_progress.reset()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            x_bow = batch.to(device)
            _, _, loss = model(x_bow, output_loss=True)
            eval_loss += loss.item()
            eval_progress.update(1)
    eval_loss /= len(eval_loader)

    ntm_metrics = {}
    if args.compute_metrics:
        topic_words = model.get_top_topic_words()
        topic_words = [
            [eval_set.dictionary.id2token[word_id] for word_id in sample]
            for sample in topic_words
        ]
        ntm_metrics = _compute_metrics(topic_words, eval_set)
    
    return {**ntm_metrics, "eval_loss": eval_loss}


def _evaluate(args, model, eval_set, training_state):
    train_loss, current_step = training_state

    metrics = _evaluation_loop(args, model, eval_set)
    metrics["train_loss"] = train_loss / current_step
    
    fmt_output = f"Step {current_step}:"
    for metric, result in metrics.items():
        fmt_output += f"    {metric}: {result:.6f}"
    
    if args.logging_dir is not None:
        if not os.path.exists(args.logging_dir):
            os.makedirs(args.logging_dir)
        with open(os.path.join(args.logging_dir, "logs.txt"), "a+") as writer:
            writer.write(fmt_output + "\n")
    
    print(fmt_output)


def _train(args, model, train_set, eval_set):
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
    
    num_train_steps = 0
    if args.train_steps == -1:
        num_train_steps = args.train_epochs * len(train_loader)
    else:
        num_train_steps = args.train_steps

    num_eval_steps = 0
    if args.eval_strategy == "epoch":
        num_eval_steps = len(train_loader)
    elif args.eval_strategy == "steps":
        num_eval_steps = args.eval_steps

    num_save_steps = 0
    if args.save_checkpoint_strategy == "epoch":
        num_save_steps = len(train_loader)
    elif args.save_checkpoint_strategy == "steps":
        num_save_steps = args.save_checkpoint_steps

    # Load model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Use Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1,args.beta2),
        eps=args.epsilon,
        weight_decay=args.weight_decay,
    )

    # Use tqdm progress bars
    training_progress = tqdm(range(num_train_steps))
    
    current_step = 0
    train_loss = 0
    model.train()
    while current_step < num_train_steps:
        for batch in train_loader:
            # Forward the model and compute loss backward
            x_bow = batch.to(device)
            _, _, loss = model(x_bow, output_loss=True)
            train_loss += loss.item()
            loss.backward()

            # Update the model parameters via optimizer
            optimizer.step()
            optimizer.zero_grad()

            # Update the progress
            training_progress.update(1)
            current_step += 1
            
            # Compute evaluation metrics
            if num_eval_steps > 0 and current_step % num_eval_steps == 0:
                _evaluate(args, model, eval_set, (train_loss, current_step))
            
            # Save checkpoint
            if num_save_steps > 0 and current_step % num_save_steps == 0:
                if args.output_dir is not None:
                    model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{current_step}"))
            
            # Break the training loop when reaching num_train_steps
            if current_step == num_train_steps: break


def train_ntm(args):
    
    # Load preprocessed dataset
    dataset = None
    if args.load_data_from_disk:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, args.dataset_name)
    
    # Prepare data for neural topic model training
    train_set, eval_set = (
        BowDataset(
            [doc.split() for doc in dataset[s]["corpus"]],
            args.ntm_dict_path,
            args.ntm_dict_filter_no_below,
            args.ntm_dict_filter_no_above,
            args.ntm_max_vocab_size,
        ) for s in ("train", "validation")
    )

    # Load or initialize neural topic model
    ntm = None
    if args.pretrained_ntm_path:
        ntm = SusNeuralTopicModel.from_pretrained(args.pretrained_ntm_path)
    else:
        config = SusConfig(
            bow_size=train_set.vocab_size,
            ntm_dropout=args.ntm_dropout,
            topic_dim=args.ntm_topic_dim,
        )
        ntm = SusNeuralTopicModel(config)

    _train(args, ntm, train_set, eval_set)
