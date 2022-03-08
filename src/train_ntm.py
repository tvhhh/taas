import json
import os
import re
import torch

from data.doc_dataset import DocDataset
from datasets import load_dataset, load_from_disk
from gensim.models.coherencemodel import CoherenceModel
from models.configuration_sus import SusConfig
from models.modeling_sus import SusNeuralTopicModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _compute_metrics(topic_words, eval_set):
    c_v_coherence_model, c_uci_coherence_model, u_mass_coherence_model = (
        CoherenceModel(
            topics=topic_words,
            texts=eval_set.documents,
            dictionary=eval_set.dictionary,
            coherence=coherence_metric,
        ) for coherence_metric in ("c_v", "c_uci", "u_mass")
    )
    c_v_score = c_v_coherence_model.get_coherence()
    c_uci_score = c_uci_coherence_model.get_coherence()
    u_mass_score = u_mass_coherence_model.get_coherence()
    return {
        "c_v": c_v_score,
        "c_uci": c_uci_score,
        "u_mass": u_mass_score,
    }


eval_progress = None
def _evaluation_loop(args, model, eval_set):
    eval_loader = DataLoader(eval_set, batch_size=args.eval_batch_size)
    
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
    train_loss, current_step = training_state["train_loss"], training_state["train_progress"]

    metrics = _evaluation_loop(args, model, eval_set)
    metrics["train_loss"] = train_loss / current_step
    
    fmt_output = f"Step {current_step:<10d}:"
    for metric, result in metrics.items():
        fmt_output += f"\t{metric}: {result:.6f}"
    
    if args.logging_dir is not None:
        os.makedirs(args.logging_dir, exist_ok=True)
        with open(os.path.join(args.logging_dir, "logs.txt"), "a+") as writer:
            writer.write(fmt_output + "\n")
    
    print(fmt_output)


def _train(args, model, train_set, eval_set):
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.train_batch_size)

    training_state = {
        "train_progress": 0,
        "total_train_steps": 0,
        "eval_steps": 0,
        "save_steps": 0,
        "train_loss": 0,
    }

    skipped_train_steps = 0
    if args.from_checkpoint:
        # Get last checkpoint
        re_checkpoint = re.compile(r"^checkpoint\-(\d+)$")
        dirs = os.listdir(args.output_dir)
        checkpoints = [
            path for path in dirs
            if re_checkpoint.search(path) is not None
            and os.path.isdir(os.path.join(args.output_dir, path))
        ]
        last_checkpoint = os.path.join(
            args.output_dir,
            max(checkpoints, key=lambda x: int(re_checkpoint.search(x).groups()[0]))
        )

        # Load saved model
        model = SusNeuralTopicModel.from_pretrained(last_checkpoint)

        # Load optimizer state dict
        optimizer_state_file = os.path.join(last_checkpoint, "optimizer_state.pt")
        optimizer_state_dict = torch.load(optimizer_state_file)
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(optimizer_state_dict)

        # Load training state
        training_state_file = os.path.join(last_checkpoint, "training_state.json")
        with open(training_state_file, "r") as reader:
            training_state = json.load(reader)
            skipped_train_steps = training_state["train_progress"]

    else:
        if args.train_steps == -1:
            training_state["total_train_steps"] = args.train_epochs * len(train_loader)
        else:
            training_state["total_train_steps"] = args.train_steps

        if args.eval_strategy == "epoch":
            training_state["eval_steps"] = len(train_loader)
        elif args.eval_strategy == "steps":
            training_state["eval_steps"] = args.eval_steps

        if args.save_checkpoint_strategy == "epoch":
            training_state["save_steps"] = len(train_loader)
        elif args.save_checkpoint_strategy == "steps":
            training_state["save_steps"] = args.save_checkpoint_steps
        
        # Use Adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1,args.beta2),
            eps=args.epsilon,
            weight_decay=args.weight_decay,
        )

    # Load model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Use tqdm progress bars
    training_progress = tqdm(range(training_state["total_train_steps"]))
    
    current_step = 0
    model.train()
    while current_step < training_state["total_train_steps"]:
        for batch in train_loader:
            # Skip training steps if resuming from checkpoint
            if current_step < skipped_train_steps:
                current_step += 1
                continue

            # Forward the model and compute loss backward
            x_bow = batch.to(device)
            _, _, loss = model(x_bow, output_loss=True)
            training_state["train_loss"] += loss.item()
            loss.backward()

            # Update the model parameters via optimizer
            optimizer.step()
            optimizer.zero_grad()

            # Update the progress
            training_progress.update(1)
            current_step += 1
            training_state["train_progress"] = current_step
            
            # Compute evaluation metrics
            if training_state["eval_steps"] > 0 and current_step % training_state["eval_steps"] == 0:
                _evaluate(args, model, eval_set, training_state)
            
            # Save checkpoint
            if training_state["save_steps"] > 0 and current_step % training_state["save_steps"] == 0:
                if args.output_dir is not None:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{current_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    model.save_pretrained(checkpoint_dir)

                    optimizer_state_file = os.path.join(checkpoint_dir, "optimizer_state.pt")
                    torch.save(optimizer.state_dict(), optimizer_state_file)
                    
                    training_state_file = os.path.join(checkpoint_dir, "training_state.json")
                    def _to_json_string(state):
                        return json.dumps(state, indent=2, sort_keys=True) + "\n"
                    with open(training_state_file, "w", encoding="utf-8") as writer:
                        writer.write(_to_json_string(training_state))
            
            # Break the training loop when reaching num_train_steps
            if current_step == training_state["total_train_steps"]: break


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
        ntm = SusNeuralTopicModel.from_pretrained(args.pretrained_ntm_path)
    else:
        config = SusConfig(
            bow_size=train_set.vocab_size,
            ntm_dropout=args.ntm_dropout,
            topic_dim=args.ntm_topic_dim,
        )
        ntm = SusNeuralTopicModel(config)

    _train(args, ntm, train_set, eval_set)
