import json
import os
import re
import torch

from argparse import Namespace
from data.doc_dataset import DocDataset
from gensim.models import CoherenceModel
from models.modeling_topic import NeuralTopicModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class TrainingState:
    def __init__(self, **kwargs):
        self.train_progress = kwargs.pop("train_progress", 0)
        self.total_train_steps = kwargs.pop("total_train_steps", 0)
        self.train_loss = kwargs.pop("train_loss", 0)
        self.eval_steps = kwargs.pop("eval_steps", 0)
        self.eval_results = kwargs.pop("eval_results", [])
        self.save_steps = kwargs.pop("save_steps", 0)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class NTMTrainer:
    CHECKPOINT_PREFIX = "checkpoint"
    LOGGING_FILE = "logs.txt"
    OPTIMIZER_STATE_FILE = "optimizer_state.pt"
    TRAINING_STATE_FILE = "training_state.json"
    
    def __init__(
        self,
        model: NeuralTopicModel,
        args: Namespace,
        train_set: DocDataset,
        eval_set: DocDataset,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.train_set = train_set
        self.eval_set = eval_set

        self.train_loader = DataLoader(train_set, batch_size=args.train_batch_size)
        self.eval_loader = DataLoader(eval_set, batch_size=args.eval_batch_size)
        
        if args.resume_from_checkpoint:
            last_checkpoint = self.get_last_checkpoint()
            self.model.load_weights_from_pretrained(last_checkpoint)

            training_state_file = os.path.join(args.output_dir, self.TRAINING_STATE_FILE)
            with open(training_state_file, "r") as reader:
                training_state = json.load(reader)
            self.state = TrainingState(**training_state)

            optimizer_state_file = os.path.join(args.output_dir, self.OPTIMIZER_STATE_FILE)
            optimizer_state_dict = torch.load(optimizer_state_file)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(optimizer_state_dict)
        
        else:
            self.state = TrainingState()
            if args.train_steps == -1:
                self.state.total_train_steps = args.train_epochs * len(self.train_loader)
            else:
                self.state.total_train_steps = args.train_steps

            if args.eval_strategy == "epoch":
                self.state.eval_steps = len(self.train_loader)
            elif args.eval_strategy == "steps":
                self.state.eval_steps = args.eval_steps

            if args.save_checkpoint_strategy == "epoch":
                self.state.save_steps = len(self.train_loader)
            elif args.save_checkpoint_strategy == "steps":
                self.state.save_steps = args.save_checkpoint_steps
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.beta1,args.beta2),
                eps=args.epsilon,
                weight_decay=args.weight_decay,
            )

        self.compute_metrics = args.compute_metrics

        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
        self.output_dir = args.output_dir
        self.logging_dir = args.logging_dir

        self.train_progress, self.eval_progress = None, None
    
    def train(self):
        self.train_progress = tqdm(range(self.state.total_train_steps))
        
        current_step = 0
        skipped_train_steps = self.state.train_progress
        
        while current_step < self.state.total_train_steps:            
            for batch in self.train_loader:
                
                self.model.train()

                # Skip some training steps if resuming from checkpoint
                if current_step < skipped_train_steps:
                    current_step += 1
                    self.train_progress.update(1)
                    continue

                # Forward the model and compute loss backward
                bow = batch.to(self.device)
                posterior_params, word_dist, _ = self.model(bow)
                loss = self.model.loss(*posterior_params, word_dist, bow)
                self.state.train_loss += loss.item()
                loss.backward()

                # Update the model parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update training progress
                current_step += 1
                self.state.train_progress = current_step
                self.train_progress.update(1)

                # Evaluate metrics and perform logging
                if self.state.eval_steps > 0 and current_step % self.state.eval_steps == 0:
                    self.evaluate()

                # Save training checkpoint
                if self.state.save_steps > 0 and current_step % self.state.save_steps == 0:
                    checkpoint_dir = os.path.join(self.output_dir, f"{self.CHECKPOINT_PREFIX}-{current_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.model.save_pretrained(checkpoint_dir)
                    self.save_training_state(checkpoint_dir)
                    self.save_optimizer_state(checkpoint_dir)

                # Break the training loop when reaching num_train_steps
                if current_step == self.state.total_train_steps: break

    def evaluate(self):
        metrics, topic_words = self.evaluation_loop()
        metrics["train_loss"] = self.state.train_loss / self.state.train_progress

        self.state.eval_results.append(metrics)

        fmt_output = f"Step {self.state.train_progress}:\n"
        if topic_words is not None:
            for topic_i_words in topic_words:
                fmt_output += str(topic_i_words) + "\n"
            fmt_output += "-"*100 + "\n"
        fmt_output += str(metrics) + "\n" + "="*100

        with open(os.path.join(self.logging_dir, self.LOGGING_FILE), "a+") as writer:
            writer.write(fmt_output + "\n")

        print(f"Step {self.state.train_progress}:\n" + str(metrics) + "\n")
    
    def evaluation_loop(self):
        self.model.eval()

        if self.eval_progress is None:
            self.eval_progress = tqdm(range(len(self.eval_loader)))
        else:
            self.eval_progress.refresh()
            self.eval_progress.reset()
        
        eval_loss = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                bow = batch.to(self.device)
                posterior_params, word_dist, _ = self.model(bow)
                loss = self.model.loss(*posterior_params, word_dist, bow)
                eval_loss += loss.item()
                self.eval_progress.update(1)
        eval_loss /= len(self.eval_loader)

        topic_words = self.model.get_top_topic_words()
        topic_words = [
            [self.train_set.dictionary.id2token[word_id] for word_id in topic_i_words]
            for topic_i_words in topic_words
        ]

        if self.compute_metrics:
            ntm_metrics = self.compute_ntm_metrics(topic_words)
            return {**ntm_metrics, "eval_loss": eval_loss}, topic_words
        else:
            return {"eval_loss": eval_loss}, topic_words

    def compute_ntm_metrics(self, topic_words):
        c_v_coherence_model, c_uci_coherence_model, u_mass_coherence_model = (
            CoherenceModel(
                topics=topic_words,
                texts=self.eval_set.documents,
                corpus=self.eval_set.bows,
                dictionary=self.train_set.dictionary,
                coherence=coherence_metric,
            ) for coherence_metric in ("c_v", "c_uci", "u_mass")
        )
        c_v = c_v_coherence_model.get_coherence()
        c_uci = c_uci_coherence_model.get_coherence()
        u_mass = u_mass_coherence_model.get_coherence()
        return {"c_v": c_v, "c_uci": c_uci, "u_mass": u_mass}

    def save_training_state(self, save_directory):
        training_state_file = os.path.join(save_directory, self.TRAINING_STATE_FILE)
        def _to_json_string(state):
            return json.dumps(state, indent=2, sort_keys=True) + "\n"
        with open(training_state_file, "w") as writer:
            writer.write(_to_json_string(self.state.to_dict()))

    def save_optimizer_state(self, save_directory):
        optimizer_state_file = os.path.join(save_directory, self.OPTIMIZER_STATE_FILE)
        torch.save(self.optimizer.state_dict(), optimizer_state_file)
    
    def get_last_checkpoint(self):
        re_checkpoint = re.compile(r"^" + self.CHECKPOINT_PREFIX + r"\-(\d+)$")
        contents = [
            path for path in os.listdir(self.output_dir)
            if re_checkpoint.search(path) is not None
            and os.path.isdir(os.path.join(self.output_dir, path))
        ]
        return os.path.join(self.output_dir, max(contents, key=lambda x: int(re_checkpoint.search(x).groups()[0])))
