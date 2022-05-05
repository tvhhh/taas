import json
import os
import re
import torch

from argparse import Namespace
from data.doc_dataset import DocDataset
from gensim.models import CoherenceModel
from models.modeling_topic.batm import BATM
from models.modeling_topic.utils import NeuralTopicModel
from table_logger import TableLogger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List

SAMPLED_CONTEXT = [
    ["liverpool", "chelsea", "arsenal", "league", "cup", "football"],
    ["police", "gun", "arrest", "court", "crime", "officer", "accuse"],
    ["bank", "stock", "price", "market", "money"]
]


def _doc2bow(dataset: DocDataset, docs: List[List[str]]):
    vec = torch.zeros(len(docs), dataset.vocab_size)
    for i, doc in enumerate(docs):
        word_freq = dataset.dictionary.doc2bow(doc)
        if len(word_freq) > 0:
            ids, vals = zip(*word_freq)
            vec[i, list(ids)] = torch.tensor(list(vals)).float()
    return vec


class TrainerState:
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
    TRAINER_STATE_FILE = "trainer_state.json"
    
    def __init__(
        self,
        model: NeuralTopicModel,
        args: Namespace,
        train_set: DocDataset,
        eval_set: DocDataset,
    ):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
        self.output_dir = args.output_dir
        self.logging_dir = args.logging_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.train_set = train_set
        self.eval_set = eval_set

        self.train_loader = DataLoader(train_set, batch_size=args.train_batch_size)
        self.eval_loader = DataLoader(eval_set, batch_size=args.eval_batch_size)
        
        if args.resume_from_checkpoint:
            last_checkpoint = self.get_last_checkpoint()
            self.model.load_weights_from_pretrained(last_checkpoint)

            trainer_state_file = os.path.join(last_checkpoint, self.TRAINER_STATE_FILE)
            with open(trainer_state_file, "r") as reader:
                trainer_state = json.load(reader)
            self.state = TrainerState(**trainer_state)

            optimizer_state_file = os.path.join(last_checkpoint, self.OPTIMIZER_STATE_FILE)
            optimizer_state_dict = torch.load(optimizer_state_file)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(optimizer_state_dict)
        
        else:
            self.state = TrainerState()
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

        self.train_progress, self.eval_progress = None, None
        self.logger = None
    
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
                loss_params, _ = self.model(bow)
                loss = self.model.loss(*loss_params)
                self.state.train_loss = ((self.state.train_loss * current_step) + loss.item()) / (current_step + 1)
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
                    self.save_trainer_state(checkpoint_dir)
                    self.save_optimizer_state(checkpoint_dir)

                # Break the training loop when reaching num_train_steps
                if current_step == self.state.total_train_steps: break

    def evaluate(self, sampled_context=None):
        if sampled_context is None:
            sampled_context = SAMPLED_CONTEXT
        
        metrics, topic_words, sampled_words = self.evaluation_loop(sampled_context)
        metrics["train_loss"] = self.state.train_loss

        self.state.eval_results.append(metrics)

        fmt_output = f"Step {self.state.train_progress}:\n"
        fmt_output += str(metrics) + "\n" + "-"*100 + "\n"
        for topic_i_words in topic_words:
            fmt_output += str(topic_i_words) + "\n"
        fmt_output += "-"*100 + "\n" + "Sampled words:\n"
        for sampled_i_context, sampled_i_words in zip(sampled_context, sampled_words):
            fmt_output += str(sampled_i_context) + " -> " + str(sampled_i_words) + "\n"
        fmt_output += "="*100

        with open(os.path.join(self.logging_dir, self.LOGGING_FILE), "a+") as writer:
            writer.write(fmt_output + "\n")

        if self.logger is None:
            self.logger = TableLogger(columns=["steps"]+list(metrics.keys()))
        
        self.logger(self.state.train_progress, *tuple(metrics.values()))
    
    def evaluation_loop(self, sampled_context: List[List[str]]):
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
                loss_params, _ = self.model(bow)
                loss = self.model.loss(*loss_params)
                eval_loss += loss.item()
                self.eval_progress.update(1)
        eval_loss /= len(self.eval_loader)

        topic_words = self.model.get_topic_words()
        topic_words = [
            [self.train_set.dictionary.id2token[word_id] for word_id in topic_i_words]
            for topic_i_words in topic_words
        ]

        bow = _doc2bow(self.train_set, sampled_context)
        sampled_words = self.model.sample(bow)
        sampled_words = [
            [self.train_set.dictionary.id2token[word_id] for word_id in sampled_i_words]
            for sampled_i_words in sampled_words
        ]

        if self.compute_metrics:
            ntm_metrics = self.compute_ntm_metrics(topic_words)
            return {**ntm_metrics, "eval_loss": eval_loss}, topic_words, sampled_words
        else:
            return {"eval_loss": eval_loss}, topic_words, sampled_words

    def compute_ntm_metrics(self, topic_words: List[List[str]]):
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

    def save_trainer_state(self, save_directory: str):
        trainer_state_file = os.path.join(save_directory, self.TRAINER_STATE_FILE)
        def _to_json_string(state):
            return json.dumps(state, indent=2, sort_keys=True) + "\n"
        with open(trainer_state_file, "w") as writer:
            writer.write(_to_json_string(self.state.to_dict()))

    def save_optimizer_state(self, save_directory: str):
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


class BATMTrainerState:
    def __init__(self, **kwargs):
        self.train_progress = kwargs.pop("train_progress", 0)
        self.total_train_steps = kwargs.pop("total_train_steps", 0)
        self.train_loss_discriminator = kwargs.pop("train_loss_discriminator", 0)
        self.train_loss_encoder = kwargs.pop("train_loss_encoder", 0)
        self.train_loss_generator = kwargs.pop("train_loss_generator", 0)
        self.eval_steps = kwargs.pop("eval_steps", 0)
        self.eval_results = kwargs.pop("eval_results", [])
        self.save_steps = kwargs.pop("save_steps", 0)
        self.clip = kwargs.pop("clip", 0.01)
        self.n_critic = kwargs.pop("n_critic", 5)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class BATMTrainer:

    CHECKPOINT_PREFIX = "checkpoint"
    LOGGING_FILE = "logs.txt"
    OPTIMIZER_STATE_FILE = "optimizer_state.pt"
    TRAINER_STATE_FILE = "trainer_state.json"

    def __init__(
        self,
        model: BATM,
        args: Namespace,
        train_set: DocDataset,
        eval_set: DocDataset,
        clip=0.01,
        n_critic=5,
    ):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
        self.output_dir = args.output_dir
        self.logging_dir = args.logging_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.train_set = train_set
        self.eval_set = eval_set

        self.train_loader = DataLoader(train_set, batch_size=args.train_batch_size)
        self.eval_loader = DataLoader(eval_set, batch_size=args.eval_batch_size)
        
        if args.resume_from_checkpoint:
            last_checkpoint = self.get_last_checkpoint()
            self.model.load_weights_from_pretrained(last_checkpoint)

            trainer_state_file = os.path.join(last_checkpoint, self.TRAINER_STATE_FILE)
            with open(trainer_state_file, "r") as reader:
                trainer_state = json.load(reader)
            self.state = BATMTrainerState(**trainer_state)

            optimizer_state_file = os.path.join(last_checkpoint, self.OPTIMIZER_STATE_FILE)
            self.load_optimizer_state(optimizer_state_file)
        
        else:
            self.state = BATMTrainerState(clip=clip, n_critic=n_critic)
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
            
            self.optimizer_encoder = torch.optim.Adam(
                self.model.encoder_network.parameters(),
                lr=args.lr,
                betas=(args.beta1,args.beta2),
                eps=args.epsilon,
                weight_decay=args.weight_decay,
            )

            self.optimizer_generator = torch.optim.Adam(
                self.model.generator_network.parameters(),
                lr=args.lr,
                betas=(args.beta1,args.beta2),
                eps=args.epsilon,
                weight_decay=args.weight_decay,
            )

            self.optimizer_discriminator = torch.optim.Adam(
                self.model.discriminator_network.parameters(),
                lr=args.lr,
                betas=(args.beta1,args.beta2),
                eps=args.epsilon,
                weight_decay=args.weight_decay,
            )

        self.compute_metrics = args.compute_metrics

        self.train_progress, self.eval_progress = None, None
        self.logger = None
    
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

                # Update training progress
                current_step += 1
                self.state.train_progress = current_step
                self.train_progress.update(1)

                # Forward the model and compute loss backward
                bow = batch.to(self.device)
                bow /= torch.sum(bow, dim=1, keepdim=True)

                self.optimizer_discriminator.zero_grad()
                
                loss_params, _ = self.model(bow)
                loss_discriminator = self.model.loss(*loss_params)
                self.state.train_loss_discriminator = (
                    (self.state.train_loss_discriminator * (current_step - 1) + loss_discriminator.item()) / current_step
                )
                loss_discriminator.backward()

                # Update the discriminator parameters
                self.optimizer_discriminator.step()
                
                # Clipping
                for param in self.model.discriminator_network.parameters():
                    param.data.clamp_(-self.state.clip, self.state.clip)

                # Update the generator and encoder parameters
                if current_step % self.state.n_critic == 0:
                    (p_real, p_fake), _ = self.model(bow)
                    
                    self.optimizer_generator.zero_grad()
                    loss_generator = -1.0 * torch.mean(p_fake)
                    loss_generator.backward()
                    self.state.train_loss_generator = (
                        self.state.n_critic * (self.state.train_loss_generator * (current_step - 1) / self.state.n_critic + loss_generator.item()) / current_step
                    )
                    self.optimizer_generator.step()

                    self.optimizer_encoder.zero_grad()
                    loss_encoder = torch.mean(p_real)
                    loss_encoder.backward()
                    self.state.train_loss_encoder = (
                        self.state.n_critic * (self.state.train_loss_encoder * (current_step - 1) / self.state.n_critic + loss_encoder.item()) / current_step
                    )
                    self.optimizer_encoder.step()

                # Evaluate metrics and perform logging
                if self.state.eval_steps > 0 and current_step % self.state.eval_steps == 0:
                    self.evaluate()

                # Save training checkpoint
                if self.state.save_steps > 0 and current_step % self.state.save_steps == 0:
                    checkpoint_dir = os.path.join(self.output_dir, f"{self.CHECKPOINT_PREFIX}-{current_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.model.save_pretrained(checkpoint_dir)
                    self.save_trainer_state(checkpoint_dir)
                    self.save_optimizer_state(checkpoint_dir)

                # Break the training loop when reaching num_train_steps
                if current_step == self.state.total_train_steps: break
    
    def evaluate(self, sampled_context=None):
        if sampled_context is None:
            sampled_context = SAMPLED_CONTEXT

        metrics, topic_words, sampled_words = self.evaluation_loop(sampled_context)
        
        metrics["train_loss_D"] = self.state.train_loss_discriminator
        metrics["train_loss_E"] = self.state.train_loss_encoder
        metrics["train_loss_G"] = self.state.train_loss_generator

        self.state.eval_results.append(metrics)

        fmt_output = f"Step {self.state.train_progress}:\n"
        fmt_output += str(metrics) + "\n" + "-"*100 + "\n"
        for topic_i_words in topic_words:
            fmt_output += str(topic_i_words) + "\n"
        fmt_output += "-"*100 + "\n" + "Sampled words:\n"
        for sampled_i_context, sampled_i_words in zip(sampled_context, sampled_words):
            fmt_output += str(sampled_i_context) + " -> " + str(sampled_i_words) + "\n"
        fmt_output += "="*100

        with open(os.path.join(self.logging_dir, self.LOGGING_FILE), "a+") as writer:
            writer.write(fmt_output + "\n")

        if self.logger is None:
            self.logger = TableLogger(columns=["steps"]+list(metrics.keys()))
        
        self.logger(self.state.train_progress, *tuple(metrics.values()))
    
    def evaluation_loop(self, sampled_context: List[List[str]]):
        self.model.eval()

        if self.eval_progress is None:
            self.eval_progress = tqdm(range(len(self.eval_loader)))
        else:
            self.eval_progress.refresh()
            self.eval_progress.reset()
        
        eval_loss_discriminator = 0
        eval_loss_generator = 0
        eval_loss_encoder = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                bow = batch.to(self.device)
                (p_real, p_fake), _ = self.model(bow)
                loss_discriminator = self.model.loss(p_real, p_fake)
                loss_generator = -torch.mean(p_fake)
                loss_encoder = torch.mean(p_real)
                eval_loss_discriminator += loss_discriminator.item()
                eval_loss_generator += loss_generator.item()
                eval_loss_encoder += loss_encoder.item()
                self.eval_progress.update(1)
        eval_loss_discriminator /= len(self.eval_loader)
        eval_loss_generator /= len(self.eval_loader)
        eval_loss_encoder /= len(self.eval_loader)

        topic_words = self.model.get_topic_words()
        topic_words = [
            [self.train_set.dictionary.id2token[word_id] for word_id in topic_i_words]
            for topic_i_words in topic_words
        ]

        bow = _doc2bow(self.train_set, sampled_context)
        sampled_words = self.model.sample(bow)
        sampled_words = [
            [self.train_set.dictionary.id2token[word_id] for word_id in sampled_i_words]
            for sampled_i_words in sampled_words
        ]

        if self.compute_metrics:
            ntm_metrics = self.compute_ntm_metrics(topic_words)
            return {
                **ntm_metrics,
                "eval_loss_D": eval_loss_discriminator,
                "eval_loss_E": eval_loss_encoder,
                "eval_loss_G": eval_loss_generator,
            }, topic_words, sampled_words
        else:
            return {
                "eval_loss_D": eval_loss_discriminator,
                "eval_loss_E": eval_loss_encoder,
                "eval_loss_G": eval_loss_generator,
            }, topic_words, sampled_words

    def compute_ntm_metrics(self, topic_words: List[List[str]]):
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
    
    def save_trainer_state(self, save_directory: str):
        trainer_state_file = os.path.join(save_directory, self.TRAINER_STATE_FILE)
        def _to_json_string(state):
            return json.dumps(state, indent=2, sort_keys=True) + "\n"
        with open(trainer_state_file, "w") as writer:
            writer.write(_to_json_string(self.state.to_dict()))
    
    def save_optimizer_state(self, save_directory: str):
        optimizer_state_file = os.path.join(save_directory, self.OPTIMIZER_STATE_FILE)
        torch.save({
            "encoder": self.optimizer_encoder.state_dict(),
            "generator": self.optimizer_generator.state_dict(),
            "discriminator": self.optimizer_discriminator.state_dict(),
        }, optimizer_state_file)
    
    def load_optimizer_state(self, optimizer_path: str):
        self.optimizer_encoder = torch.optim.Adam(self.model.encoder_network.parameters())
        self.optimizer_generator = torch.optim.Adam(self.model.generator_network.parameters())
        self.optimizer_discriminator = torch.optim.Adam(self.model.discriminator_network.parameters())

        optimizer_state = torch.load(optimizer_path)
        self.optimizer_encoder.load_state_dict(optimizer_state["encoder"])
        self.optimizer_generator.load_state_dict(optimizer_state["generator"])
        self.optimizer_discriminator.load_state_dict(optimizer_state["discriminator"])
    
    def get_last_checkpoint(self):
        re_checkpoint = re.compile(r"^" + self.CHECKPOINT_PREFIX + r"\-(\d+)$")
        contents = [
            path for path in os.listdir(self.output_dir)
            if re_checkpoint.search(path) is not None
            and os.path.isdir(os.path.join(self.output_dir, path))
        ]
        return os.path.join(self.output_dir, max(contents, key=lambda x: int(re_checkpoint.search(x).groups()[0])))
