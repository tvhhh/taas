import numpy as np
import torch
import torch.nn as nn

from .utils import NeuralTopicModel


class EncoderNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_topics,
        hidden_dim,
        leaky_relu_slope=0.1,
        dropout=0.0,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_topics),
            nn.Softmax(dim=1)
        )
    
    def forward(self, bow):
        theta = self.encoder(bow)
        d_real = torch.cat([theta, bow], dim=1)
        return d_real


class GeneratorNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_topics,
        hidden_dim,
        leaky_relu_slope=0.1,
        dropout=0.0,
    ):
        super().__init__()

        self.generator = nn.Sequential(
            nn.Linear(n_topics, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
            nn.Softmax(dim=1),
        )
    
    def inference(self, theta):
        return self.generator(theta)
    
    def forward(self, theta):
        bow = self.generator(theta)
        d_fake = torch.cat([theta, bow], dim=1)
        return d_fake


class DiscriminatorNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_topics,
        hidden_dim,
        leaky_relu_slope=0.1,
        dropout=0.0,
    ):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(vocab_size + n_topics, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, pair):
        score = self.discriminator(pair)
        return score


class BATM(NeuralTopicModel):
    
    def __init__(
        self,
        vocab_size=20000,
        n_topics=100,
        hidden_dim=1024,
        leaky_relu_slope=0.1,
        dropout=0.0,
        prior_param=None,
    ):
        super().__init__()

        assert isinstance(vocab_size, int), f"vocab_size must be integer, got {type(vocab_size)}"
        assert isinstance(n_topics, int), f"n_topics must be integer, got {type(n_topics)}"
        assert isinstance(hidden_dim, int), f"hidden_dim must be integer, got {type(hidden_dim)}"
        assert leaky_relu_slope >= 0, f"leaky_relu_slope must be non-negative, got {leaky_relu_slope}"
        assert dropout >= 0, f"dropout must be non-negative, got {dropout}"

        self.config = {
            "vocab_size": vocab_size,
            "n_topics": n_topics,
            "hidden_dim": hidden_dim,
            "leaky_relu_slope": leaky_relu_slope,
            "dropout": dropout,
            "prior_param": prior_param,
        }

        self.prior_param = prior_param if prior_param is not None else 1 / n_topics

        self.encoder_network = EncoderNetwork(vocab_size, n_topics, hidden_dim, leaky_relu_slope, dropout)
        self.generator_network = GeneratorNetwork(vocab_size, n_topics, hidden_dim, leaky_relu_slope, dropout)
        self.discriminator_network = DiscriminatorNetwork(vocab_size, n_topics, hidden_dim, leaky_relu_slope, dropout)
    
    def forward(self, x):
        d_real = self.encoder_network(x)
        theta = d_real[:, :self.config["n_topics"]]

        theta_fake = torch.from_numpy(
            np.random.dirichlet(
                alpha=self.prior_param*np.ones(self.config["n_topics"]),
                size=len(x)
            )
        ).float().to(self.device)
        d_fake = self.generator_network(theta_fake)

        p_real = self.discriminator_network(d_real)
        p_fake = self.discriminator_network(d_fake)

        return (p_real, p_fake), theta
    
    def loss(self, p_real, p_fake):
        return torch.mean(p_fake) - torch.mean(p_real)
    
    def get_topic_words(self, topk=10):
        theta = torch.eye(self.config["n_topics"], device=self.device)
        word_dist = self.generator_network.inference(theta)
        _, word_ids = torch.topk(word_dist, topk, dim=1)
        word_ids = word_ids.cpu().tolist()
        return word_ids
    
    def sample(self, x, topk=10):
        x = x.to(self.device)
        d_real = self.encoder_network(x)
        theta = d_real[:, :self.config["n_topics"]]
        word_dist = self.generator_network.inference(theta)
        _, word_ids = torch.topk(word_dist, topk, dim=1)
        word_ids = word_ids.cpu().tolist()
        return word_ids
