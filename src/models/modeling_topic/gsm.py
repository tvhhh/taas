import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import NeuralTopicModel


class InferenceNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_topics,
        hidden_sizes,
        dropout=0.0,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(*(
            nn.Sequential(
                nn.Linear(h_in, h_out),
                nn.BatchNorm1d(h_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for h_in, h_out in zip(type(hidden_sizes)([vocab_size]) + hidden_sizes[:-1], hidden_sizes)
        ))

        self.fc_mu = nn.Linear(hidden_sizes[-1], n_topics)
        self.batchnorm_mu = nn.BatchNorm1d(n_topics)

        self.fc_logvar = nn.Linear(hidden_sizes[-1], n_topics)
        self.batchnorm_logvar = nn.BatchNorm1d(n_topics)
    
    def forward(self, x):
        h = self.encoder(x)
        
        mu = self.batchnorm_mu(self.fc_mu(h))
        logvar = self.batchnorm_logvar(self.fc_logvar(h))

        return mu, logvar


class GeneratorNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_topics,
    ):
        super().__init__()

        self.topic_word_dist = nn.Parameter(
            torch.empty((n_topics, vocab_size))
        )
        nn.init.xavier_uniform_(self.topic_word_dist)

        self.batchnorm_generator = nn.BatchNorm1d(vocab_size)
    
    def forward(self, z):
        return self.batchnorm_generator(torch.matmul(z, self.topic_word_dist))


class GSM(NeuralTopicModel):

    def __init__(
        self,
        vocab_size=20000,
        n_topics=100,
        hidden_sizes=(1024,512,256),
        dropout=0.0,
        prior_params=None,
    ):
        super().__init__()

        assert isinstance(vocab_size, int), f"vocab_size must be integer, got {type(vocab_size)}."
        assert isinstance(n_topics, int), f"n_topics must be integer, got {type(n_topics)}."
        assert isinstance(hidden_sizes, (list, tuple)), f"hidden_sizes must be list or tuple, got {type(hidden_sizes)}."
        assert dropout >= 0, f"dropout must be non-negative, got {dropout}."

        # Use standard Gaussian distribution if prior is not given
        if prior_params is None:
            prior_params = (0, 1)
        else:
            assert isinstance(prior_params, (list, tuple)) and len(prior_params) == 2, \
            f"prior distribution parameters must contain mean and variance, got {prior_params}."
        
        self.config = {
            "vocab_size": vocab_size,
            "n_topics": n_topics,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "prior_params": prior_params,
        }
        
        p_mean, p_variance = prior_params
        self.register_buffer("prior_mean", torch.tensor([p_mean] * n_topics))
        self.register_buffer("prior_variance", torch.tensor([p_variance] * n_topics))

        self.inference_network = InferenceNetwork(vocab_size, n_topics, hidden_sizes, dropout)
        self.generator_network = GeneratorNetwork(vocab_size, n_topics)
        self.fc_mixture = nn.Linear(n_topics, n_topics)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        posterior_mean, posterior_logvar = self.inference_network(x)
        posterior_var = torch.exp(posterior_logvar)

        # Latent topics distribution
        z = self.reparameterize(posterior_mean, posterior_logvar)
        theta = F.softmax(self.fc_mixture(z), dim=1)

        x_recons = self.generator_network(theta)
        word_dist = F.softmax(x_recons, dim=1)

        # Return posterior distribution parameters, latent topics distribution, reconstructed words distribution
        return (posterior_mean, posterior_var, word_dist, x), theta
    
    def loss(self, posterior_mean, posterior_var, word_dist, inputs):
        # KL divergence: KL(q(x|z) || P(z))
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        prior_mean, prior_var = self.prior_mean, self.prior_variance

        var_division = posterior_var / prior_var
        logvar_division = (torch.log(prior_var) - torch.log(posterior_var))
        diff_terms = (posterior_mean - prior_mean)**2 / prior_var

        KL = 0.5 * torch.sum(logvar_division + var_division + diff_terms - 1, dim=1)

        # Reconstruction loss
        RC = -1.0 * torch.sum(inputs * torch.log(word_dist + 1e-9), dim=1)

        loss = (RC + KL).mean()

        return loss
    
    def get_topic_words(self, topk=10):
        theta = torch.eye(self.config["n_topics"], device=self.device)
        word_dist = F.softmax(self.generator_network(theta), dim=1)
        _, word_ids = torch.topk(word_dist, topk, dim=1)
        word_ids = word_ids.cpu().tolist()
        return word_ids
    
    def sample(self, x, topk=10):
        x = x.to(self.device)
        posterior_mean, posterior_logvar = self.inference_network(x)
        z = self.reparameterize(posterior_mean, posterior_logvar)
        theta = F.softmax(self.fc_mixture(z), dim=1)
        word_dist = F.softmax(self.generator_network(theta), dim=1)
        _, word_ids = torch.topk(word_dist, topk, dim=1)
        word_ids = word_ids.cpu().tolist()
        return word_ids
