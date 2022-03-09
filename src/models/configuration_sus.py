from transformers.configuration_utils import PretrainedConfig
from transformers.models.pegasus.configuration_pegasus import PegasusConfig
from typing import Any, Dict


class SusConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]
    keys_to_ignore_pegasus = ["use_ntm", "corpus_size", "n_topics", "ntm_activation", "ntm_dropout", "ntm_prior_params", "ntm_loss_weight"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
    }
    
    def __init__(
        self,
        vocab_size=96103,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=0,
        classifier_dropout=0.0,
        scale_embedding=False,
        corpus_size=100000,
        use_ntm=True,
        n_topics=100,
        ntm_hidden_sizes=(1024,512,256),
        ntm_activation="softplus",
        ntm_dropout=0.0,
        ntm_prior_params=(0,1),
        ntm_loss_weight=0.1,
        pad_token_id=0,
        eos_token_id=1,
        forced_eos_token_id=1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding
        
        self.use_ntm = use_ntm
        self.corpus_size = corpus_size
        self.n_topics = n_topics
        self.ntm_hidden_sizes = ntm_hidden_sizes
        self.ntm_activation = ntm_activation
        self.ntm_dropout = ntm_dropout
        self.ntm_prior_params = ntm_prior_params
        self.ntm_loss_weight = ntm_loss_weight
        
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
    
    def to_pegasus_config(self) -> PegasusConfig:
        config_dict = self.__dict__.copy()

        for attr in self.keys_to_ignore_pegasus:
            config_dict.pop(attr, None)
        
        pegasus_config = PegasusConfig(**config_dict)

        return pegasus_config
    
    def copy_pegasus_config(self, config: PegasusConfig):
        pegasus_config_dict = config.__dict__.copy()
        self.__dict__ = {**self.__dict__, **pegasus_config_dict}
    
    def to_ntm_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.corpus_size,
            "n_topics": self.n_topics,
            "hidden_sizes": self.ntm_hidden_sizes,
            "activation": self.ntm_activation,
            "dropout": self.ntm_dropout,
            "prior_params": self.ntm_prior_params,
        }

    def copy_ntm_config(self, config: Dict[str, Any]):
        mapped_attrs = {
            "corpus_size": "vocab_size",
            "n_topics": "n_topics",
            "ntm_hidden_sizes": "hidden_sizes",
            "ntm_activation": "activation",
            "ntm_dropout": "dropout",
            "ntm_prior_params": "prior_params",
        }
        for attr, ntm_attr in mapped_attrs.items():
            self.__dict__[attr] = config[ntm_attr]
