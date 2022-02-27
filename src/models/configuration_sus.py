from transformers.configuration_utils import PretrainedConfig
from transformers.models.pegasus.configuration_pegasus import PegasusConfig


class SusConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
    }
    const_attr = [
        "vocab_size",
        "max_position_embeddings",
    ]
    
    def __init__(
        self,
        vocab_size=96105,
        max_position_embeddings=1024,
        max_input_sentences=128,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        stacked_encoder_type="transformer",
        stacked_layers=3,
        stacked_ffn_dim=4096,
        stacked_attention_heads=16,
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
        bow_size=96105,
        vae_hidden_dims=(1024,512,256),
        topic_dim=100,
        pad_token_id=0,
        eos_token_id=1,
        cls_token_id=96103,
        sep_token_id=96104,
        forced_eos_token_id=1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.max_input_sentences = max_input_sentences
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.stacked_encoder_type = stacked_encoder_type
        self.stacked_ffn_dim = stacked_ffn_dim
        self.stacked_layers = stacked_layers
        self.stacked_attention_heads = stacked_attention_heads
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
        self.bow_size = bow_size
        self.vae_hidden_dims = vae_hidden_dims
        self.topic_dim = topic_dim
        self.cls_token_id = cls_token_id
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
    
    def to_pegasus_config(self, is_stacked_layers=False) -> PegasusConfig:
        config_dict = self.__dict__.copy()

        config_dict.pop("bow_size", None)
        config_dict.pop("vae_hidden_dim", None)
        config_dict.pop("topic_dim", None)
        config_dict.pop("cls_token_id", None)
        config_dict.pop("sep_token_id", None)

        if is_stacked_layers:
            config_dict["max_position_embeddings"] = config_dict.pop("max_input_sentences", 128)
            config_dict["encoder_layers"] = config_dict.pop("stacked_layers", 3)
            config_dict["encoder_ffn_dim"] = config_dict.pop("stacked_ffn_dim", 4096)
            config_dict["encoder_attention_heads"] = config_dict.pop("stacked_attention_heads", 16)

        else:
            popped_attr = [
                "max_position_embeddings",
                "stacked_encoder_type",
                "stacked_layers",
                "stacked_ffn_dim",
                "stacked_attention_heads",
            ]
            [config_dict.pop(attr, None) for attr in popped_attr]
        
        pegasus_config = PegasusConfig(**config_dict)

        return pegasus_config
    
    def copy_pegasus_config(self, config: PegasusConfig):
        pegasus_config_dict = config.__dict__.copy()

        for attr in self.const_attr:
            pegasus_config_dict[attr] = self.__dict__[attr]
        
        self.__dict__ = {**self.__dict__, **pegasus_config_dict}
