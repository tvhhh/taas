import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.pegasus.modeling_pegasus import (
    PegasusEncoder,
    PegasusDecoder,
    PegasusModel,
    PegasusSinusoidalPositionalEmbedding
)
from typing import Optional, Tuple

from .configuration_sus import SusConfig
from .modeling_topic import NeuralTopicModel


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


@dataclass
class SusModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    topic_model_loss: Optional[torch.FloatTensor] = None


class SusPreTrainedModel(PreTrainedModel):
    config_class = SusConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, PegasusSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PegasusDecoder, PegasusEncoder)):
            module.gradient_checkpointing = value


class SusModel(SusPreTrainedModel):
    def __init__(
        self,
        config: SusConfig,
        pretrained_pegasus_path: Optional[str] = None,
        pretrained_ntm_path: Optional[str] = None,
    ):
        super().__init__(config)

        if pretrained_pegasus_path is not None:
            pegasus = PegasusModel.from_pretrained(pretrained_pegasus_path)
            config.copy_pegasus_config(pegasus.config)
            self.shared_embed = pegasus.get_input_embeddings()
            self.encoder = pegasus.get_encoder()
            self.decoder = pegasus.get_decoder()
        else:
            padding_idx, vocab_size = config.pad_token_id, config.vocab_size
            self.shared_embed = nn.Embedding(vocab_size, config.d_model, padding_idx)
            pegasus_config = config.to_pegasus_config()
            self.encoder = PegasusEncoder(pegasus_config, embed_tokens=self.shared_embed)
            self.decoder = PegasusDecoder(pegasus_config, embed_tokens=self.shared_embed)
            self.encoder.post_init()
            self.decoder.post_init()
        
        if config.use_ntm:
            if pretrained_ntm_path is not None:
                self.neural_topic_model = NeuralTopicModel.from_pretrained(pretrained_ntm_path)
                config.copy_ntm_config(self.neural_topic_model.config)
            else:
                ntm_config = config.to_ntm_config()
                self.neural_topic_model = NeuralTopicModel(**ntm_config)
            
            self.topic_weights = nn.Parameter(torch.empty((config.n_topics, config.d_model)))
            self.hidden_state_weights = nn.Parameter(torch.empty((config.d_model, config.d_model)))
            self.gating_bias = nn.Parameter(torch.zeros(config.d_model))
            nn.init.xavier_normal_(self.topic_weights)
            nn.init.xavier_normal_(self.hidden_state_weights)
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_input_embeddings(self):
        return self.shared_embed
    
    def set_input_embeddings(self, value):
        self.shared_embed = value
        self.encoder.embed_tokens = self.shared_embed
        self.decoder.embed_tokens = self.shared_embed
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        bag_of_words=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        # Integrate latent topics distribution into encoder outputs
        ntm_loss = None
        if self.config.use_ntm and bag_of_words is not None:
            posterior_params, word_dist, theta = self.neural_topic_model(bag_of_words)
            ntm_loss = self.neural_topic_model.loss(*posterior_params, word_dist, bag_of_words)
            gate_vec = torch.sigmoid(
                torch.matmul(theta, self.topic_weights).unsqueeze(1) +          # (batch_size * d_model) -> unsqueeze(1)
                torch.matmul(encoder_outputs[0], self.hidden_state_weights) +   # (batch_size * sequence_length * d_model)
                self.gating_bias                                                # (d_model) -> automatically use last dim
            )
            if isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs["last_hidden_state"] = torch.mul(gate_vec, encoder_outputs[0])
            elif isinstance(encoder_outputs, tuple):
                encoder_outputs[0] = torch.mul(gate_vec, encoder_outputs[0])

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            return outputs + (ntm_loss,) if self.config.use_ntm else outputs

        return SusModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            topic_model_loss=ntm_loss,
        )


class SusForConditionalGeneration(SusPreTrainedModel):
    def __init__(
        self,
        config: SusConfig,
        pretrained_pegasus_path: Optional[str] = None,
        pretrained_ntm_path: Optional[str] = None,
    ):
        super().__init__(config)

        self.model = SusModel(
            config=config,
            pretrained_pegasus_path=pretrained_pegasus_path,
            pretrained_ntm_path=pretrained_ntm_path,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared_embed.num_embeddings)))
    
    def get_encoder(self):
        return self.model.get_encoder()
    
    def get_decoder(self):
        return self.model.get_decoder()
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings
    
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
    
    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        bag_of_words=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bag_of_words=bag_of_words,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        if self.config.use_ntm and bag_of_words is not None:
            masked_lm_loss += self.config.ntm_loss_weight * outputs[-1]
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        bag_of_words=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "bag_of_words": bag_of_words,
        }
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don"t have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
