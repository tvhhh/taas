import os
import json
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.pegasus.modeling_pegasus import PegasusEncoder, PegasusDecoder, PegasusModel

from .configuration_sus import SusConfig


def _prepare_layer_ids_for_distillation(teacher_layers, student_layers):
    if student_layers > teacher_layers:
        raise ValueError("Student model must be smaller than teacher model.")
    step = int(round((teacher_layers-1) / (student_layers-1)))
    layers = list(range(0, step * student_layers, step))
    layers[-1] = teacher_layers - 1
    return tuple(layers)


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


class SusNeuralTopicModel(nn.Module):
    def __init__(self, config: SusConfig):
        super().__init__()

        self.config = config

        vae_hidden_dims = (1024, 512, 256)
        encoder_dims = vae_hidden_dims
        decoder_dims = vae_hidden_dims[::-1]
        
        self.encoder = nn.ModuleList([
            nn.Linear(encoder_dims[i-1] if i > 0 else config.bow_size, encoder_dims[i])
            for i in range(len(encoder_dims))
        ])
        
        self.fc_mu = nn.Linear(vae_hidden_dims[-1], config.topic_dim)
        self.fc_logvar = nn.Linear(vae_hidden_dims[-1], config.topic_dim)
        self.fc_theta = nn.Linear(config.topic_dim, config.topic_dim)
        
        self.decoder = nn.ModuleList([nn.Linear(config.topic_dim, decoder_dims[0])] + [
            nn.Linear(decoder_dims[i], decoder_dims[i+1] if i < len(decoder_dims)-1 else config.bow_size)
            for i in range(len(decoder_dims))
        ])

        self.dropout = nn.Dropout(p=config.ntm_dropout)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std, requires_grad=False)
        return eps.mul(std).add_(mu)
   
    def encode(self, x):
        hid = x
        for layer in self.encoder:
            hid = F.relu(self.dropout(layer(x)))
        
        mu, logvar = self.fc_mu(hid), self.fc_logvar(hid)
        
        return mu, logvar

    def decode(self, z):
        hid = z
        for i, layer in enumerate(self.decoder):
            hid = layer(hid)
            if i < len(self.decoder) - 1:
                hid = F.relu(self.dropout(hid))
        return hid
    
    def inference(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        theta = self.fc_theta(z)
        theta = torch.softmax(theta, dim=1)

        return theta

    def forward(self, x, output_loss=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        theta = self.fc_theta(z)
        theta = torch.softmax(theta, dim=1)
        
        x_recons = self.decode(theta)
        
        loss = None
        if output_loss:
            logsoftmax = torch.log_softmax(x_recons, dim=1)
            rec_loss = -1.0 * torch.sum(x * logsoftmax)
            kl_div = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
            loss = (rec_loss + kl_div) / x.size(0)
        
        outputs = tuple(t for t in (x_recons, theta, loss) if t is not None)

        return outputs
    
    def get_top_topic_words(self, topK=20):
        z = torch.eye(self.config.topic_dim).to(self.device)
        word_dist = torch.softmax(self.decode(z), dim=1)
        _, word_ids = torch.topk(word_dist, topK, dim=1)
        word_ids = word_ids.cpu().tolist()
        return word_ids
    
    def save_pretrained(self, save_directory):
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        # Save configuration
        config_file = os.path.join(save_directory, "config.json")
        config_dict = self.config.__dict__.copy()
        saved_config_dict = {
            attr: config_dict[attr] for attr in ("bow_size", "ntm_dropout", "topic_dim")
        }
        def _to_json_string(config_dict):
            return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
        with open(config_file, "w", encoding="utf-8") as writer:
            writer.write(_to_json_string(saved_config_dict))
        
        # Save model state dict
        state_dict_file = os.path.join(save_directory, "model_state_dict.pt")
        state_dict = self.state_dict()
        torch.save(state_dict, state_dict_file)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        # Load configuration
        config_file = os.path.join(pretrained_model_path, "config.json")
        with open(config_file, "r") as reader:
            config_dict = json.load(reader)
        config = SusConfig(**config_dict)

        # Load model state dict
        ntm = cls(config)
        state_dict_file = os.path.join(pretrained_model_path, "model_state_dict.pt")
        state_dict = torch.load(state_dict_file)
        ntm.load_state_dict(state_dict)

        return ntm


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
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PegasusDecoder, PegasusEncoder)):
            module.gradient_checkpointing = value


class SusEncoder(SusPreTrainedModel):
    def __init__(
        self,
        config: SusConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        from_pretrained_pegasus: Optional[PegasusEncoder] = None,
        shrink_pegasus_large: Optional[bool] = False,
    ):
        super().__init__(config)

        if from_pretrained_pegasus is not None:
            self.pegasus_encoder = from_pretrained_pegasus
            if shrink_pegasus_large:
                layer_ids = _prepare_layer_ids_for_distillation(
                    from_pretrained_pegasus.config.encoder_layers,
                    config.encoder_layers,
                )
                self.pegasus_encoder.layers = nn.ModuleList([
                    from_pretrained_pegasus.layers[i] for i in layer_ids
                ])

        else:
            pegasus_config = self.config.to_pegasus_config()
            self.pegasus_encoder = PegasusEncoder(pegasus_config, embed_tokens)
    
    def get_input_embeddings(self):
        return self.pegasus_encoder.embed_tokens
    
    def set_input_embeddings(self, value):
        self.pegasus_encoder.embed_tokens = value
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.pegasus_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SusDecoder(SusPreTrainedModel):
    def __init__(
        self,
        config: SusConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        from_pretrained_pegasus: Optional[PegasusDecoder] = None,
        shrink_pegasus_large: Optional[bool] = False,
    ):
        super().__init__(config)

        if from_pretrained_pegasus is not None:
            self.pegasus_decoder = from_pretrained_pegasus
            if shrink_pegasus_large:
                layer_ids = _prepare_layer_ids_for_distillation(
                    from_pretrained_pegasus.config.encoder_layers,
                    config.decoder_layers,
                )
                self.pegasus_decoder.layers = nn.ModuleList([
                    from_pretrained_pegasus.layers[i] for i in layer_ids
                ])

        else:
            pegasus_config = self.config.to_pegasus_config()
            self.pegasus_decoder = PegasusDecoder(pegasus_config, embed_tokens)
    
    def get_input_embeddings(self):
        return self.pegasus_decoder.embed_tokens
    
    def set_input_embeddings(self, value):
        self.pegasus_decoder.embed_tokens = value
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.pegasus_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SusModel(SusPreTrainedModel):
    def __init__(
        self,
        config: SusConfig,
        pretrained_pegasus_path: Optional[str] = None,
        shrink_pegasus_large: Optional[bool] = False,
        *from_pretrained_args,
        **from_pretrained_kwargs,
    ):
        super().__init__(config)

        if pretrained_pegasus_path is not None:
            pegasus = PegasusModel.from_pretrained(
                pretrained_pegasus_path,
                *from_pretrained_args,
                **from_pretrained_kwargs
            )
            config.copy_pegasus_config(pegasus.config, shrink_pegasus_large)
            
            self.shared_embed = pegasus.get_input_embeddings()
            
            self.encoder = SusEncoder(
                config=config,
                from_pretrained_pegasus=pegasus.get_encoder(),
                shrink_pegasus_large=shrink_pegasus_large,
            )
            
            self.decoder = SusDecoder(
                config=config,
                from_pretrained_pegasus=pegasus.get_decoder(),
                shrink_pegasus_large=shrink_pegasus_large,
            )
        
        else:
            padding_idx, vocab_size = config.pad_token_id, config.vocab_size
            self.shared_embed = nn.Embedding(vocab_size, config.d_model, padding_idx)
            self.encoder = SusEncoder(config, embed_tokens=self.shared_embed)
            self.decoder = SusDecoder(config, embed_tokens=self.shared_embed)
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_input_embeddings(self):
        return self.shared_embed
    
    def set_input_embeddings(self, value):
        self.shared_embed = value
        self.encoder.set_input_embeddings(value)
        self.decoder.set_input_embeddings(value)
    
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
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class SusForConditionalGeneration(SusPreTrainedModel):
    def __init__(
        self,
        config: SusConfig,
        pretrained_pegasus_path: Optional[str] = None,
        shrink_pegasus_large: Optional[bool] = False,
        *from_pretrained_args,
        **from_pretrained_kwargs,
    ):
        super().__init__(config)

        self.model = SusModel(
            config=config,
            pretrained_pegasus_path=pretrained_pegasus_path,
            shrink_pegasus_large=shrink_pegasus_large,
            *from_pretrained_args,
            **from_pretrained_kwargs,
        )

        self.config = config
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
    
    def get_encoder(self):
        return self.model.get_encoder()
    
    def get_decoder(self):
        return self.model.get_decoder()
    
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
        )
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        
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
