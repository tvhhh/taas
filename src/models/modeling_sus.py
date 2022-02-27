from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput, TokenClassifierOutput
from transformers.models.pegasus.modeling_pegasus import PegasusEncoder, PegasusDecoder, PegasusModel

from .configuration_sus import SusConfig


def _prepare_layer_ids_for_distill(teacher_layers, student_layers):
    if student_layers > teacher_layers:
        raise ValueError("Student model must be smaller than teacher model.")
    step = int(round(teacher_layers / student_layers))
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

        encoder_dims = config.vae_hidden_dims
        decoder_dims = config.vae_hidden_dims[::-1]
        self.encoder = nn.ModuleList([
            nn.Linear(encoder_dims[i-1] if i > 1 else config.bow_size, encoder_dims[i])
            for i in range(len(encoder_dims))
        ])
        self.fc_mu = nn.Linear(config.vae_hidden_dims[-1], config.topic_dim)
        self.fc_log_var = nn.Linear(config.vae_hidden_dims[-1], config.topic_dim)
        self.fc = nn.Linear(config.topic_dim, config.topic_dim)
        self.decoder = nn.ModuleList([nn.Linear(config.topic_dim, decoder_dims[0])] + [
            nn.Linear(decoder_dims[i], decoder_dims[i+1] if i < len(decoder_dims)-1 else config.bow_size)
            for i in range(len(decoder_dims))
        ])
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std, requires_grad=False)
        return eps.mul(std).add_(mu)
   
    def encode(self, x):
        h = x
        for layer in self.encoder:
            h = torch.tanh(layer(x))
        
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def decode(self, z):
        h = z
        for layer in self.decoder:
            h = F.relu(layer(h))
        return h

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        
        theta = self.fc(z)
        theta = torch.softmax(theta, dim=1)
        
        x_recons = self.decode(theta)
        logsoftmax = torch.log_softmax(x_recons, dim=1)
        rec_loss = -1.0 * torch.sum(x * logsoftmax)

        kl_div = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))

        loss = (rec_loss + kl_div) / x.size(0)

        return loss, x_recons, theta


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


class SusLinearClassifier(SusPreTrainedModel):
    def __init__(self, config: SusConfig):
        super().__init__(config)

        self.linear = nn.Linear(config.d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
    ):
        logits = self.sigmoid(self.linear(inputs_embeds))
        logits = logits.squeeze(-1) * attention_mask

        return logits


class SusTransformerClassifier(SusPreTrainedModel):
    def __init__(self, config: SusConfig):
        super().__init__(config)

        self.config = config

        transformer_config = self.config.to_pegasus_config(is_stacked_layers=True)
        self.encoder = PegasusEncoder(transformer_config)

        self.linear = nn.Linear(self.config.d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
    ):
        outputs = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        logits = self.sigmoid(self.linear(outputs[0]))
        logits = logits.squeeze(-1) * attention_mask

        return logits


class SusEncoder(SusPreTrainedModel):
    def __init__(
        self,
        config: SusConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        from_pretrained_pegasus: Optional[PegasusEncoder] = None,
        distill_pegasus: Optional[bool] = False,
        copied_layers: Optional[int] = None,
    ):
        super().__init__(config)

        if from_pretrained_pegasus is not None:
            self.pegasus_encoder = from_pretrained_pegasus
            if distill_pegasus and copied_layers is not None and copied_layers > 0:
                layer_ids = _prepare_layer_ids_for_distill(
                    from_pretrained_pegasus.config.encoder_layers,
                    copied_layers,
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
        distill_pegasus: Optional[bool] = False,
        copied_layers: Optional[int] = None,
    ):
        super().__init__(config)

        if from_pretrained_pegasus is not None:
            self.pegasus_decoder = from_pretrained_pegasus
            if distill_pegasus and copied_layers is not None and copied_layers > 0:
                layer_ids = _prepare_layer_ids_for_distill(
                    from_pretrained_pegasus.config.encoder_layers,
                    copied_layers,
                )
                self.pegasus_encoder.layers = nn.ModuleList([
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
        distill_pegasus: Optional[bool] = False,
        copied_encoder_layers: Optional[int] = None,
        copied_decoder_layers: Optional[int] = None,
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
            config.copy_pegasus_config(pegasus.config)
            
            if distill_pegasus:
                if copied_encoder_layers is not None and copied_encoder_layers > 0:
                    config.encoder_layers = copied_encoder_layers
                if copied_decoder_layers is not None and copied_decoder_layers > 0:
                    config.decoder_layers = copied_decoder_layers
            
            self.shared_embed = pegasus.get_input_embeddings()
            
            self.encoder = SusEncoder(
                config,
                from_pretrained_pegasus=pegasus.get_encoder(),
                distill_pegasus=distill_pegasus,
                copied_layers=copied_encoder_layers,
            )
            
            self.decoder = SusDecoder(
                config,
                from_pretrained_pegasus=pegasus.get_decoder(),
                distill_pegasus=distill_pegasus,
                copied_layers=copied_decoder_layers,
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


class SusForExtractiveSummarization(SusPreTrainedModel):
    def __init__(
        self,
        config: SusConfig,
        pretrained_pegasus_path: Optional[str] = None,
        distill_pegasus: Optional[bool] = False,
        copied_encoder_layers: Optional[Tuple[int]] = None,
        copied_decoder_layers: Optional[Tuple[int]] = None,
        *from_pretrained_args,
        **from_pretrained_kwargs,
    ):
        super().__init__(config)

        self.model = SusModel(
            config=config,
            pretrained_pegasus_path=pretrained_pegasus_path,
            distill_pegasus=distill_pegasus,
            copied_encoder_layers=copied_encoder_layers,
            copied_decoder_layers=copied_decoder_layers,
            *from_pretrained_args,
            **from_pretrained_kwargs,
        )

        self.config = config

        stacked_encoder_type = self.config.stacked_encoder_type
        if stacked_encoder_type == "classifier":
            self.ext_module = SusLinearClassifier(config)
        elif stacked_encoder_type == "transformer":
            self.ext_module = SusTransformerClassifier(config)
        else:
            raise ValueError(f"Unsupported stacked layer type {stacked_encoder_type}.")
    
    def get_encoder(self):
        return self.model.get_encoder()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pegasus_outputs = self.model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = pegasus_outputs[0]
        clss, cls_masks = self._prepare_inputs_for_stacked_layers(input_ids, labels)
        cls_inputs_embeds = last_hidden_state[torch.arange(last_hidden_state.size(0)).unsqueeze(1), clss]

        logits = self.ext_module(
            inputs_embeds=cls_inputs_embeds,
            attention_mask=cls_masks,
        )
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()
            masked_labels = labels * cls_masks.float()
            loss = loss_fct(logits, masked_labels.float())
        
        if not return_dict:
            output = (logits,) + pegasus_outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=pegasus_outputs.hidden_states,
            attentions=pegasus_outputs.attentions,
        )
    
    def _prepare_inputs_for_stacked_layers(self, input_ids, labels):
        
        max_sentence_length = labels.size(1)

        def _get_clss_and_cls_mask(input_ids):
            clss_ids = torch.where(input_ids == self.config.cls_token_id)[0].long()
            pad_size = max_sentence_length - len(clss_ids)
            if pad_size >= 0:
                clss_ids = torch.cat((clss_ids, torch.full((pad_size,), -1, dtype=torch.long, device=self.device, requires_grad=False)))
            else:
                clss_ids = clss_ids[:max_sentence_length]
            cls_mask = torch.where(clss_ids == -1, 0, 1)
            return clss_ids.unsqueeze(0), cls_mask.unsqueeze(0)
        
        ids, masks = zip(*map(_get_clss_and_cls_mask, input_ids))
        
        clss = torch.LongTensor().to(self.device)
        torch.cat(ids, out=clss)
        
        cls_masks = torch.Tensor().to(self.device)
        torch.cat(masks, out=cls_masks)
        
        return clss, cls_masks


class SusForAbstractiveSummarization(SusPreTrainedModel):
    def __init__(
        self,
        config: SusConfig,
        pretrained_pegasus_path: Optional[str] = None,
        distill_pegasus: Optional[bool] = False,
        copied_encoder_layers: Optional[Tuple[int]] = None,
        copied_decoder_layers: Optional[Tuple[int]] = None,
        *from_pretrained_args,
        **from_pretrained_kwargs,
    ):
        super().__init__(config)

        self.model = SusModel(
            config=config,
            pretrained_pegasus_path=pretrained_pegasus_path,
            distill_pegasus=distill_pegasus,
            copied_encoder_layers=copied_encoder_layers,
            copied_decoder_layers=copied_decoder_layers,
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
