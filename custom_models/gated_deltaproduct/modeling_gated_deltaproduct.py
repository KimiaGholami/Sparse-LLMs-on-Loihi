import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from fla.layers import GatedDeltaProduct   # second-order layer
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP
from fla.modules.activations import swiglu_linear

from .config_gated_deltaproduct import GatedDeltaProductConfig


class GatedDeltaProductMLP(nn.Module):
    def __init__(self, config: GatedDeltaProductConfig):
        super().__init__()
        hidden_size = config.hidden_size
        inter = config.intermediate_size or int(hidden_size * config.hidden_ratio)
        # Round to multiple of 64 for efficiency
        inter = (inter + 63) // 64 * 64

        self.gate_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate, up = self.gate_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class GatedDeltaProductBlock(nn.Module):
    def __init__(self, config: GatedDeltaProductConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.attn = GatedDeltaProduct(
            hidden_size=config.hidden_size,
            expand_v=config.expand_v,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            mode=config.attn_mode,
            use_output_gate=config.use_output_gate,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            layer_idx=layer_idx,
            norm_eps=config.norm_eps,
            use_forget_gate=config.use_forget_gate,
            allow_neg_eigval=config.allow_neg_eigval,
            num_householder=config.num_householder,  # key second-order param
        )
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, _, past_key_values = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_values


class GatedDeltaProductModel(PreTrainedModel):
    config_class = GatedDeltaProductConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: GatedDeltaProductConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GatedDeltaProductBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, past_key_values = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, past_key_values, use_cache
                )
            else:
                hidden_states, past_key_values = layer(
                    hidden_states,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class GatedDeltaProductForCausalLM(PreTrainedModel):
    config_class = GatedDeltaProductConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GatedDeltaProductConfig):
        super().__init__(config)
        self.model = GatedDeltaProductModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.fuse_cross_entropy:
            self.criterion = FusedCrossEntropyLoss(inplace_backward=True)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        loss = None
        logits = None

        if labels is not None:
            # Shift tokens for next-token prediction
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_hidden = shift_hidden.view(-1, self.config.hidden_size)
            shift_labels = shift_labels.view(-1)
            loss = self.criterion(self.lm_head(shift_hidden), shift_labels)
        else:
            logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)