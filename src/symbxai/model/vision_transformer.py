import torch
from transformers import ViTForImageClassification
import math
from torch import nn as nn
from ..lrp.rules import gamma
from ..lrp.core import ModifiedLinear, ModifiedLayerNorm, ModifiedAct


class ModifiedViTSelfAttention(nn.Module):
    def __init__(self, self_attention):
        super(ModifiedViTSelfAttention, self).__init__()
        self.query = ModifiedLinear(fc=self_attention.query, transform='gamma', gam=0.05)
        self.key = ModifiedLinear(fc=self_attention.key, transform='gamma', gam=0.05)
        self.value = ModifiedLinear(fc=self_attention.value, transform='gamma', gam=0.05)

        self.num_attention_heads = self_attention.num_attention_heads
        self.attention_head_size = self_attention.attention_head_size
        self.all_head_size = self_attention.all_head_size

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores).detach()

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, )

        return outputs


class ModifiedViTSelfOutput(nn.Module):
    def __init__(self, self_output):
        super(ModifiedViTSelfOutput, self).__init__()
        self.dense = ModifiedLinear(fc=self_output.dense, transform='gamma', gam=0.05)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)

        return hidden_states


class ModifiedViTAttention(nn.Module):
    def __init__(self, attention):
        super(ModifiedViTAttention, self).__init__()
        self.attention = ModifiedViTSelfAttention(attention.attention)
        self.output = ModifiedViTSelfOutput(attention.output)

    def forward(self, hidden_states):
        self_outputs = self.attention(hidden_states)
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output, ) + self_outputs[1:]

        return outputs


class ModifiedViTIntermediate(nn.Module):
    def __init__(self, intermediate):
        super(ModifiedViTIntermediate, self).__init__()
        self.dense = ModifiedLinear(fc=intermediate.dense, transform='gamma', gam=0.05)
        self.intermediate_act_fn = ModifiedAct(intermediate.intermediate_act_fn)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ModifiedViTOutput(nn.Module):
    def __init__(self, output):
        super(ModifiedViTOutput, self).__init__()
        self.dense = ModifiedLinear(fc=output.dense, transform='gamma', gam=0.05)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states + input_tensor

        return hidden_states


class ModifiedViTLayer(nn.Module):
    def __init__(self, layer):
        super(ModifiedViTLayer, self).__init__()
        self.attention = ModifiedViTAttention(layer.attention)
        self.intermediate = ModifiedViTIntermediate(layer.intermediate)
        self.output = ModifiedViTOutput(layer.output)
        self.layernorm_before = ModifiedLayerNorm(norm_layer=layer.layernorm_before)
        self.layernorm_after = ModifiedLayerNorm(norm_layer=layer.layernorm_after)

    def forward(self, hidden_states):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states)  # # In ViT, LayerNorm is applied before self-attention.
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # First residual connection.
        hidden_states = attention_output + hidden_states

        # In ViT, LayerNorm is also applied after self-attention.
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # Second residual connection is done here.
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ModifiedViTEncoder(nn.Module):
    def __init__(self, encoder):
        super(ModifiedViTEncoder, self).__init__()
        layers = []
        for i, layer in enumerate(encoder.layer):
            layers.append(ModifiedViTLayer(layer))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states):
        for i, layer in enumerate(self.layer):
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]
        return hidden_states


class ModifiedViTPooler(nn.Module):
    def __init__(self, pooler):
        super(ModifiedViTPooler, self).__init__()
        self.dense = ModifiedLinear(fc=pooler.dense, transform='gamma', gam=0.05)
        self.activation = ModifiedAct(pooler.activation)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class ModifiedViTModel(nn.Module):
    def __init__(self, vit):
        super(ModifiedViTModel, self).__init__()
        self.encoder = ModifiedViTEncoder(vit.encoder)
        self.layernorm = ModifiedLayerNorm(vit.layernorm)
        self.pooler = ModifiedViTPooler(vit.pooler) if vit.pooler is not None else None

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        sequence_output = self.layernorm(encoder_outputs)

        pooled_output = self.pooler(encoder_outputs) if self.pooler is not None else sequence_output

        return pooled_output


class ModifiedViTForImageClassification(nn.Module):
    def __init__(self, vit_classification):
        super(ModifiedViTForImageClassification, self).__init__()
        self.vit = ModifiedViTModel(vit_classification.vit)
        self.classifier = ModifiedLinear(fc=vit_classification.classifier, transform='gamma', gam=0.05)

    def forward(self, x):
        outputs = self.vit(x)
        logits = self.classifier(sequence_output[:, 0])

        return logits
