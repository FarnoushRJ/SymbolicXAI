import torch
from transformers import BertForSequenceClassification
import math
from torch import nn as nn
from ..lrp.rules import gamma
from ..lrp.core import ModifiedLinear, ModifiedLayerNorm, ModifiedAct


def bert_base_uncased_model(
        pretrained_model_name_or_path
):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path
    )
    model.bert.embeddings.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith('embeddings'):
            param.requires_grad = False

    return model


# ------ LRP for BERT Model ------
class ModifiedBertSelfAttention(nn.Module):
    def __init__(self, self_attention, gam=.09):
        super(ModifiedBertSelfAttention, self).__init__()
        self.query = ModifiedLinear(fc=self_attention.query, transform=None)
        self.key = ModifiedLinear(fc=self_attention.key, transform=None)
        self.value = ModifiedLinear(fc=self_attention.value, transform=None)

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
        outputs = context_layer

        return outputs


class ModifiedBertSelfOutput(nn.Module):
    def __init__(self, self_output, gam=.09):
        super(ModifiedBertSelfOutput, self).__init__()
        self.dense = ModifiedLinear(fc=self_output.dense, transform=None)
        self.LayerNorm = ModifiedLayerNorm(norm_layer=self_output.LayerNorm)


    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ModifiedBertAttention(nn.Module):
    def __init__(self, attention, gam=.09):
        super(ModifiedBertAttention, self).__init__()
        self.self = ModifiedBertSelfAttention(attention.self)
        self.output = ModifiedBertSelfOutput(attention.output)

    def forward(self, hidden_states):
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)

        return attention_output


class ModifiedBertIntermediate(nn.Module):
    def __init__(self, intermediate, gam=.15):
        super(ModifiedBertIntermediate, self).__init__()
        self.dense = ModifiedLinear(fc=intermediate.dense, transform=None)
        self.intermediate_act_fn = ModifiedAct(intermediate.intermediate_act_fn)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ModifiedBertOutput(nn.Module):
    def __init__(self, output, gam=.09):
        super(ModifiedBertOutput, self).__init__()
        self.dense = ModifiedLinear(fc=output.dense, transform=None)
        self.LayerNorm = ModifiedLayerNorm(norm_layer=output.LayerNorm)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ModifiedBertLayer(nn.Module):
    def __init__(self, layer, gam=.15):
        super(ModifiedBertLayer, self).__init__()
        self.attention = ModifiedBertAttention(layer.attention)
        self.intermediate = ModifiedBertIntermediate(layer.intermediate, gam=gam)
        self.output = ModifiedBertOutput(layer.output)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        hidden_states = self.output(intermediate_output, attention_output)

        return hidden_states


class ModifiedBertEncoder(nn.Module):
    def __init__(self, encoder, gam=0.15):
        super(ModifiedBertEncoder, self).__init__()
        layers = []
        for i, layer in enumerate(encoder.layer):
            layers.append(ModifiedBertLayer(layer, gam=gam))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states):
        for i, layer in enumerate(self.layer):
            hidden_states = layer(hidden_states)
        return hidden_states


class ModifiedBertPooler(nn.Module):
    def __init__(self, pooler, gam=0.15):
        super(ModifiedBertPooler, self).__init__()
        self.dense = ModifiedLinear(fc=pooler.dense, transform=None)
        self.activation = ModifiedAct(pooler.activation)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class ModifiedBertModel(nn.Module):
    def __init__(self, bert, gam=0.15, add_pooling_layer=True):
        super(ModifiedBertModel, self).__init__()
        self.encoder = ModifiedBertEncoder(bert.encoder, gam=gam)
        self.add_pooling_layer = add_pooling_layer
        if add_pooling_layer:
            self.pooler = ModifiedBertPooler(bert.pooler, gam=gam)

    def forward(self, x):
        hidden_states = self.encoder(x)
        if self.add_pooling_layer:
            hidden_states = self.pooler(hidden_states)

        return hidden_states


class ModifiedBertForSequenceClassification(nn.Module):
    def __init__(self, bert_classification, gam=0.15):
        super(ModifiedBertForSequenceClassification, self).__init__()
        self.bert = ModifiedBertModel(bert_classification.bert, gam=gam)
        self.classifier = ModifiedLinear(fc=bert_classification.classifier, transform=None)

    def forward(self, x):
        hidden_states = self.bert(x)
        hidden_states = self.classifier(hidden_states)

        return hidden_states
