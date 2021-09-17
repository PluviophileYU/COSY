import torch
import torch.nn as nn
import numpy as np
import copy
import math
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, KLDivLoss, MSELoss
from transformers.modeling_bert import (
    BertPreTrainedModel, BertModel, BertLayerNorm, BERT_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC, add_start_docstrings_to_callable, add_code_sample_docstrings)

class mBertForQuestionAnswering_dep_beta_v3(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dep_gan = Dependency_GAN(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.embedding_size = config.addtional_feature_size
        self.pos_embedding = nn.Embedding(18, self.embedding_size)
        self.qa_outputs_1 = nn.Linear(config.hidden_size+self.embedding_size, config.hidden_size)
        self.qa_outputs_2 = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

        for name, p in self.bert.embeddings.word_embeddings.named_parameters():
            p.requires_grad = False

        for i in range(4):
            for name, p in self.bert.encoder.layer[i].named_parameters():
                p.requires_grad = False

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased")
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        pos_label=None,
        dep_graph_coo=None,
        dep_graph_etype=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        gan_output, elmwise_hidden, dep_graphs, dep_masks = self.dep_gan(sequence_output, dep_graph_coo, dep_graph_etype)
        sequence_output_ = self.LayerNorm(sequence_output + gan_output)

        pos_feature = self.pos_embedding(pos_label)
        sequence_output_f = torch.cat([sequence_output_, pos_feature], dim=-1)

        factual_feature = self.qa_outputs_1(sequence_output_f)
        logits = self.qa_outputs_2(factual_feature)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            loss_fct = LabelSmoothingLoss(classes=start_logits.size(1), smoothing=0.2)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

            attention_mask_ = attention_mask.type_as(loss.data)
            # Counterfactual computation for first type
            gan_output_cf1 = self.dep_gan.counterfactual_1(elmwise_hidden, sequence_output, dep_graphs, dep_masks, attention_mask)
            sequence_output_cf1 = self.LayerNorm(sequence_output + gan_output_cf1)
            sequence_output_cf1 = torch.cat([sequence_output_cf1, pos_feature], dim=-1)
            cf1_features = self.qa_outputs_1(sequence_output_cf1)
            loss_cf1 = torch.mean(cf1_features * factual_feature * (attention_mask_.unsqueeze(-1)))

            # Counterfactual computation for second type
            pos_label_cf = self.create_counterfactual_2(pos_label, attention_mask)
            pos_feature_cf = self.pos_embedding(pos_label_cf)
            sequence_output_cf2 = torch.cat([sequence_output_, pos_feature_cf], dim=-1)
            cf2_features = self.qa_outputs_1(sequence_output_cf2)
            loss_cf2 = torch.mean(cf2_features * factual_feature * (attention_mask_.unsqueeze(-1)))

            outputs = (loss, loss_cf1, loss_cf2,) + outputs
        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

    def create_counterfactual_2(self, pos_label, attention_mask):
        with torch.no_grad():
            new_pos = (torch.rand(pos_label.size()) * 17).round().type_as(self.pos_embedding.weight.data)
            new_pos = new_pos * (attention_mask.type_as(self.pos_embedding.weight.data))
            new_pos = new_pos.type_as(pos_label.data)
        return new_pos

class Dependency_GAN(nn.Module):
    def __init__(self, config):
        super(Dependency_GAN, self).__init__()
        self.config = config

        # Embedding part
        self.embedding_size = config.addtional_feature_size
        self.etype_embedding = nn.Embedding(76, self.embedding_size, padding_idx=0)

        # Attention part
        self.query = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.key = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.attention = nn.Linear(int(config.hidden_size/2)+self.embedding_size, 1)

        self.dropout = nn.Dropout(config.gan_dropout_prob)

    def create_graph(self, hidden, dep_graph_coo, dep_graph_etype):
        '''create original graph with attention mask'''
        with torch.no_grad():
            batch_size, seq_len = hidden.size()[0], hidden.size()[1]
            dep_graphs = []
            for i in range(batch_size):
                dep_graph = torch.sparse.LongTensor(dep_graph_coo[i], dep_graph_etype[i], torch.Size([seq_len, seq_len])).to_dense().unsqueeze(0)
                dep_graphs.append(dep_graph)
            dep_graphs = torch.cat(dep_graphs, dim=0)
            dep_masks = (dep_graphs > 0).to(dtype=hidden.dtype)
            dep_masks = (1.0 - dep_masks) * -10000.0
        return dep_graphs, dep_masks

    def extract_elmwise_hidden(self, hidden):
        '''create element-wise hidden state, shape: (batch * seq * seq * h/2)'''
        batch_size, seq_len, hidden_size = hidden.size()[0], hidden.size()[1], hidden.size()[2]
        shape = (batch_size, seq_len, seq_len, int(hidden_size / 2))
        hidden_1 = self.query(hidden).unsqueeze(2)  # batch * seq * 1 * hidden/2
        hidden_1 = hidden_1.expand(shape)
        hidden_2 = self.key(hidden).unsqueeze(1)  # batch * 1 * seq * hidden/2
        hidden_2 = hidden_2.expand(shape)
        elmwise_hidden = torch.mul(hidden_1, hidden_2)  # batch * seq * seq * hidden/2
        return elmwise_hidden

    def compute_attention_score(self, elmwise_hidden, edge_features):
        '''Compute attention score for Graph Attention Network'''
        cat_hidden = torch.cat([elmwise_hidden, edge_features], dim=-1)
        cat_hidden = self.dropout(cat_hidden)
        output = self.attention(cat_hidden).squeeze(-1)
        return output

    def forward(self, hidden, dep_graph_coo, dep_graph_etype):
        dep_graphs, dep_masks = self.create_graph(hidden, dep_graph_coo, dep_graph_etype)
        edge_feature = self.etype_embedding(dep_graphs)

        elmwise_hidden = self.extract_elmwise_hidden(hidden)
        attention_scores = self.compute_attention_score(elmwise_hidden, edge_feature)
        attention_scores = attention_scores + dep_masks

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, hidden)
        return output, elmwise_hidden, dep_graphs, dep_masks

    def counterfactual_1(self, elmwise_hidden, hidden, dep_graphs, dep_masks, attention_mask):
        '''Create counterfactual for edge type
           Here ONLY the dependency relation type is changed
        '''
        with torch.no_grad():
            # create counterfactual graph
            batch_size, seq_len = dep_graphs.size()[0], dep_graphs.size()[1]
            new_graph = (torch.rand(dep_graphs.size()) * 75).round()
            for i in range(batch_size):
                new_graph[i].fill_diagonal_(1)  # maintain the self-loop property, all ROOT relations are also turned to self-loop
            new_graph = new_graph.type_as(dep_graphs.data)
        edge_feature = self.etype_embedding(new_graph)
        attention_scores = self.compute_attention_score(elmwise_hidden, edge_feature)
        attention_scores = attention_scores + dep_masks

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        output = torch.matmul(attention_probs, hidden)
        return output

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))