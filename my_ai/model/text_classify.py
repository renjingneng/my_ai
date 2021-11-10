import torch
import torch.nn as nn
import torch.nn.functional as F

import my_ai.pipeline
import my_ai.model


class TextCNN(nn.Module):
    def __init__(self, config: my_ai.pipeline.TextClassifyConfig):
        super(TextCNN, self).__init__()
        if config.is_pretrained == 1:
            pretrained_embedding = torch.tensor(config.embedding.get_all_representation())
            residual_index = torch.tensor(config.embedding.get_residual_index())
            self.embedding = my_ai.model.ElasticEmbedding(pretrained_embedding, residual_index)
        else:
            self.embedding = nn.Embedding(config.vocab.get_len(), config.embedding_length)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.other_params['num_filters'], (k, config.embedding.len)) for k in
             config.other_params['filter_sizes']])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.other_params['num_filters'] * len(config.other_params['filter_sizes']),
                            config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
