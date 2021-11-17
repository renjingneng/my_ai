import torch
import torch.nn
import torch.nn.functional

import my_ai.pipeline
import my_ai.model


class TextCNN(torch.nn.Module):
    """ok
    """

    def __init__(self, config: my_ai.pipeline.TextClassifyConfig):
        super(TextCNN, self).__init__()
        num_filters = config.other_params['num_filters']
        filter_sizes = config.other_params['filter_sizes']

        if config.is_pretrained:
            pretrained_embedding = config.embedding.get_all_representation()
            residual_index = config.embedding.get_residual_index()
            self.embedding = my_ai.model.ElasticEmbedding(pretrained_embedding, residual_index)
        else:
            self.embedding = torch.nn.Embedding(config.vocab.get_len(), config.embedding_length)

        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, num_filters, (k, config.embedding_length)) for k in
             filter_sizes])

        self.max_pool_1d_list = [torch.nn.MaxPool1d((config.text_length-k-1)) for k in filter_sizes]
        self.dropout = torch.nn.Dropout(config.dropout)

        self.fc = torch.nn.Linear(num_filters * len(filter_sizes),
                                  config.num_classes)

    def forward(self, x):
        # [batch_size, seq_len] => [batch_size, seq_len, embed_size]
        out = self.embedding(x)
        # [batch_size, seq_len, embed_size] => [batch_size, 1, seq_len, embed_size]
        out = out.unsqueeze(1)
        # [batch_size, 1, seq_len, embed_size] => [batch_size, 100, seq_len-1 , 1] => [batch_size, 100, seq_len-1]
        # [batch_size, 1, seq_len, embed_size] => [batch_size, 100, seq_len-2 , 1] => [batch_size, 100, seq_len-2]
        # ...
        list1 = [torch.nn.functional.relu(conv(out)).squeeze(3) for conv in self.convs]
        # [batch_size, 100, seq_len-1] =>[batch_size, 100, 1] => [batch_size, 100]
        # [batch_size, 100, seq_len-2] =>[batch_size, 100, 1] => [batch_size, 100]
        # ...
        list2 = [self.max_pool_1d_list[idx](item1).squeeze(2) for idx, item1 in enumerate(list1)]
        # all => [batch_size, 300]
        pack = torch.cat(list2, 1)
        pack = self.dropout(pack)
        # [batch_size, 300]=>[batch_size, num_class]
        result = self.fc(pack)
        return result
