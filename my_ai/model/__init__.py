import torch
import torch.nn


class ElasticEmbedding(torch.nn.Module):
    def __init__(self, pretrained_embedding, residual_index):
        super(ElasticEmbedding, self).__init__()
        self.residual_map = {}
        for i in range(residual_index.size(0)):
            index_value = residual_index[i].item()
            self.residual_map[index_value] = i
        self.pretrained_embedding = pretrained_embedding
        self.residual_embedding = torch.nn.Parameter(self.pretrained_embedding.index_select(0, residual_index))

    def forward(self, x):
        y = torch.zeros(size=(x.size(0), x.size(1), self.pretrained_embedding.size(1)))
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                residual_key = self.residual_map.get(x[i][j].item())
                if residual_key is None:
                    y[i][j] = self.pretrained_embedding[x[i][j]]
                else:
                    y[i][j] = self.residual_embedding[residual_key]
        return y
