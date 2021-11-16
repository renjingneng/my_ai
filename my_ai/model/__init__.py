import torch
import torch.nn
import numpy
from typing import Union


class ElasticEmbedding(torch.nn.Module):
    """ok
    """
    def __init__(self, pretrained_embedding: Union[torch.Tensor, numpy.ndarray], residual_index: list[int]):
        super(ElasticEmbedding, self).__init__()
        if isinstance(pretrained_embedding, numpy.ndarray):
            self.pretrained_embedding = torch.from_numpy(pretrained_embedding)
        else:
            self.pretrained_embedding = pretrained_embedding
        residual_index = torch.tensor(residual_index)

        self.residual_map = {}
        for i in range(residual_index.size(0)):
            index_value = residual_index[i].item()
            self.residual_map[index_value] = i
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
