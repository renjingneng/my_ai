import torch
from torch import nn


class ElasticEmbedding(nn.Module):
    def __init__(self,pretrained_embedding,residual_index):
        super(ElasticEmbedding, self).__init__()
        self.map = {}
        for no,index_value in enumerate(residual_index):
            self.map[no] = index_value
        residual_index_tensor = torch.tensor(residual_index)
        self.pretrained_embedding = torch.tensor(pretrained_embedding)
        self.residual_embedding = torch.nn.Parameter(self.pretrained_embedding.index_select(0,residual_index_tensor))


    def forward(self, x):
        result = ''
        return result
