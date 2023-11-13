import torch
import torch.nn as nn
from resnet1d.net1d import Net1D

embedding_size = 64
project_size = 16

class Projector(nn.Module):
    def __init__(self, embedding_size=embedding_size) -> None:
        super().__init__()
        
        self.linear = nn.Linear(embedding_size, embedding_size)
        self.norm = nn.LayerNorm([embedding_size, ])
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(embedding_size, embedding_size)
    
    def forward(self, x):
        x1 = self.linear(x)
        x1 = self.norm(x1)
        x1 = self.act(x1)
        out = self.linear2(x1) + x

        return out

class Extractor(nn.Module):
    def __init__(self, 
                n_features,
                n_channels=3, 
                embedding_size=64,
                filter_list=[64, 64, 64, 64], 
                block_list=[3, 4, 6, 3]) -> None:
        super().__init__()

        self.resnet = Net1D(
            in_channels=n_channels,
            n_classes=embedding_size,
            base_filters=n_features,
            filter_list=filter_list,
            m_blocks_list=block_list,
            kernel_size=16,
            stride=2,
            
            # not sure what they does
            ratio=1.0,
            groups_width=16, 

            verbose=False,
        )
    
    def forward(self, x):
        return self.resnet(x)

