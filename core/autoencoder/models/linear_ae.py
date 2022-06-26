# -*- coding: utf-8 -*-"
"""
Created on 05/27/2021  10:43 AM


@author: Zhuo
"""
from torch import nn


class Linear_AE(nn.Module):
    """ Linear AE """

    def __init__(self, input_size=64*64, out_features=32):
        super(Linear_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, out_features * 16),
            nn.ReLU(True),
            nn.Linear(out_features * 16, out_features * 4),
            nn.ReLU(True),
            nn.Linear(out_features * 4, out_features),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_features, out_features * 4),
            nn.ReLU(True),
            nn.Linear(out_features * 4, out_features * 16),
            nn.ReLU(True),
            nn.Linear(out_features * 16, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def hook(self, x):
        x = self.encoder(x)
        code = x.detach()
        return code


# class Linear_AE(nn.Module):
#     """ Linear AE """
# 
#     def __init__(self, input_size=64*64, out_features=32):
#         super(Linear_AE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, out_features * 8),
#             nn.ReLU(True),
#             nn.Linear(out_features * 8, out_features * 4),
#             nn.ReLU(True),
#             nn.Linear(out_features * 4, out_features * 2),
#             nn.ReLU(True),
#             nn.Linear(out_features * 2, out_features),
#             nn.ReLU(True),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(out_features, out_features * 2),
#             nn.ReLU(True),
#             nn.Linear(out_features * 2, out_features * 4),
#             nn.ReLU(True),
#             nn.Linear(out_features * 4, out_features * 8),
#             nn.ReLU(True),
#             nn.Linear(out_features * 8, input_size),
#             nn.Sigmoid()
#         )
# 
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
# 
#     def hook(self, x):
#         x = self.encoder(x)
#         code = x.detach()
#         return code
