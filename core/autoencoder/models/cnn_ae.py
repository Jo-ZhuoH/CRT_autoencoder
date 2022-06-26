# -*- coding: utf-8 -*-"
"""
Created on 05/27/2021  10:44 AM


@author: Zhuo
"""
from torch import nn

from .types_ import *


class CNN_AE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(CNN_AE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3)
                )
            )

# class CNN_AE(nn.Module):
#     """ CNN autoencoder """
#
#     def __init__(self):
#         super(CNN_AE, self).__init__()
#         # self.features = features
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 8, 3, stride=1, padding=1),  # d: 8x64x64
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # d: 8x32x32
#             nn.Conv2d(8, 16, 3, stride=1, padding=1),  # d: 16x32x32
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # d: 16x16x16
#             nn.Conv2d(16, 32, 3, stride=1, padding=1),  # d: 32x16x16
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2)  # d: 32x8x8
#         )
#         self.conv = nn.Sequential(
#             nn.Conv2d(32, 32, 3, stride=1, padding=1)  # d: 32x8x8
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
#                                output_padding=1),  # d: 16x16x16
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1,
#                                output_padding=1),  # d: 8x31x31
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1,
#                                output_padding=1),  # d: 1x64x64
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.conv(x)
#         self.features = x.detach()
#         x = self.decoder(x)
#         return x