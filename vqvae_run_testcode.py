#%%
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
import torchaudio
from vqvae_dataloader import MyDataLoader, VQVAE_Dataset, VQVAE_Mel_Dataset
import os,sys
from vqvae_model import VQVAE
import numpy as np
#from .types_ import *

#torch.cuda.is_available()
# %%
num_embeddings=256
embedding_dim=8192
K = num_embeddings
D = embedding_dim
beta = 0.25
embedding = nn.Embedding(K, D)

hidden_dims=[512, 2048, 8192]

in_channels=80

## usecase
# input=torch.tensor(1)
# embedding(input)

## re-init embeddings
#embedding.weight.data.uniform_(-1 / K, 1 / K)

#%%
batch_size = 2
num_workers = 1
index_root_dir = './data'
utt2wav = [line.split() for line in open(os.path.join(index_root_dir, 'feat.scp'))]

vqvae_dataset = VQVAE_Mel_Dataset(utt2wav, need_aug=False, with_output=True, shuffle=False)
vqvae_dataloader = MyDataLoader(vqvae_dataset, batch_size=batch_size, num_workers=1)

# %%
utt, x, y=vqvae_dataset.__getitem__([0])
# %%
in_x=torch.stack([x,x], dim=0)

# # %%
# from vqvae_model import ResidualLayer

# modules=[]
# # Build Encoder
# for h_dim in hidden_dims:
#     modules.append(
#         nn.Sequential(
#             nn.Conv1d(in_channels, out_channels=h_dim,
#                         kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU())
#     )

#     in_channels = h_dim

# modules.append(
#     nn.Sequential(
#         nn.Conv1d(in_channels, in_channels,
#                     kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU())
# )

# for _ in range(6):
#     modules.append(ResidualLayer(in_channels, in_channels))
# modules.append(nn.LeakyReLU())

# encoder = nn.Sequential(*modules)
# encoder_out=encoder(in_x)

# from vqvae_model import VectorQuantizer

# vq_model=VectorQuantizer(num_embeddings, embedding_dim)
# vq_out=vq_model(encoder_out)
# vq_latents=vq_out[0]
# vq_loss=vq_out[1]

# # Build Decoder
# decoder_modules = []
# decoder_modules.append(
#     nn.Sequential(
#         nn.Conv1d(embedding_dim,
#                     hidden_dims[-1],
#                     kernel_size=3,
#                     stride=1,
#                     padding=1),
#         nn.LeakyReLU())
# )

# for _ in range(6):
#     decoder_modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

# decoder_modules.append(nn.LeakyReLU())

# hidden_dims.reverse()

# for i in range(len(hidden_dims) - 1):
#     decoder_modules.append(
#         nn.Sequential(
#             nn.ConvTranspose1d(hidden_dims[i],
#                                 hidden_dims[i + 1],
#                                 kernel_size=4,
#                                 stride=2,
#                                 padding=1),
#             nn.LeakyReLU())
#     )

# decoder_modules.append(
#     nn.Sequential(
#         nn.ConvTranspose1d(hidden_dims[-1],
#                             out_channels=80, #out_channel for mel
#                             kernel_size=4,
#                             stride=2, padding=1),
#         nn.Tanh()))

# decoder = nn.Sequential(*decoder_modules)

# decoder_out=decoder(vq_latents)

# recons_loss = F.mse_loss(in_x, decoder_out)

# loss = recons_loss + vq_loss
# %%
from vqvae_model import VQVAE

vqvae=VQVAE(in_channels=in_channels, embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, beta=0.25)

# %%
out=vqvae(in_x)
# %%
