#%%
import os,sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
import math
from tqdm import tqdm
from vqvae_dataloader import VQVAE_Mel_Dataset, MyDataLoader
from vqvae_model import VQVAE
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

#%%
parser = argparse.ArgumentParser(description = "VQ-VAE");
parser.add_argument('--index_dir', type=str, default='./index', dest='index_dir')
parser.add_argument('--save_dir', type=str, default='./outputs/samples' , dest='save_dir')
parser.add_argument('--model_path', type=str, default='./outputs/vqvae_batchnorm_/models/model_100' , dest='model_path')
parser.add_argument('--batch_size', type=int, default=1 , dest='batch_size')
parser.add_argument('--nj', type=int, default=0 , dest='nj')
args = parser.parse_args();

#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.is_available()
#model hparameters
in_channels=80 # Mel
num_embeddings=8196 
embedding_dim=256
K = num_embeddings
D = embedding_dim
hidden_dims=[128,256,256]
beta = 0.25
res_depth=6

#sample setup
# BATCH_SIZE = args.batch_size
# NUM_WORKERS= args.nj
# data_dir = args.index_dir
# save_dir=args.save_dir
# model_path=args.model_path

BATCH_SIZE = 1
NUM_WORKERS= 0
data_dir = './index'
save_dir='./outputs'
model_path='./outputs/vqvae_batchnorm_/models/model_100'

SAVE_PATH = os.path.join(save_dir, "samples")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

dev_utt2wav = [line.split() for line in open(os.path.join(data_dir, 'test.index'))]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_model(modelfile, net):
    state = torch.load(modelfile)
    net.load_state_dict(state['state_dict'])
    return net

def loss_function(*args,
                    **kwargs) -> dict:
    """
    :param args:
    :param kwargs:
    :return:
    """
    recons = args[0]
    input = args[1]
    vq_loss = args[2]

    recons_loss = F.mse_loss(recons, input)

    loss = recons_loss + vq_loss
    return {'loss': loss,
            'Reconstruction_Loss': recons_loss,
            'VQ_Loss':vq_loss}

def batch_handle(feats, device='cuda'):
    if device=='cuda':
        input_x = feats.float().cuda()
    else:
        input_x=feats.float().cpu()
    return input_x

def plot_spectrogram(spectrogram, save_path=None):
    fig, ax= plt.subplots(figsize=(10,2))
    im=ax.imshow(spectrogram, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    if not save_path==None:
        plt.savefig(save_path)
    plt.close()
    return fig
#%%
#nn model
net=VQVAE(in_channels=in_channels, embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, res_depth=res_depth, beta=0.25)
net=load_model(model_path, net)
print(net)
net = net.cuda()
status=net.eval()
#%%
#data loader
vqvae_dataset = VQVAE_Mel_Dataset(dev_utt2wav, need_aug=False, with_output=False, shuffle=False)
vqvae_dataloader = MyDataLoader(vqvae_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

#%%
print('Sampling...')

overall_losses_avg = AverageMeter()
recon_losses_avg = AverageMeter()
vq_losses_avg = AverageMeter()
metadata=['utt', 'init_spec_path', 'recon_spec_path', 'overall_loss', 'recon_loss', 'vq_loss']

with torch.no_grad():
    for batch_utt, batch_x in tqdm(vqvae_dataloader, total=len(vqvae_dataloader)):
        batch_x = batch_handle(batch_x)
        #print(batch_x.shape)
        outputs, init_x, vq_loss= net(batch_x)
        losses = loss_function(outputs, init_x, vq_loss)

        overall_loss=losses['loss'].cpu()
        recon_loss=losses['Reconstruction_Loss'].cpu()
        vq_loss=losses['VQ_Loss'].cpu()
        #update avg loss
        overall_losses_avg.update(overall_loss.data, batch_x.size()[0])
        recon_losses_avg.update(recon_loss.data, batch_x.size()[0])
        vq_losses_avg.update(vq_loss.data, batch_x.size()[0])
        
        #draw spectrum and save reconstructed sample
        spec_save_path=os.path.join(SAVE_PATH, 'spec')
        mel_save_path=os.path.join(SAVE_PATH, 'mel')
        if not os.path.exists(spec_save_path) or not os.path.exists(mel_save_path):
            os.makedirs(spec_save_path)
            os.makedirs(mel_save_path)
        for i in range(0,len(batch_x)):
            utt=batch_utt[i].replace('.wav', '')
            init_mel=batch_x[i].cpu()
            recon_mel=outputs[i].cpu()
            init_spec_path=os.path.join(spec_save_path, utt+'.png')
            recon_spec_path=os.path.join(spec_save_path, utt+'_recon.png')
            np.save(os.path.join(spec_save_path, utt), recon_mel)
            fig=plot_spectrogram(init_mel, init_spec_path)
            fig=plot_spectrogram(recon_mel, recon_spec_path)
            metadata.append([utt, init_spec_path, recon_spec_path, overall_loss, recon_loss, vq_loss])

## test code
# batch_gen=iter(vqvae_dataloader)
# batch_utt, batch_x=next(batch_gen)
# batch_x = batch_handle(batch_x)
with open(os.path.join(SAVE_PATH, 'metadata.scp'), 'w') as f:
    for line in metadata:
        f.write(' '.join(line)+'\n')
