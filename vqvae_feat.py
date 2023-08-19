#%%
import os, sys
import torch
import torchaudio
from func_scripts import file2list
import numpy as np
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description = "VQ-VAE");
parser.add_argument('--data_dir', type=str, default='./data', dest='data_dir')
parser.add_argument('--save_dir', type=str, default='./feats' , dest='save_dir')
args = parser.parse_args();

data_dir=args.data_dir
save_dir=args.save_dir

#%%
#prepare features
sr=16000
win_size=800
step_size=160
n_mels=80
n_fft=1024

FeatMel=torchaudio.transforms.MelSpectrogram(sample_rate=sr,n_fft=n_fft, win_length=win_size, hop_length=step_size, n_mels=n_mels)


# %%
wavscp=file2list(os.path.join(data_dir,'wav.scp'))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#%%
featscp=[]
for utt, wavpath in tqdm(wavscp):
    sig, raw_sr=torchaudio.load(wavpath)
    sig = torchaudio.functional.resample(sig, orig_freq=raw_sr, new_freq=sr)
    feat=FeatMel(sig)

    feats_save_path=os.path.join(save_dir, utt.replace('.wav','.npy'))

    if not os.path.exists(feats_save_path):
        feats = FeatMel(sig)
        feats = np.array(feats).T
        np.save(feats_save_path, feats)
    
    featscp.append([utt, feats_save_path])
# %%
with open(os.path.join(data_dir, 'feat.scp'), 'w') as f:
    for line in featscp:
        f.write(' '.join(line)+'\n')

# %%
