#data preparation script
#prepare opensource datasets:
#LJSpeech
#output in: ./data/wav.scp
#   utt    wavpath
#%%
import os, sys
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description = "VQ-VAE");
parser.add_argument('--LJSpeech_dir', type=str, default='C:/Users/YC/Desktop/TTS/data/LJSpeech-1.1', dest='LJSpeech_dir')
parser.add_argument('--save_dir', type=str, default='./data' , dest='save_dir')
args = parser.parse_args();

LJSpeech_dir=args.LJSpeech_dir
save_dir=args.save_dir

#%%
#prepare LJSpeech
wav_dir=os.path.join(LJSpeech_dir, 'wavs')
wav_list=os.listdir(wav_dir)
wavscp=[]

for i in wav_list:
    wavscp.append([i, os.path.join(wav_dir, i)])

save_dir=r'./data'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

with open(os.path.join(save_dir, 'wav.scp'), 'w') as f:
    for line in wavscp:
        f.write(' '.join(line)+'\n')
