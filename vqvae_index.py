#%%
import os, sys
import pandas as pd
import argparse
from func_scripts import file2dict, list2file
import random

#%%
parser = argparse.ArgumentParser(description = "VQ-VAE");
parser.add_argument('--feat_dir', type=str, default='./data', dest='data_dir')
parser.add_argument('--save_dir', type=str, default='./index' , dest='save_dir')
args = parser.parse_args()

data_dir=args.data_dir
index_dir=args.save_dir

if not os.path.exists(index_dir):
    os.mkdir(index_dir)

#%%
#train test split
feat=file2dict(os.path.join(data_dir, 'feat.scp'))
ratio=0.8

random.seed(5)
utt_list=list(feat.keys())
random.shuffle(utt_list)

train_utt=utt_list[0:int(len(utt_list)*ratio)]
test_utt=utt_list[int(len(utt_list)*ratio):]

train=[]
test=[]
for utt in train_utt:
    train.append([utt, feat[utt]])
for utt in test_utt:
    test.append([utt, feat[utt]])

list2file(train, os.path.join(index_dir, 'train.index'))
list2file(test, os.path.join(index_dir, 'test.index'))
