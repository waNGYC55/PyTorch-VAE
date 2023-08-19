#%%
import os
import random
import numpy as np
import math
import torch
import soundfile as sf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import time

#%%
class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, num_workers=10):
        self.bs = batch_size
        self.n_samples = len(dataset)
        self.num_workers = num_workers
        super(self.__class__, self).__init__(batch_sampler=self.batch_gen(), num_workers=self.num_workers, dataset=dataset)

    def __len__(self):
        return math.floor(self.n_samples / self.bs)

    def batch_gen(self):
        for i in range(math.floor(self.n_samples / self.bs)):
            ind = np.arange(self.bs).reshape(self.bs, 1) + i * self.bs
            yield ind

class VQVAE_Dataset(Dataset):
    def __init__(self, utt2data, sequence_size=22000*2, need_aug=False, shuffle=True, with_output=True):
        self.utt2data = utt2data
        self.dataset_size = len(self.utt2data)
        self.shuffle = shuffle
        self.need_aug = need_aug
        self.sequence_size = sequence_size
        self.with_output=with_output

        if shuffle:
            random.shuffle(self.utt2data)

    def __len__(self):
        return self.dataset_size

    def read_feat(self, filename):

        samples = self.sequence_size
        try:
            feat, sr = torchaudio.load(filename)
            if feat.shape[1] == samples:
                new_feat = feat
            elif feat.shape[1] > samples:
                start_point = random.randrange(0, feat.shape[1] - samples)
                new_feat = np.array(feat[:,start_point:start_point + samples])
                #print(new_feat.shape)
            else:
                new_feat = np.zeros((1, samples))
                pad_beg = int((samples - feat.shape[1])/ 2)
                new_feat[:,pad_beg:pad_beg + feat.shape[1]] = feat
                #print(new_feat.shape)
            assert new_feat.shape[1] == samples
            return new_feat                 
        except:
            print(filename)

        return 1      

    def __getitem__(self, sample_idx):
        idx = int(sample_idx[0])
        #idx = sample_idx
        assert 0 <= idx and idx < self.dataset_size, "invalid index"
        
        utt, filename = self.utt2data[idx]
        feat = self.read_feat(filename)
        feat = feat.astype('float32')
        #print(feat.shape)
        if self.with_output:
            return utt, feat, feat
        else:
            return utt, feat
        
class VQVAE_Mel_Dataset(VQVAE_Dataset):
    
    def __init__(self, utt2data, sequence_size=int(256), need_aug=False, shuffle=True, with_output=True):
        super().__init__(utt2data, sequence_size, need_aug, shuffle, with_output)
    
    def read_feat(self, filename):
        samples = self.sequence_size
        try:
            feat=np.load(filename)
            if feat.shape[0] == samples:
                new_feat = feat
            elif feat.shape[0] > samples:
                start_point = random.randrange(0, feat.shape[0] - samples)
                new_feat = np.array(feat[start_point:start_point + samples,:,:])
                #print(new_feat.shape)
            else:
                new_feat = np.zeros((samples, feat.shape[1], feat.shape[2]))
                pad_beg = int((samples - feat.shape[0])/ 2)
                new_feat[pad_beg:pad_beg + feat.shape[0],:,:] = feat
                #print(new_feat.shape)
            assert new_feat.shape[0] == samples
            return torch.tensor(new_feat, dtype=torch.float).squeeze(-1).permute(1,0)    
        except:
            print(filename)
        return 1
    
    def __getitem__(self, sample_idx):
        idx = int(sample_idx[0])
        #idx = sample_idx
        assert 0 <= idx and idx < self.dataset_size, "invalid index"
        
        utt, filename = self.utt2data[idx]
        feat = self.read_feat(filename)
        #print(feat.shape)
        if self.with_output:
            return utt, feat, feat
        else:
            return utt, feat
    
#%%
##############################
def test():
    batch_size = 1
    num_workers = 1
    index_root_dir = './data'
    utt2wav = [line.split() for line in open(os.path.join(index_root_dir, 'feat.scp'))]

   
    dev_dataset = VQVAE_Mel_Dataset(utt2wav, need_aug=False, with_output=True, shuffle=False)
    dear_dev_dataloader = MyDataLoader(dev_dataset, batch_size=batch_size, num_workers=1)

    print("Data loader prepared...")
    for utt, feat_in, feat_out in dear_dev_dataloader:
        # print(utt)
        print(feat_in.shape)
        time.sleep(1)
if __name__ == "__main__":
    test()
# %%



