#%%
import os,sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datetime
from tqdm import tqdm
from vqvae_dataloader import VQVAE_Mel_Dataset, MyDataLoader
from vqvae_model import VQVAE

import argparse

#%%
parser = argparse.ArgumentParser(description = "VQ-VAE");
parser.add_argument('--index_dir', type=str, default='./index', dest='index_dir')
parser.add_argument('--save_dir', type=str, default='./outputs' , dest='save_dir')
parser.add_argument('--num_epoch', type=int, default=50 , dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=1 , dest='batch_size')
parser.add_argument('--resnet_depth', type=int, default=6 , dest='resnet_depth')
parser.add_argument('--nj', type=int, default=1 , dest='nj')
args = parser.parse_args();

#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.is_available()
#model hparameters
in_channels=80 # Mel
num_embeddings=256 
embedding_dim=8192
K = num_embeddings
D = embedding_dim
hidden_dims=[512, 2048, 8192]
res_depth=args.resnet_depth
# res_depth=1
beta = 0.25

#train hparameters
EPOCH_NUM = args.num_epoch
BATCH_SIZE = args.batch_size
NUM_WORKERS= args.nj
LEARNING_RATE = 0.01
RESUME_MODEL_NAME = None
RESUME_LR = 0.005

# EPOCH_NUM = 50
# BATCH_SIZE = 1
# NUM_WORKERS= 1
# LEARNING_RATE = 0.01
# RESUME_MODEL_NAME = None
# RESUME_LR = 0.005

#file io variables
data_dir = args.index_dir
save_dir=args.save_dir
# data_dir ='./index'
# save_dir='./outputs'
NAME_SPACE = '_'.join(['vqvae','Mel'+str(in_channels),'K'+str(K),'D'+str(D)])
SAVE_PATH = os.path.join(save_dir, "models")
LOG_PATH = os.path.join(save_dir, "log")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    os.makedirs(LOG_PATH)
LOG_FILE = LOG_PATH + "/" + NAME_SPACE + ".log"
f_log = open(LOG_FILE, "wt")

train_utt2wav = [line.split() for line in open(os.path.join(data_dir, 'train.index'))]
dev_utt2wav = [line.split() for line in open(os.path.join(data_dir, 'dev.index'))]

#reproductivity
seed=5
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
# 
#%%
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

def saveModel(epoch, temp=""):
    global net
    global optimizer

    now = datetime.datetime.now()
    time_str = now.strftime('%Y_%m_%d_%H')
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, SAVE_PATH + "/" +  "model_" + str(epoch))

def batch_handle(feats, device='cuda'):
    if device=='cuda':
        input_x = feats.float().cuda()
    else:
        input_x=feats.float().cpu()
    return input_x

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

#%%
def train(epc=1):

    #initialization
    global net
    global optimizer

    #data loader
    vqvae_dataset = VQVAE_Mel_Dataset(train_utt2wav, need_aug=False, with_output=False, shuffle=False)
    vqvae_dataloader = MyDataLoader(vqvae_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    print('Training.....')
    status=net.train()

    losses_avg = AverageMeter()
    
    iteration_num = 0

    for batch_utt, batch_x in tqdm(vqvae_dataloader, total=len(vqvae_dataloader)):
        iteration_num += 1
        batch_x = batch_handle(batch_x)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs, init_x, vq_loss = net(batch_x)
        losses = loss_function(outputs, init_x, vq_loss)
        overall_loss=losses['loss']
        recon_loss=losses['Reconstruction_Loss']
        vq_loss=losses['VQ_Loss']
        overall_loss.backward()
        optimizer.step()

        #update avg loss
        losses_avg.update(overall_loss.data, batch_x.size()[0])

        if iteration_num % 30 == 29: 
            curr_log = '[%d, %5d] recon_loss: %.3f, vq_loss: %.3f. \n' % (epc, iteration_num + 1, recon_loss, vq_loss)
            tqdm.write(curr_log)
            f_log.write(curr_log)
    return losses_avg.avg

    ## test code
    # batch_utt, batch_x=vqvae_dataset.__getitem__([0])
    # batch_x=batch_x.unsqueeze(0)
    # batch_x = batch_handle(batch_x)

# %%
def validate(epc):
    global net

    print('Validate.....')
    dev_dataset=VQVAE_Mel_Dataset(dev_utt2wav, need_aug=False, with_output=False, shuffle=False)
    dev_dataloader = MyDataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    status=net.eval()

    overall_losses=[]
    recon_losses=[]
    vq_losses=[]

    with torch.no_grad():
        for batch_utt, batch_x in tqdm(dev_dataloader, total=len(dev_dataloader)):
            batch_x = batch_handle(batch_x)
            outputs, init_x, vq_loss= net(batch_x)
            losses = loss_function(outputs, init_x, vq_loss)

            overall_loss=losses['loss'].cpu()
            recon_loss=losses['Reconstruction_Loss'].cpu()
            vq_loss=losses['VQ_Loss'].cpu()

            overall_losses.append(overall_loss)
            recon_losses.append(recon_loss)
            vq_losses.append(vq_loss)
    
    overall_losses=np.concatenate(overall_losses)
    recon_losses=np.concatenate(recon_losses)
    vq_losses=np.concatenate(vq_losses)
    cur_log='Validate: (Average) overall Loss: %.3f, recon_loss: %.3f, vq_loss: %.3f. \n'% (np.mean(overall_losses), np.mean(recon_losses), np.mean(vq_losses))
    tqdm.write(cur_log)
    f_log.write(cur_log)
    return np.mean(overall_losses)

# %%
#nn network
net=VQVAE(in_channels=in_channels, embedding_dim=embedding_dim, num_embeddings=num_embeddings, hidden_dims=hidden_dims, res_depth=res_depth, beta=0.25)
print(net)

net = nn.DataParallel(net)
net = net.cuda()

optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, min_lr=1e-4)


#%%
def main(epc = 1):

    for epoch in range(epc, EPOCH_NUM + 1):
        print("Current running model [ " + NAME_SPACE + " ]")
        loss_avg = train(epoch)
        scheduler.step(loss_avg)
        if epoch in [1, 3, 5] or epoch % 5 == 0: # epoch in [1, 3, 5, 10, 20, 30, 40, 50]:
            cur_val_losses_avg = validate(epoch)
            saveModel(epoch)
    f_log.close()

#%%

if __name__ == '__main__':
    main()
