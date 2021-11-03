import os, pickle, time
import numpy as np
from scipy.sparse import load_npz
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from task1_dataset import Task1Dataset
from load_data import load_data, concat_data, generate_label_mask
from cct_model import CCTbert
import argparse
from layers import Balanced_AsymmetricLoss
from scheduler import WarmupLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_util import pos_class_weight

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--load_model', default=False, action='store_true', help='load trained model')
parser.add_argument('--test', default=False, action='store_true', help='model testing')
parser.add_argument('--signal_length', type=int, default=10)
args = parser.parse_args()


def find_gapped(ref_matrix):
    # find gapped region including 'N'
    temp_sum = ref_matrix.sum(axis=1).sum(axis=1)
    assert temp_sum.shape[0] == ref_matrix.shape[0]
    gapped_idx = np.where(temp_sum != ref_matrix.shape[2])[0]
    return gapped_idx


cls = ['A549', 'H1','K562']
 # 'MCF-7', 'GM12878', 'HepG2', 'HeLa-S3'
lengths = {'DNA': 1800, 'HM': 11, 'TF': 257}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = CCTbert(
    kmers=6, dim=320, lengths=lengths,
    signal_length=args.signal_length, depth=6, heads=4,
    dim_head=128, mlp_dim=320, dropout=0.2
)
model.to(device)

pos_ratio = np.load('/scratch/drjieliu_root/drjieliu/zhenhaoz/pos_class_weight_sqrt.npy')
# asymmetric loss
# criterion = Balanced_AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05, alpha=torch.FloatTensor(pos_ratio).to(device))

# normal bcelosswithlogits
criterion = Balanced_AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
# learning rate decay
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)
# warmup
scheduler = WarmupLR(scheduler, init_lr=1e-5, num_warmup=3, warmup_strategy='cos')
optimizer.zero_grad()
optimizer.step()

def shuffle_data():
    chrs = np.arange(1, 6)
    temp = np.tile(chrs, (len(cls), 1))
    for i in range(len(cls)):
        np.random.shuffle(temp[i, :])
    return temp


def prepare_date(chr_data, ref_genomes, dnase_data, labels, chip_data, label_masks):
    input_seqs = np.concatenate(
        (np.concatenate([ref_genomes[chr] for chr in chr_data], axis=0),
         np.expand_dims(concat_data(dnase_data, chr_data), axis=1))
        , 1)
    print(input_seqs.shape)
    target_label = concat_data(labels, chr_data, sparse=True)
    input_chip = torch.FloatTensor(concat_data(chip_data, chr_data))
    input_label_mask = generate_label_mask(label_masks, [ref_genomes[chr].shape[0] for chr in chr_data])
    return input_seqs, target_label, input_chip, input_label_mask

# load all the training data to memory
ref_genomes, dnase_data, chip_data, labels = load_data()
# miss chip-seq in each cell line
with open('/scratch/drjieliu_root/drjieliu/zhenhaoz/label_mask.pickle', 'rb') as f:
    label_masks = pickle.load(f)

best_loss = 1e10
scheduler.step(best_loss)

for epoch in range(args.epochs):
    iterations = shuffle_data()
    training_losses = []
    validation_losses = []
    valid_mask_los=[[],[],[],[],[]]
    for iter in range(iterations.shape[1]):
        if iter < 4:
            model.train()
            mode = 'train'
        else:
            model.eval()
            mode = 'validation'
        chr_data = iterations[:, iter].tolist()
        print(chr_data)
        input_seqs, target_label, input_chip, input_label_mask=\
            prepare_date(chr_data, ref_genomes, dnase_data, labels, chip_data, label_masks)
        dataloader = DataLoader(dataset=Task1Dataset(chr_data, target_label), batch_size=args.batchsize
                                , shuffle=True, num_workers=2)
        for step, (train_batch_idx) in enumerate(dataloader):
            t = time.time()
            train_seq = torch.FloatTensor(input_seqs[train_batch_idx])\
                .view(train_batch_idx.shape[0],5,lengths['DNA']).to(device)
            train_tf_data = input_chip[train_batch_idx, :lengths['TF'], :]\
                .view(train_batch_idx.shape[0],lengths['TF'],args.signal_length).to(device)
            train_hm_data = input_chip[train_batch_idx, lengths['TF']:, :]\
                .view(train_batch_idx.shape[0],lengths['HM'],args.signal_length).to(device)
            train_lmask = torch.FloatTensor(input_label_mask[train_batch_idx])\
                .view(train_batch_idx.shape[0],lengths['TF']+lengths['HM']).to(device)
            train_target = torch.FloatTensor(target_label[train_batch_idx])\
                .view(train_batch_idx.shape[0], lengths['TF'] + lengths['HM']).to(device)
            if mode == 'train':
                output = model(train_seq, train_tf_data, train_hm_data,
                        input_label_mask[train_batch_idx].reshape(train_batch_idx.shape[0],lengths['TF']+lengths['HM']),mask_prob=0.6)
                loss = criterion(output, train_target, train_lmask)
                cur_loss = loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_losses.append(cur_loss)
                if step % 1000 == 0:
                    print(mode)
                    print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1), "train_loss=",
                          "{:.7f}".format(cur_loss),
                          "time=", "{:.5f}".format(time.time() - t)
                          )
            else:
                temp_loss=[]
                validmasks = [0.3,0.45, 0.6,0.8, 1]
                for maski in range(len(validmasks)):
                    output = model(train_seq, train_tf_data, train_hm_data,
                                   input_label_mask[train_batch_idx].reshape(train_batch_idx.shape[0],lengths['TF']+lengths['HM']),mask_prob=validmasks[maski])
                    loss = criterion(output, train_target, train_lmask)
                    cur_loss = loss.item()
                    valid_mask_los[maski].append(cur_loss)
                    temp_loss.append(cur_loss)
                vloss=np.mean(temp_loss)
                validation_losses.append(vloss)
                if step % 1000 == 0:
                    print(mode)
                    print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1), "validation_loss=",
                          "{:.7f}".format(vloss),
                          "time=", "{:.5f}".format(time.time() - t)
                          )

    train_loss = np.average(training_losses)
    print('Epoch: {} LR: {:.8f} train_loss: {:.7f}'.format(epoch, optimizer.param_groups[0]['lr'], train_loss))
    valid_loss = np.average(validation_losses)
    scheduler.step(valid_loss)
    print('Epoch: {} LR: {:.8f} valid_loss: {:.7f}'.format(epoch, optimizer.param_groups[0]['lr'], valid_loss))

    for i in range(5):
        print(np.average(valid_mask_los[i]))


    torch.save(model.state_dict(),
                       '/scratch/drjieliu_root/drjieliu/zhenhaoz/models/pre_train_task1.pt')
    print('save model')

