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
from train_util import pos_class_weight

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.5, help='Learning rate.')
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--warm_up', type=int, default=0,help='warmup training')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--pre_model', type=str, choices=['deepsea', 'expecto', 'danq'], default='expecto')
parser.add_argument('--load_model', default=False, action='store_true', help='load trained model')
parser.add_argument('--test', default=False, action='store_true', help='model testing')
args = parser.parse_args()


def find_gapped(ref_matrix):
    # find gapped region including 'N'
    temp_sum = ref_matrix.sum(axis=1).sum(axis=1)
    assert temp_sum.shape[0] == ref_matrix.shape[0]
    gapped_idx = np.where(temp_sum != ref_matrix.shape[2])[0]
    return gapped_idx


# chrs=[]
# dataloader = DataLoader(dataset=Task1Dataset(), batch_size=32, shuffle=False, num_workers=2)
cls = ['A549', 'H1', 'K562', 'MCF-7', 'GM12878', 'HepG2', 'HeLa-S3']
lengths = {'DNA': 2000, 'HM': 11, 'TF': 256}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CCTbert(
    kmers=5, dim=360, lengths=lengths,
    signal_length=10, depth=6, heads=4,
    dim_head=128, mlp_dim=360, dropout=0.1
)
model.to(device)

pos_ratio = np.load('/scratch/drjieliu_root/drjieliu/zhenhaoz/pos_class_weight_sqrt.npy')
criterion = Balanced_AsymmetricLoss(gamma_neg=4, gamma_pos=2, clip=0.05, alpha=torch.FloatTensor(pos_ratio).to(device))
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)


def shuffle_data():
    chrs = np.arange(1, 23)
    temp = np.tile(chrs, (len(cls), 1))
    for i in range(len(cls)):
        np.random.shuffle(temp[i, :])
    return temp


# load all the training data to memory
ref_genomes, dnase_data, chip_data, labels = load_data()

# miss chip-seq in each cell line
with open('/scratch/drjieliu_root/drjieliu/zhenhaoz/label_mask.pickle', 'rb') as f:
    label_masks = pickle.load(f)

best_loss=1e10
for epoch in range(args.epochs):
    iterations = shuffle_data()
    print(iterations)
    training_losses = []
    validation_losses = []
    for iter in range(iterations.shape[1]):
        if iter < 20 or epoch < args.warm_up:
            model.train()
            mode = 'train'
        else:
            model.eval()
            mode = 'validation'
        chr_data = iterations[:, iter].tolist()
        print(chr_data)

        input_ref_genome = np.concatenate([ref_genomes[chr] for chr in chr_data], axis=0)
        input_dnase = concat_data(dnase_data, chr_data)
        target_label = concat_data(labels, chr_data, sparse=True)
        input_chip = concat_data(chip_data, chr_data)
        input_label_mask = generate_label_mask(label_masks, [ref_genomes[chr].shape[0] for chr in chr_data])
        print(input_label_mask.shape)
        dataloader = DataLoader(dataset=Task1Dataset(chr_data, target_label), batch_size=args.batchsize
                                , shuffle=False, num_workers=2)

        for step, (train_batch_idx) in enumerate(dataloader):
            if step > 2:
                break
            t = time.time()
            train_seq = np.concatenate((input_ref_genome[train_batch_idx], input_dnase[train_batch_idx]), axis=0)
            train_seq = torch.FloatTensor(train_seq).to(device)
            print(train_seq.shape)
            train_tf_data = torch.FloatTensor(input_chip[:, :lengths['TF'], :]).to(device)
            train_hm_data = torch.FloatTensor(input_chip[:, lengths['TF']:, :]).to(device)
            train_lmask = torch.FloatTensor(input_label_mask[train_batch_idx]).to(device)
            train_target = torch.FloatTensor(target_label[train_batch_idx]).to(device)
            output = model(train_seq, train_tf_data, train_hm_data, train_lmask)
            print(output.shape)
            loss = criterion(output, train_target, train_lmask)
            cur_loss = loss.item()
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_losses.append(cur_loss)
                if step % 30000 == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1), "train_loss=",
                          "{:.7f}".format(cur_loss),
                          "time=", "{:.5f}".format(time.time() - t)
                          )
            else:
                validation_losses.append(cur_loss)
    if epoch>= args.warm_up:
        valid_loss = np.average(validation_losses)
        # schedule.step(valid_loss)
        print('valid Loss is %s' % valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(),
                       '/scratch/drjieliu_root/drjieliu/zhenhaoz/models/models/pre_train_task1.pt')
            print('save model')

