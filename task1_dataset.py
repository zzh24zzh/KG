from torch.utils.data import Dataset
import pickle, os, torch
from scipy.sparse import load_npz
import numpy as np
from train_util import undersampling_negatives

cls = ['A549', 'H1', 'K562']
# , 'MCF-7', 'GM12878', 'HepG2', 'HeLa-S3'
chr_lens=[248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422,
          135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468]
with open('/scratch/drjieliu_root/drjieliu/zhenhaoz/investigated_chips.txt', 'r') as f:
    all_chips= f.read().splitlines()



with open('/scratch/drjieliu_root/drjieliu/zhenhaoz/input_loc.pickle','rb') as f:
    input_loc=pickle.load(f)
with open('/scratch/drjieliu_root/drjieliu/zhenhaoz/ref_genome/gapped_indices.pickle','rb') as f:
    gapped_indices=pickle.load(f)
input_indices={}
for chr in range(1,6):
    temp=np.delete(input_loc[chr] ,
                          np.intersect1d(input_loc[chr],gapped_indices[chr],return_indices=True)[1])
    input_indices[chr]=temp//1000

sample_nums=[chr_lens[i-1]//1000 for i in range(1,6)]
class Task1Dataset(Dataset):
    def __init__(self, chroms,target_label):
        input_nums=[0]+[sample_nums[chr-1] for chr in chroms]
        pad_sum=[sum(input_nums[:i+1]) for i in range(len(chroms))]
        xindices=[]
        for i in range(len(pad_sum)):
            xindices.append(input_indices[chroms[i]]+pad_sum[i])
        self.x_index=undersampling_negatives(np.concatenate(xindices),target_label)
        self.num=self.x_index.shape[0]
    def __getitem__(self, index):
        return self.x_index[index]
    def __len__(self):
        return self.num
