import os,pickle
from scipy.sparse import load_npz
import numpy as np

cls = ['A549', 'H1', 'K562', 'MCF-7', 'GM12878', 'HepG2', 'HeLa-S3']
#
def pad_seq_matrix(matrix, pad_len=400):
    # add flanking region to each sample
    paddings = np.zeros((1, 4, pad_len)).astype('int8')
    dmatrix = np.concatenate((paddings, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], paddings), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)


def pad_signal_matrix(matrix, pad_len=400):
    paddings = np.zeros(pad_len).astype('float32')
    dmatrix = np.vstack((paddings, matrix[:, -pad_len:]))[:-1, :]
    umatrix = np.vstack((matrix[:, :pad_len], paddings))[1:, :]
    return np.hstack((dmatrix, matrix, umatrix))


def load_data():
    ref_path = '/scratch/drjieliu_root/drjieliu/zhenhaoz/ref_genome'
    dnase_path = '/scratch/drjieliu_root/drjieliu/zhenhaoz/DNase'
    chip_signal_dir = '/scratch/drjieliu_root/drjieliu/zhenhaoz/chip_seq_signal/100_resolution/'
    ref_genomes = {}
    for chr in range(1, 23):
        ref_file = os.path.join(ref_path, 'chr%s.npz' % chr)
        ref_gen_data = load_npz(ref_file).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
        ref_gen_data = pad_seq_matrix(ref_gen_data)
        ref_genomes[chr] = ref_gen_data
    print('sequence load finished')
    dnase_data = {}
    for cl in cls:
        dnase_data[cl] = {}
        for chr in range(1, 23):
            dnase_file = os.path.join(dnase_path, cl, 'chr%s.npy' % chr)
            temp_data = np.load(dnase_file)
            dnase_data[cl][chr] = pad_signal_matrix(temp_data.reshape(-1, 1000))
    print('dnase load finished')
    chip_data = {}
    for cl in cls:
        chip_data[cl] = {}
        for chr in range(1, 23):
            chip_file = os.path.join(chip_signal_dir, cl, 'chr%s.npy' % chr)
            temp_data = np.load(chip_file)
            chip_data[cl][chr]=temp_data
    print('chip load finished')
    with open('/scratch/drjieliu_root/drjieliu/zhenhaoz/labels/input_initial_label.pickle','rb') as f:
        labels=pickle.load(f)
    return ref_genomes, dnase_data, chip_data,labels

def concat_data(data,chroms,sparse=False):
    temp=[]
    for i in range(len(chroms)):
        if sparse:
            temp.append(data[cls[i]][chroms[i]].toarray())
        else:
            temp.append(data[cls[i]][chroms[i]])
    temp=np.concatenate(temp,axis=0)
    return temp

def generate_label_mask(label_masks,num_list):
    temp=[]
    for i in range(len(cls)):
        cl=cls[i]
        temp.append(np.tile(label_masks[cl],(num_list[i],1)))
    return np.vstack(temp)

