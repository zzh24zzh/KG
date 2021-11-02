import numpy as np
import torch,os,pickle
def undersampling_negatives(xidx,targets,undersampling_ratio=0.6):
    all_negative_idx=np.where(np.sum(targets[xidx],1)==0)[0]

    probs=np.random.uniform(0,1,all_negative_idx.shape[0])
    mask_idx=np.where(probs<undersampling_ratio)[0]
    return np.delete(xidx,all_negative_idx[mask_idx])


def pos_class_weight(targets):
    pos_numbers=np.sum(targets,0)
    print(np.sort(pos_numbers))
    pos_numbers=np.amax(pos_numbers)/pos_numbers
    pos_ratio=np.sqrt(pos_numbers)
    return pos_ratio



def label_mask():
    cls = ['A549', 'H1', 'K562', 'MCF-7', 'GM12878', 'HepG2', 'HeLa-S3']
    with open('/scratch/drjieliu_root/drjieliu/zhenhaoz/investigated_chips.txt', 'r') as f:
        all_chips = f.read().splitlines()
    masks={}
    for cl in cls:
        masks[cl]=np.zeros(len(all_chips)).astype('int8')
    chip_file='/scratch/drjieliu_root/drjieliu/zhenhaoz/chip_seq_signal/100_resolution'
    for f in os.listdir(chip_file):
        if '.pickle' in f:
            cl,factor=f.split('_')[:2]
            if cl not in cls or factor not in all_chips:
                continue
            factor_idx=all_chips.index(factor)
            masks[cl][factor_idx]=1
    for cl in cls:
        print(np.sum(masks[cl]),cl)
    return masks
