import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
from layers import Transformer,generate_mask
class KGbert(nn.Module):
    def __init__(self,kmers,dim,channel,lengths,signal_length,depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.tokenize=Rearrange('b c (L k) -> b (L) (k c)',k=kmers)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mask_token=nn.Parameter(torch.randn(1, 1, dim))

        self.types_token=['DNA','TF','HM']
        self.type_embedding = nn.Parameter(torch.randn(1, len(self.types_token), dim))
        # embed ChIP-seq
        seq_length, TF_num, HM_num = [lengths[t] for t in self.types_token]
        # embed kmers
        self.kmer_embedding=nn.Linear(kmers*channel,dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, int(seq_length // kmers), dim))
        #embed tf
        self.chip_embedding = nn.Linear(signal_length, dim)
        self.tf_pos_embedding = nn.Parameter(torch.randn(1, TF_num, dim))
        #embed hm
        self.hm_pos_embedding = nn.Parameter(torch.randn(1, HM_num, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self,sequence,tf_data,hm_data):
        seq_embed=self.kmer_embedding(self.tokenize(sequence))
        seq_embed+=self.pos_embedding
        seq_embed+=self.type_embedding[:,0,:]

        tf_embed=self.chip_embedding(tf_data)
        tf_embed+=self.tf_pos_embedding
        tf_embed+=self.type_embedding[:,1,:]

        hm_embed = self.chip_embedding(hm_data)
        hm_embed += self.hm_pos_embedding
        hm_embed += self.type_embedding[:,2,:]

        x=torch.cat((seq_embed,tf_embed,hm_embed),dim=1)
        x = self.dropout(x)
        x = self.transformer(x)

