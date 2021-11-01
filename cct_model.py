import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
from layers import Transformer,generate_mask
import numpy as np
class CCTbert(nn.Module):
    def __init__(self,kmers,dim,lengths,signal_length,depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv1d(5,out_channels=dim,kernel_size=11,padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=kmers)
        )
        # self.tokenize=Rearrange('b c (L k) -> b (L) (k c)',k=kmers)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mask_token=nn.Parameter(torch.randn(1, 1, dim))

        self.types_token=['DNA','TF','HM']
        self.type_embedding = nn.Parameter(torch.randn(1, len(self.types_token), dim))
        # embed ChIP-seq
        seq_length, TF_num, HM_num = [lengths[t] for t in self.types_token]

        self.pos_embedding = nn.Parameter(torch.randn(1, int(seq_length // kmers), dim))
        #embed tf
        self.chip_embedding = nn.Linear(signal_length, dim)
        self.tf_pos_embedding = nn.Parameter(torch.randn(1, TF_num, dim))
        #embed hm
        self.hm_pos_embedding = nn.Parameter(torch.randn(1, HM_num, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self,sequence,tf_data,hm_data,label_mask,embed_seq=False):
        seq_embed=self.conv_layer(sequence).transpose(-2,-1)
        seq_embed+=self.pos_embedding
        seq_embed+=self.type_embedding[:,0,:]
        print(seq_embed.shape)
        tf_embed=self.chip_embedding(tf_data)
        b,n,_=tf_embed.shape
        print(b,n,label_mask.shape)
        tf_embed=generate_mask(tf_embed,label_mask[:,:n],self.mask_token)
        print(tf_embed.shape)
        tf_embed+=self.tf_pos_embedding
        tf_embed+=self.type_embedding[:,1,:]

        hm_embed = self.chip_embedding(hm_data)
        print(hm_embed.shape)
        hm_embed = generate_mask(hm_embed, label_mask[:, n:], self.mask_token)
        hm_embed += self.hm_pos_embedding
        hm_embed += self.type_embedding[:,2,:]


        x=torch.cat((seq_embed,tf_embed,hm_embed),dim=1)
        x = self.dropout(x)
        x = self.transformer(x)

        x=self.mlp_head(x[:,n:,:]).squeeze()
        return x