import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
import numpy as np
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Balanced_AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05,alpha=None, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(Balanced_AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.alpha=alpha
    def forward(self, x, y,mask):
        # Calculating Probabilities
        assert y.shape== mask.shape
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        if self.alpha is not None:
            los_pos=self.alpha*los_pos
        loss = los_pos + los_neg
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
            loss*=mask
        return -loss.sum()/(torch.sum(mask)+self.eps)


def generate_mask(input,label_mask,mask_token,mask_prob):
    b,n,_=input.shape
    probs = np.random.uniform(0, 1, (b,n))
    tmask=np.where(probs<mask_prob,0,1)
    tmask=tmask*label_mask
    input[np.where(tmask==0)]=mask_token
    return input



# class AsymmetricLoss(nn.Module):
#     def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
#         super(AsymmetricLoss, self).__init__()
#
#         self.gamma_neg = gamma_neg
#         self.gamma_pos = gamma_pos
#         self.clip = clip
#         self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
#         self.eps = eps
#
#     def forward(self, x, y,mask,pos_weights):
#         # Calculating Probabilities
#         x_sigmoid = torch.sigmoid(x)
#         xs_pos = x_sigmoid
#         xs_neg = 1 - x_sigmoid
#         # Asymmetric Clipping
#         if self.clip is not None and self.clip > 0:
#             xs_neg = (xs_neg + self.clip).clamp(max=1)
#
#         # Basic CE calculation
#         los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
#         los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
#         loss = los_pos + los_neg
#
#         # Asymmetric Focusing
#         if self.gamma_neg > 0 or self.gamma_pos > 0:
#             if self.disable_torch_grad_focal_loss:
#                 torch.set_grad_enabled(False)
#             pt0 = xs_pos * y
#             pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
#             pt = pt0 + pt1
#             one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
#             one_sided_w = torch.pow(1 - pt, one_sided_gamma)
#             if self.disable_torch_grad_focal_loss:
#                 torch.set_grad_enabled(True)
#             loss *= one_sided_w
#             loss*=mask
#         return -loss.sum()

# criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
# input = torch.randn((2,3), requires_grad=True)
# target=torch.empty((2,3)).random_(2)
# mask=torch.empty((2,3)).random_(2)
# print(mask)
# l=criterion(input,target,mask)
# print(l)