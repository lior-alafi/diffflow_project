import torch
import torch.nn as nn
import torch.nn.functional as F
#

class CoPE1D(nn.Module):
    '''
    D - head_dim
    P - max_npos
    e - pos_emb

    paper: https://arxiv.org/pdf/2405.18719
CoPE: q_i^T(k_j+e[p_{i,j}]

Notice that p values are capped:
for any contextual position p_{i,j} >= P-1, we set p_{i,j} = P-1.
Therefore, all positions beyond P-1
collapse to the same final bucket.

originally:
e[p_{i,j}] = (p_{i,j}-floor(p_{i,j})ceil(e[p_{i,j}])
                    + (1-p_{i,j}+floor(p_{i,j}))floor(e[p_{i,j}])

let w_{i,j} = p_{i,j}-floor(p_{i,j})
we can rewrite
e[p_{i,j}] =w_{i,j}e[ceil(p_{i,j})]+(1-w_{i,j})e[floor(p_{i,j})]
if we'll do that we'll get [B, heads, T, T, head_dim]
the article does this interpolation using a trick:
z_i[p]=q_i^Te[p]
z_i[p_{i,j}]=wz_i[ceil(p_{i,j})]+(1-w)z_i[floor(p_{i,j})]

usage:
...
# attn_logits is masked already
attn_logits += self.cope(q,k,attn_logits)
s = softmax(attn_logits,dim=-1).
...
    '''
    def __init__(self,max_npos,head_dim):
        super().__init__()
        self.max_npos = max_npos
        self.head_dim =head_dim
        #e
        self.pos_emb = nn.Parameter(torch.zeros((1,head_dim,max_npos)))


    def forward(self,q,k,attn_logits,**kwargs):
        # g_{i,j}=sigmoid(q_i^Tk_j)
        g = F.sigmoid(attn_logits)
        #[B,heads,T,T]
        p = g.flip(-1).cumsum(-1).flip(-1)
        #[g0,g1,g2]->[g2,g1,g0]->[g2,g2+g1,g2+g1+g0]->[g0+g1+g2,g1+g2,g2]

        p = p.clamp(max=self.max_npos - 1)
        #[B,heads,T,T]


        # z_i(p)=q_i^Te[p]
        logits_int  = q @ self.pos_emb

        p_floor = p.floor().long()
        p_ceil = p.ceil().long()
        w = p - p_floor

        logits_floor  = logits_int.gather(-1,p_floor)
        #z_i[floor(p_{i,j})]
        # [B, heads, T, T]

        logits_ceil = logits_int.gather(-1,p_ceil)
        #z_i[ceil(p_{i,j})]
        # [B, heads, T, T]


        return w*logits_ceil+(1-w)*logits_floor

