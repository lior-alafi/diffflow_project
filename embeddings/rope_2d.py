import torch
import torch.nn as nn

from embeddings.rope_utils import rotate_by_angle

ERR_DIV_BY_4 = '''head_dim must be divisible by 4
because heads_dim = D/2 + D/2 
                    H     W 
and RoPE(1D) works on heads_dim/2
'''

class RoPE2D(nn.Module):
    def __init__(self,base=10000):
        super().__init__()
        self.B = base

    def forward(self,q,k,H_patch,W_patch,**kwargs):
        # q, k: [B, heads, T, head_dim]
        _, _, T_q, heads_dim_q = q.shape
        _, _, T_k, heads_dim_k = k.shape

        #self attention assumption
        assert heads_dim_q == heads_dim_k, \
            "q and k must have the same head_dim"

        assert T_q == H_patch * W_patch, \
            "q token count must equal H_patch * W_patch"

        assert T_k == H_patch * W_patch, \
            "k token count must equal H_patch * W_patch"
        assert heads_dim_q % 4 == 0, ERR_DIV_BY_4
        half_head_dim = heads_dim_q // 2
        angles_H,angles_W = self.create_angles(half_head_dim,H_patch,W_patch,q.device)
        angles_H, angles_W = angles_H.to(q.dtype),angles_W.to(q.dtype)


        q_rot = self.rotator(q,angles_H,angles_W)
        k_rot = self.rotator(k,angles_H,angles_W)


        return q_rot,k_rot
    def create_angles(self,half_head_dim,H_patch,W_patch,device):
        '''
         m_H and m_W explained:
         because T_q is the flattened grid:
         e.g H_patch = 3 W_patch =2 T_q = H_patch*W_patch
         1- (0,0) 2- (0, 1)
         3- (1,0) 4- (1, 1)
         5- (2,0) 6- (2, 1)
         i.e in H = [0 0 1 1 2 2],W=[0,1,0,1,0,1]
        :param half_head_dim:
        :param H_patch:
        :param W_patch:
        :param device:
        :return:
        '''
        j = torch.arange(0, half_head_dim, 2, device=device, dtype=torch.float32)
        # [head_dim/4]

        h,w = torch.meshgrid(torch.arange(H_patch,device=device),
                             torch.arange(W_patch, device=device),
                             indexing='ij'
                             )
        m_H = h.flatten().unsqueeze(1)
        # [T, 1]
        m_W = w.flatten().unsqueeze(1)
        # [T, 1]
        theta = self.B ** (-j / half_head_dim)
        angles_H = m_H * theta
        angles_W = m_W * theta
        return angles_H,angles_W


    def rotator(self,x,angles_H,angles_W):
        half_head_dim = x.shape[-1] // 2
        x_H = x[..., :half_head_dim]
        x_W = x[..., half_head_dim:]
        x_H_rot = rotate_by_angle(x_H, angles_H)
        x_W_rot = rotate_by_angle(x_W, angles_W)
        x_rot = torch.cat((x_H_rot, x_W_rot), dim=-1)
        return x_rot