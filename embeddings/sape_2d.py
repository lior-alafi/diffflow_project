import torch
import torch.nn as nn


class SaPE2D(nn.Module):
    def __init__(self,head_dim,max_npos):
        super().__init__()
        self.max_npos = max_npos
        self.head_dim = head_dim
        self.e_x = nn.Parameter(torch.empty(size=[1,head_dim,max_npos]))
        self.e_y = nn.Parameter(torch.empty(size=[1,head_dim,max_npos]))

        nn.init.trunc_normal_(self.e_x,std=0.02)
        nn.init.trunc_normal_(self.e_y,std=0.02)

    def forward(self,q,k,H_patched,W_patched,**kwargs):
        g_x = self.__calc_g_x(q,k,H_patched,W_patched)
        p_x = self.__calc_p_x(g_x)

        z = q @ self.e_x
        interp_x = []
        for p_x_i in p_x:
            u = p_x_i.ceil()
            l = p_x_i.floor()
            w = p_x_i - l
            interp_i = w @ self.e_x[l]+(1-w)@self.e_x[u]
            interp_x.append(interp_i)


        return q

    def __calc_g_x(self,q,k,H_patched,W_patched):
        g_x = []

        for i in range(H_patched):
            start = i*W_patched
            end = (i+1)*W_patched
            q_row = q[...,start:end,:]
            #q_row: [B, heads, W, d]
            k_row = k[...,start:end,:]
            # k_row: [B, heads, W, d]
            g_x_i = torch.sigmoid(q_row @ k_row.transpose(-2,-1))

            #[B, heads, W, W]
            g_x.append(g_x_i)
        return g_x

    def __calc_p_x(self,g):
        p_x = []
        for g_x_i in g:
            p_x_i = g_x_i.flip(-1).cumsum(-1).flip(-1)
            p_x_i = p_x_i.clamp(max=self.max_npos - 1)
            p_x.append(p_x_i)

        return p_x



