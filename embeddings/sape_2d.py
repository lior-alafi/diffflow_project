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
        # q, k[B, heads, T, d]
        g_x = self._calc_g_x(q,k,H_patched,W_patched)
        p_x = self._calc_p_x(g_x)

        sape_x = self._interpx__(p_x,W_patched,q)




        return sape_x

    def _calc_g_x(self,q,k,H_patched,W_patched):
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

    def _calc_p_x(self,g):
        p_x = []
        for g_x_i in g:
            p_x_i = g_x_i.flip(-1).cumsum(-1).flip(-1)
            p_x_i = p_x_i.clamp(max=self.max_npos - 1)
            # [B, heads, W, W]
            p_x.append(p_x_i)

        return p_x


    def _interp_x(self,p_x,W_patched,q):
        #creating SaPE_x^2[i] \in R^W

        z = q @ self.e_x
        # [B, heads, T, P]
        interp_x = []
        for i,p_x_i in enumerate(p_x):
            start = i*W_patched
            end = (i+1)*W_patched
            z_row = z[...,start:end,:]
            # [B,heads, W, P]

            u = p_x_i.ceil().long()
            l = p_x_i.floor().long()
            w = p_x_i - l
            z_ceil = z_row.gather(-1,u)
            # gather causes [B, heads, W, W]
            z_floor = z_row.gather(-1,l)
            interp = w*z_ceil+(1-w)*z_floor
            # [B, heads, W, W]
            interp_x.append(interp)

        interp_x = torch.stack(interp_x, dim=2)
        # [B, heads, H, W, W]
        interp_x = interp_x.flatten(2, 3)
        # [B, heads, T , W]
        return interp_x
