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

        B, heads, T, d = q.shape

        assert k.shape == q.shape
        assert d == self.head_dim
        assert T == H_patched * W_patched

        g_x = self._calc_g_x(q,k,H_patched,W_patched)
        p_x = self._calc_p(g_x)

        sape_x = self._interp_x(p_x,W_patched,q)

        b_x = torch.cdist(sape_x,sape_x,p=2.0)
        # ||sape_x[i] - sape_x[n]||_2
        # equivalent to
        # a = x.unsqueeze(-2)
        # b = x.unsqueeze(-3)
        # diff = a - b
        # torch.norm(diff)

        g_y = self._calc_g_y(q,k,H_patched,W_patched)
        p_y = self._calc_p(g_y)
        sape_y = self._interp_y(p_y, W_patched, q)
        b_y = torch.cdist(sape_y,sape_y,p=2.0)

        return b_x + b_y

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

    def _calc_p(self, g):
        p = []
        for g_i in g:
            p_i = g_i.flip(-1).cumsum(-1).flip(-1)
            p_i = p_i.clamp(max=self.max_npos - 1)
            # if g_x: [B, heads, W, W] if g_y: [B, heads, H, H]
            p.append(p_i)

        return p


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

    def _calc_g_y(self,q,k,H_patched,W_patched):
        g_y = []

        for i in range(W_patched):
            q_col = q[...,i::W_patched,:]
            # q_col: [B, heads, H, d]
            k_col = k[..., i::W_patched, :]

            # [B, heads, H, H]
            g_y_i = torch.sigmoid(q_col @ k_col.transpose(-2,-1))
            g_y.append(g_y_i)

        return g_y

    def _interp_y(self,p_y,W_patched,q):
        interp_y = []
        z = q @ self.e_y
        # [B, heads, T, P]

        for i,p_y_i in enumerate(p_y):
            z_col = z[...,i::W_patched,:]
            u = p_y_i.ceil().long()
            l = p_y_i.floor().long()
            w = p_y_i - l

            z_ceil = z_col.gather(-1,u)
            z_floor = z_col.gather(-1,l)
            interp = w * z_ceil + (1-w)*z_floor
            interp_y.append(interp)

        interp_y = torch.stack(interp_y,dim=2)
        #now we need to fix # [B, heads, W, H, H] -> # [B,heads,H,W,H]
        interp_y = interp_y.transpose(2,3)
        interp_y = interp_y.flatten(2,3)


        return interp_y