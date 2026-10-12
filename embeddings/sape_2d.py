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

        sape_x = self._interp_x(p_x,q)

        b_x = torch.cdist(sape_x,sape_x,p=2.0)
        # ||sape_x[i] - sape_x[n]||_2
        # equivalent to
        # a = x.unsqueeze(-2)
        # b = x.unsqueeze(-3)
        # diff = a - b
        # torch.norm(diff)

        g_y = self._calc_g_y(q,k,H_patched,W_patched)
        p_y = self._calc_p(g_y)
        sape_y = self._interp_y(p_y, q)
        b_y = torch.cdist(sape_y,sape_y,p=2.0)

        return b_x + b_y

    def _calc_g_x(self,q,k,H_patched,W_patched):
        B_q, heads_q, T_q, d_q = q.shape
        B_k, heads_k, T_k, d_k = k.shape
        q_row = q.reshape(B_q, heads_q, H_patched,W_patched, d_q)
        k_row = k.reshape(B_k, heads_k, H_patched,W_patched, d_k)
        g_x = torch.sigmoid(q_row @ k_row.transpose(-2, -1))
        # [B, heads,H, W, W]

        return g_x

    def _calc_p(self, g):

        p = g.flip(-1).cumsum(-1).flip(-1)
        p = p.clamp(max=self.max_npos - 1)
        # if g_x: [B, heads,H, W, W] if g_y: [B, heads,W, H, H]

        return p


    def _interp_x(self,p_x,q):
        #creating SaPE_x^2[i] \in R^W

        z = q @ self.e_x
        # [B, heads, T, P]
        B, heads, T, P = z.shape

        u = p_x.ceil().long()
        l = p_x.floor().long()
        w = p_x - l
        # [B, heads, H, W, W]
        _, _, H, W, _ = p_x.shape
        assert T == H * W
        assert P == self.max_npos
        z_row = z.reshape(B, heads,H ,W, P)
        z_ceil = z_row.gather(-1, u)
        # gather causes [B, heads, H, W, W]
        z_floor = z_row.gather(-1, l)
        interp_x = w * z_ceil + (1 - w) * z_floor

        interp_x = interp_x.flatten(2, 3)
        # [B, heads, T , W]
        return interp_x

    def _calc_g_y(self,q,k,H_patched,W_patched):
        B_q, heads_q, T_q, d_q = q.shape
        B_k, heads_k, T_k, d_k = k.shape

        q_col = q.reshape(B_q, heads_q, H_patched,W_patched, d_q)
        q_col = q_col.transpose(2,3) # the cool trick here to force (0,3)(1,4) etc..
        k_col = k.reshape(B_k, heads_k, H_patched,W_patched, d_k)
        k_col = k_col.transpose(2,3)
        g_y = torch.sigmoid(q_col @ k_col.transpose(-2, -1))

        return g_y

    def _interp_y(self,p_y,q):
        z = q @ self.e_y
        B, heads, T, P = z.shape
        u = p_y.ceil().long()
        l = p_y.floor().long()
        w = p_y - l

        _, _, W, H, _ = p_y.shape
        z_col = z.reshape(B, heads, H,W, P)
        z_col = z_col.transpose(2,3)
        # [B, heads, W,H, P]
        z_ceil = z_col.gather(-1, u)
        # [B, heads, W,H, H]
        z_floor = z_col.gather(-1, l)
        interp_y = w * z_ceil + (1 - w) * z_floor

        #now we need to fix # [B, heads, W, H, H] -> # [B,heads,H,W,H]
        interp_y = interp_y.transpose(2,3)
        interp_y = interp_y.flatten(2,3)


        return interp_y