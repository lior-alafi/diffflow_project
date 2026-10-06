import torch
from torch import nn


class AbsolutePosEnc2D(nn.Module):
    '''
        B=10000
        i - token index
        j - embed dim index
        T - token size
        D - embed size

        embedding dimensions:
       0 ................ 127 | 128 ............... 255
       H encoding      |       W encoding
          128          |          128


        perform APE1D for Height//patch
        and APE1D for Width//patch
    '''
    def __init__(self):
        super().__init__()
        self.B = 10000

    def forward(self,x,H_patch,W_patch):
        N,T,D = x.shape
        assert D % 4 == 0, 'embedding dim(D) must be divisable by 4'
        assert T == H_patch * W_patch
        dtype = x.dtype
        device = x.device
        half_D = D // 2


        pos = torch.empty(T,D,dtype=dtype,device=device)
        # spatial position of every token
        i_h = torch.arange(H_patch,device=device,dtype=dtype).unsqueeze(1)
        i_w = torch.arange(W_patch,device=device,dtype=dtype).unsqueeze(1)
        # [T, 1] (both)

        j = torch.arange(0,half_D,2,dtype=dtype,device=device)
        # [D/4]

        angle_h = i_h/(self.B**(j/half_D))
        # [W_patch, D/4]
        angle_w = i_w/(self.B**(j/half_D))
        # [W_patch, D/4]

        angle_h = angle_h.repeat_interleave(W_patch, dim=0)
        # [T, D/4]
        angle_w = angle_w.repeat(H_patch,1)
        # [T, D/4]


        #encode height positions
        pos[:,0:half_D:2] = torch.sin(angle_h)
        pos[:,1:half_D:2] = torch.cos(angle_h)

        pos[:,half_D::2] = torch.sin(angle_w)
        pos[:,half_D+1::2] = torch.cos(angle_w)

        return x + pos

