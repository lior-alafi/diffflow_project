import torch
import torch.nn as nn


class AbsolutePosEnc1D(nn.Module):
        '''
        B=10000
        i - token index
        j - embed dim index
        T - token size
        D - embed size
        p_i^{2j}=sin(i\B^{i/2j/D}
        p_i^{2j+1}=cos(i\B^{i/2j/D}
        '''
        def __init__(self):
            super().__init__()
            self.B:int = 10000

        def forward(self,x):
            _,T,D = x.shape

            assert D % 2 == 0,'embedding dim (D) must be even'

            i = torch.arange(0,T,device=x.device,
                             dtype=x.dtype).unsqueeze(1)
            #[T,1]
            j = torch.arange(0,D,2,
                             device=x.device,
                             dtype=x.dtype)
            #[D/2]
            pos = torch.empty(T,D,
                              device=x.device,
                              dtype=x.dtype
                              )
            angles = i / (self.B**(j/D))
            pos[:,0::2] = torch.sin(angles)
            pos[:,1::2] = torch.cos(angles)

            return x+pos



