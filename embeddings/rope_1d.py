import torch
import torch.nn as nn

from embeddings.rope_utils import rotate_by_angle


class RoPE1D(nn.Module):
    '''
    Roformer(RoPE) https://arxiv.org/pdf/2104.09864
    creating the rotation matrix

    thus
    e^{in\theta} is eqiv to rotation matrix
    and we rotate
    q_rot  = qe^{in\theta}
    k_rot  = ke^{in\theta}
    https://huggingface.co/blog/designing-positional-encoding
    '''
    def __init__(self, base=10000):
        super().__init__()
        self.B = base



    def forward(self,q,k,**kwargs):
        '''
        m- token position
        q, k: [B, heads, T, head_dim]
        '''
        # q, k: [B, heads, T, head_dim]

        q_angles = self.calc_angles(q)
        k_angles = self.calc_angles(k)


        q_rot = rotate_by_angle(q,q_angles)
        k_rot = rotate_by_angle(k,k_angles)


        return q_rot,k_rot


    def calc_angles(self,x):

        # x: [B, heads, T, head_dim]
        _,_,T,heads_dim = x.shape
        assert heads_dim % 2 == 0,'heads_dim must be even'
        j = torch.arange(0,heads_dim,2,
                            device=x.device,dtype=torch.float32)
        theta = self.B**(-j/heads_dim)

        m = torch.arange(T,device=x.device,dtype=torch.float32).unsqueeze(1)
        #[T,1]
        angles = m*theta
        return angles.to(x.dtype)





