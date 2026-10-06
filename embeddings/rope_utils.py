import torch


def rotate_by_angle(x, angles):
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    # repeat twice
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    return x * cos + rotate_trick(x) * sin


def rotate_trick(x):
        '''

        :param x:
        :param angles:
        :return:
        '''
        even = x[...,0::2]
        odd = - x[...,1::2]
        x_new = torch.stack((odd,even),-1)
        # [B, heads, T, head_dim/2, 2]
        x_new = x_new.flatten(-2)
        # [B, heads, T, head_dim/2 * 2]=[B, heads, T, head_dim]
        return x_new