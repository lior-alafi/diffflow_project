from torch.nn import Module,Conv2d

class PatchEmbedd(Module):
    def __init__(self,patch_size:int ,in_channels:int =3,embed_size:int =256):
        super().__init__()
        self.patch_size = patch_size
        self.conv = Conv2d(in_channels=in_channels,out_channels=embed_size,
                           kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        # D - embed_dim
        N,C,H,W = x.shape
        latent = self.conv(x)
        # [N, D , H/P, W/P]
        latent = latent.flatten(2)
        # [N, D, T]
        latent = latent.transpose(1,2)
        # [N, T, D]
        return  latent,H//self.patch_size,W//self.patch_size
