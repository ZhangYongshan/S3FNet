import torch.nn as nn
import torch
import torch.nn.functional as F
from models.ablation.cdmmunet_base.mssaf import MSSAF


class Decoder(nn.Module):
    def __init__(self, dim_in=128, dim_head=16, dim_out=8, se_ratio_mlp=2):
        super(Decoder, self).__init__()

        dim0 = dim_in
        dim1 = dim_in // 2
        dim2 = dim_in // 4

        self.MSSAF0 = MSSAF(dim0, dim0 // dim_head, dim_head, int(dim0 * se_ratio_mlp))
        self.MSSAF1 = MSSAF(dim1, dim1 // dim_head, dim_head, int(dim1 * se_ratio_mlp))
        self.MSSAF2 = MSSAF(dim2, dim2 // dim_head, dim_head, int(dim2 * se_ratio_mlp))

        self.up01 = nn.Sequential(
            nn.ConvTranspose2d(dim0, dim1, 2, 2, 0),
            nn.LeakyReLU()
        )

        self.up12 = nn.Sequential(
            nn.ConvTranspose2d(dim1, dim2, 2, 2, 0),
            nn.LeakyReLU()
        )

        self.final_conv = nn.Conv2d(in_channels=dim0 + dim1 + dim2,
                                    out_channels=dim_out, kernel_size=3, padding=1)

    def forward(self, fuse_base, ms_detail, pan_detail):
        x = fuse_base[0]
        x = self.MSSAF0(ms_detail[0], pan_detail[0], x)
        x0 = x
        x = self.up01(x)
        x = x + fuse_base[1]
        x = self.MSSAF1(ms_detail[1], pan_detail[1], x)
        x1 = x
        x = self.up12(x)
        x = x + fuse_base[2]
        x = self.MSSAF2(ms_detail[2], pan_detail[2], x)
        x2 = x

        x0 = F.interpolate(x0, scale_factor=4, mode='bicubic')
        x1 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        xF = torch.cat((x0, x1, x2), dim=1)

        out = self.final_conv(xF)

        return out


if __name__ == '__main__':
    from torchsummary import summary

    summary(Decoder())
