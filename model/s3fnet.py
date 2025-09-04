import torch.nn as nn
import torchsummary
from decoder import Decoder
from encoder import Encoder
import torch.nn.functional as F


class net(nn.Module):
    def __init__(self, ms_dim=8, use_attention=(False, False, False, False),
                 resnet_block_nums=(1, 1, 1), restormer_block_nums=(1, 1, 1)):
        super(net, self).__init__()
        # 这里是在改变数据集wv3->gf2时候增加的 如果在测试wv3时候出错，注释这一行
        self.ms_dim = ms_dim
        self.Encoder = Encoder(inp_channels=ms_dim,resnet_block_nums=resnet_block_nums, restormer_block_nums=restormer_block_nums,
                                    ms_use_ca=use_attention[0],
                                    ms_use_sa=use_attention[1], pan_use_ca=use_attention[2],
                                    pan_use_sa=use_attention[3])
        # 在此处修改se_ratio_mlp
        self.Decoder = Decoder(dim_in=128, dim_head=16, dim_out=ms_dim, se_ratio_mlp=4)

    def forward(self, ms, pan):
        pan = pan.repeat(1, self.ms_dim, 1, 1)
        upms = F.interpolate(ms, scale_factor=4, mode='bilinear', align_corners=False)
        # ms = F.interpolate(ms.clone(), scale_factor=4, mode='bilinear', align_corners=False)
        spa_feature, spe_feature, fuse_feature = self.Encoder(upms, pan)
        img_Fuse = self.Decoder(fuse_feature, spe_feature, spa_feature) + upms

        return img_Fuse, spa_feature, spe_feature, fuse_feature


if __name__ == '__main__':
    torchsummary.summary(net(use_attention=(1, 0, 0, 1), resnet_block_nums=(1, 1, 1)).to('cuda'), input_size=[(8, 16, 16), (1, 64, 64)])
