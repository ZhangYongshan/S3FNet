import math
import torch
import torch.nn as nn
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y=None, z=None, **kwargs):
        if z is not None:
            # 注意残差连接的方式不同，这里我是和base（融合）特征做残差
            return self.fn(x, y, z, **kwargs) + z
        elif y is not None:
            return self.fn(x, y, **kwargs) + x
        else:
            return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y=None, z=None, **kwargs):
        if z is not None:
            return self.fn(self.norm(x), self.norm(y), self.norm(z), **kwargs)
        elif y is not None:
            return self.fn(self.norm(x), self.norm(y), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)

        return x

    # def flops(self, H, W):
    #     flops = 0
    #     # fc1
    #     flops += H * W * self.dim * self.hidden_dim
    #     # dwconv
    #     flops += H * W * self.hidden_dim * 3 * 3
    #     # fc2
    #     flops += H * W * self.hidden_dim * self.dim
    #     print("LeFF:{%.2f}" % (flops / 1e9))
    #     # eca
    #     if hasattr(self.eca, 'flops'):
    #         flops += self.eca.flops()
    #     return flops


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # 其实在本项目中 dim = heads * dim_head, inner_dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        # self.temperature = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.sa1 = nn.Linear(dim, inner_dim, bias=False)  # B (HW) C -> B (HW) inner_dim
        self.sa2 = nn.Linear(dim, inner_dim, bias=False)
        self.se1 = nn.Linear(dim, inner_dim, bias=False)
        self.se2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y, z):  # x ms细节特征 y pan细节特征 z 共同的基础特征（融合特征）
        # x,y:B (H W) C
        b, n, _, h = *x.shape, self.heads
        z1 = rearrange(self.sa1(z), 'b n (h d) -> b h n d', h=h)  # B (HW) inner_dim -> B heeds (HW) dim_head
        y = rearrange(self.sa2(y), 'b n (h d) -> b h n d', h=h)
        x = rearrange(self.se1(x), 'b n (h d) -> b h n d', h=h)
        z2 = rearrange(self.se2(z), 'b n (h d) -> b h n d', h=h)
        sacm = (z1 @ y.transpose(-2, -1)) * self.scale  # b h n n
        secm = (x.transpose(-2, -1) @ z2) * self.scale / (n / self.dim_head)  # b h d d
        sacm = sacm.softmax(dim=-1)
        secm = secm.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', sacm, x)  # b h n d
        out2 = torch.einsum('b h n i, b h i j -> b h n j', y, secm)  # b h n d
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out = out1 + out2
        out = self.to_out(out)
        return out


class MSSAF(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, depth=1, dropout=0., use_LeFF=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        if use_LeFF:
            mlp = LeFF(dim,hidden_dim=mlp_dim, drop=dropout)
        else:
            mlp = MLP(dim, hidden_dim=mlp_dim, dropout=dropout)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, mlp))]))

    def forward(self, x, y, z):
        H = x.shape[2]
        x = rearrange(x, 'B C H W -> B (H W) C', H=H)
        y = rearrange(y, 'B C H W -> B (H W) C', H=H)
        z = rearrange(z, 'B C H W -> B (H W) C', H=H)
        for attn, ff in self.layers:
            x = attn(x, y, z)
            x = ff(x)
        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    summary(MSSAF(64, 8, 8, 32), input_size=(64, 64, 64))
