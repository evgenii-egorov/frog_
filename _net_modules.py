import torch
import torch.nn as nn


class CircularConv(nn.Module):
    def __init__(self, ch_in, ch_out, k_size, depthwise=False, bias=True, groups=1):
        super().__init__()
        if depthwise:
            conv_spatial = nn.Conv2d(ch_in, ch_out,
                                  kernel_size=k_size, stride=1, padding_mode='circular',
                                  groups=ch_in, bias=False, padding='same')
            conv_depth = nn.Conv2d(ch_out, ch_out,
                                  kernel_size=1, stride=1, padding_mode='circular',
                                  groups=groups, bias=bias,
                                  padding='same')

            nn.init.kaiming_normal_(conv_spatial.weight)
            nn.init.kaiming_normal_(conv_depth.weight)
            self.conv = nn.Sequential(conv_spatial, conv_depth)
        else:
            conv = nn.Conv2d(ch_in, ch_out,
                                  kernel_size=k_size, stride=1, padding_mode='circular',
                                  groups=groups, bias=bias,
                                  padding='same')
            nn.init.kaiming_normal_(conv.weight)
            self.conv = conv

    def forward(self, x):
        x = self.conv(x)
        return x


class I(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Res(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x) + x


class MakeSquare(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L

    def forward(self, syndrome):
        b_size = syndrome.shape[0]
        return syndrome.reshape(b_size, 2, self.L, self.L)


class Scaler(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean',  mean)
        self.register_buffer('std', std)

    def forward(self, syndrome):
        out = syndrome - self.mean
        out = out / self.std
        return out


class ConvEqviHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b_size = x.shape[0]
        L = x.shape[-1]

        out = torch.permute(x, (0, 2, 3, 1))
        out = torch.flip(torch.roll(out, (-1, -1), (0+1, 1+1)), [0+1, 1+1])
        out = out.reshape(b_size, L, L, 4, 2, 2)
        return out


class ConvAveragePoolingHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b_size = x.shape[0]
        L = x.shape[-1]

        out = torch.permute(x, (0, 2, 3, 1))
        out = out.reshape(b_size, L, L, 4, 2, 2)
        return out


class ConvFCHead(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.FC = nn.Linear(L ** 2 * 16, 16)

    def forward(self, x):
        b_size = x.shape[0]
        L = x.shape[-1]
        out = torch.permute(x, (0, 2, 3, 1))
        out = out.reshape(b_size, L ** 2 * 16)
        out = self.FC(out)
        out = out.reshape(b_size, 4, 2, 2)
        return out


def init_net(net, bias_last=None):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    if bias_last is not None:
        last_module = list(net.modules())[-1]
        last_module.bias.data = bias_last.data


def conv_maker(ch, k_size, ch_in=2, ch_out=16, f_non_linear=nn.GELU(), norm=nn.BatchNorm2d):
    depth = len(ch) - 1
    net = nn.Sequential(

        CircularConv(ch_in, ch[0], k_size[0]),
        f_non_linear,
        norm(ch[0]),

        CircularConv(ch[0], ch[0], k_size[0]),
        f_non_linear,
        norm(ch[0]),

        *[
            nn.Sequential(
                Res(nn.Sequential(
                    CircularConv(ch[i], ch[i], k_size[i]),
                    f_non_linear,
                    norm(ch[i]))
                ),
                f_non_linear,
                norm(ch[i]),

                CircularConv(ch[i], ch[i+1], k_size[i]),
                f_non_linear,
                norm(ch[i+1])

            ) for i in range(depth)
        ],
       CircularConv(ch[-1], ch_out, k_size[-1])
    )
    return net
