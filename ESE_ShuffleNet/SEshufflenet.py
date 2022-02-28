
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)
    
def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.
    

class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:            
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                )
      
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels*2, 1, groups=2),
                nn.BatchNorm2d(in_channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels*2, in_channels*2, 3, stride=stride, padding=1, groups=in_channels*2),
                nn.BatchNorm2d(in_channels*2),
                #nn.ReLU(inplace=True)
                )
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Linear(in_channels*2, in_channels*2//16),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels*2//16, in_channels*2),
                #nn.Sigmoid()
                h_sigmoid()
                )
            self.final_conv = nn.Sequential(    
                nn.Conv2d(in_channels*2, int(out_channels/2), 1, groups=2),
                nn.BatchNorm2d(int(out_channels/2)),
                #nn.ReLU(inplace=True)
                )
            
        else:
            in_channels = int(in_channels/2)
            
            self.shortcut = nn.Sequential()
            
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels*2, 1, groups=2),
                nn.BatchNorm2d(in_channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels*2, in_channels*2, 3, stride=stride, padding=1, groups=in_channels*2),
                nn.BatchNorm2d(in_channels*2),
                #nn.ReLU(inplace=True)
                )
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Linear(in_channels*2, in_channels*2//16),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels*2//16, in_channels*2),
                #nn.Sigmoid()
                h_sigmoid()
                )
            self.final_conv = nn.Sequential(
                nn.Conv2d(in_channels*2, in_channels, 1, groups=2),
                nn.BatchNorm2d(in_channels),
                #nn.ReLU(inplace=True) 
                )
    
    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
            
            x1 = self.residual(residual)
            x2 = self.squeeze(x1)
            x3 = x2.view(x2.size(0), -1)
            x4 = self.excitation(x3)
            x5 = x4.view(x1.size(0), x1.size(1), 1, 1)
            x6 = x1*x5.expand_as(x1)
            resi_dual_pre = self.final_conv(x6)
            resi_dual = resi_dual_pre + residual
            
            short_cut = self.shortcut(shortcut)
            
        else:
            shortcut = x
            residual = x
            
            x1 = self.residual(residual)
            x2 = self.squeeze(x1)
            x3 = x2.view(x2.size(0), -1)
            x4 = self.excitation(x3)
            x5 = x4.view(x1.size(0), x1.size(1), 1, 1)
            x6 = x1*x5.expand_as(x1)
            resi_dual = self.final_conv(x6)
            
            short_cut = self.shortcut(shortcut)
            
        out = torch.cat([short_cut, resi_dual], dim=1)
        out = channel_shuffle(out, 2)
        
        return out


class SEshufflenet(nn.Module):

    def __init__(self, ratio=1, class_num=2):
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [128, 256, 512, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')
        
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage2 = self._make_stage(64, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(out_channels[3], class_num)

    def forward(self, x):

        x = self.pre(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1
        
        return nn.Sequential(*layers)

def seshufflenet():
    return SEshufflenet()


