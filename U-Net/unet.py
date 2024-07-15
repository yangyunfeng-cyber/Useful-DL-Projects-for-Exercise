from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义双卷积类
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            # 定义padding为1，保持卷积后特征图的宽度和高度不变，具体计算N= (W-F+2P)/S+1
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # 加入BN层，提升训练速度，并提高模型效果
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 第二个卷积层，同第一个
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # BN层
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# 调用上面定义的双卷积类，定义下采样类
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            # 最大池化，步长为2，池化核大小为2，计算公式同卷积，则 N = (W-F+2P)/S+1,  N= (W-2+0)/4 + 1
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        # 调用转置卷积的方法进行上采样，使特征图的高和宽翻倍，out  =(W−1)×S−2×P+F，通道数减半
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # 调用双层卷积类，通道数是否减半要看out_channels接收的值
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # X的shape为[N, C, H, W]，下面三行代码主要是为了保证x1和x2在维度为2和3的地方保持一致，方便cat操作不出错。
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # 增加padding操作，padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# 定义输出卷积类
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,  # 默认输入图像的通道数为1，这里一般黑白图像为1，而彩色图像为3
                 num_classes: int = 2,  # 默认输出的分类类别数为2
                 # 默认基础通道为64，这里也可以改成大于2的任意2的次幂，不过越大模型的复杂度越高，参数越大，模型的拟合能力也就越强
                 base_c: int = 64):

        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # 编码器的第1个双卷积层，不包含下采样过程，输入通道为1，输出通道数为base_c,这个值可以为64或者32
        self.in_conv = DoubleConv(in_channels, base_c)
        # 编码器的第2个双卷积层，先进行下采样最大池化使得特征图高和宽减半，然后通道数翻倍
        self.down1 = Down(base_c, base_c * 2)
        # 编码器的第3个双卷积层，先进行下采样最大池化使得特征图高和宽减半，然后通道数翻倍
        self.down2 = Down(base_c * 2, base_c * 4)
        # 编码器的第4个双卷积层，先进行下采样最大池化使得特征图高和宽减半，然后通道数翻倍
        self.down3 = Down(base_c * 4, base_c * 8)
        # 编码器的第5个双卷积层，先进行下采样最大池化使得特征图高和宽减半，然后通道数翻倍
        self.down4 = Down(base_c * 8, base_c * 16)

        # 解码器的第1个上采样模块，首先进行一个转置卷积，使特征图的高和宽翻倍，通道数减半；
        # 对x1（x1可以到总的forward函数中可以知道它代指什么）进行padding，使其与x2的尺寸一致，然后在第1维通道维度进行concat，通道数翻倍。
        # 最后再进行一个双卷积层，通道数减半，高和宽不变。
        self.up1 = Up(base_c * 16, base_c * 8)
        # 解码器的第2个上采样模块，操作同上
        self.up2 = Up(base_c * 8, base_c * 4)
        # 解码器的第3个上采样模块，操作同上
        self.up3 = Up(base_c * 4, base_c * 2)
        # 解码器的第4个上采样模块，操作同上
        self.up4 = Up(base_c * 2, base_c)
        # 解码器的输出卷积模块，改变输出的通道数为分类的类别数
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 假设输入的特征图尺寸为[N, C, H, W]，[4, 3, 480, 480],依次代表BatchSize, 通道数量，高，宽；   则输出为[4, 64, 480,480]
        x1 = self.in_conv(x)
        # 输入的特征图尺寸为[4, 64, 480, 480];  输出为[4, 128, 240,240]
        x2 = self.down1(x1)
        # 输入的特征图尺寸为[4, 128, 240,240];  输出为[4, 256, 120,120]
        x3 = self.down2(x2)
        # 输入的特征图尺寸为[4, 256, 120,120];  输出为[4, 512, 60,60]
        x4 = self.down3(x3)
        # 输入的特征图尺寸为[4, 512, 60,60];  输出为[4, 1024, 30,30]
        x5 = self.down4(x4)

        # 输入的特征图尺寸为[4, 1024, 30,30];  输出为[4, 512, 60, 60]
        x = self.up1(x5, x4)
        # 输入的特征图尺寸为[4, 512, 60,60];  输出为[4, 256, 120, 120]
        x = self.up2(x, x3)
        # 输入的特征图尺寸为[4, 256, 120,120];  输出为[4, 128, 240, 240]
        x = self.up3(x, x2)
        # 输入的特征图尺寸为[4, 128, 240,240];  输出为[4, 64, 480, 480]
        x = self.up4(x, x1)
        # 输入的特征图尺寸为[4, 64, 480,480];  输出为[4, 2, 480, 480]
        logits = self.out_conv(x)

        return {"out": logits}

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    x = torch.rand(1, 3, 256, 256)
    net = UNet(in_channels=3, num_classes=2)
    y = net(x)
    flops = FlopCountAnalysis(net,x)
    print('flops:',flops.total())    # 48331489280，即4.8B
    # print('params:',parameter_count_table(net))
    print("params: ", sum(p.numel() for p in net.parameters()))   # 31037698, 即3.1M
    # print(y.shape)
