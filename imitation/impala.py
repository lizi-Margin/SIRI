from torch import nn
import torch
import torch.nn.functional as F

class ResidualDownBlock(nn.Module):
    def __init__(self, in_chns, h, w, out_chns, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chns, out_chns, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_chns, out_chns, kernel_size=3, stride=1, padding=1)
        self.skip = nn.Conv2d(in_chns, out_chns, kernel_size=1, stride=stride) if in_chns != out_chns else nn.Identity()
        # self.norm = nn.LayerNorm([out_chns, h, w])
        self.norm = nn.BatchNorm2d(out_chns)
        
    def forward(self, x):
        residual = self.skip(x)
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        return self.norm(x + residual)

class ImpalaStyleEncoder(nn.Module):
    def __init__(self, in_shape, chns=[32, 64, 128], hidsize=256):
        """
        in_shape: (channels, height, width)
        chns: 各阶段的输出通道数
        hidsize: 最终输出的隐藏层大小
        """
        super().__init__()
        c, h, w = in_shape
        self.blocks = nn.ModuleList()
        
        # 构建降采样块
        for i, out_chns in enumerate(chns):
            stride = 2 if i > 0 else 1  # 第一层不降采样
            self.blocks.append(ResidualDownBlock(c if i==0 else chns[i-1], h, w, out_chns, stride))
        
        # 计算最终特征图尺寸
        with torch.no_grad():
            x = torch.zeros(1, *in_shape)
            for block in self.blocks:
                x = block(x)
            _, _, fh, fw = x.shape
        
        # 展平层
        self.flatten_size = chns[-1] * fh * fw
        self.dense = nn.Sequential(
            nn.Linear(self.flatten_size, hidsize),
            nn.LayerNorm(hidsize),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: (seq_len, c, h, w)
        seq_len = x.size(0)
        x = x.reshape(-1, *x.shape[1:])  # 合并batch和seq_len
        
        for block in self.blocks:
            x = block(x)
        
        x = x.reshape(seq_len, -1)  # (seq_len, chns[-1]*fh*fw)
        return self.dense(x)  # (seq_len, hidsize)