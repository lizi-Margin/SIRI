import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq, c, h, w) -> treat seq as batch dim
        orig_shape = x.shape
        x = x.view(-1, *x.shape[2:])  # (batch*seq, c, h, w)
        
        avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        out = avg_out + max_out
        scale = self.sigmoid(out).unsqueeze(-1).unsqueeze(-1)
        
        scale = scale.view(*orig_shape[:2], *scale.shape[1:])  # (batch, seq, c, 1, 1)
        return scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq, c, h, w)
        orig_shape = x.shape
        x = x.view(-1, *x.shape[2:])  # (batch*seq, c, h, w)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        scale = self.sigmoid(x)
        
        scale = scale.view(*orig_shape[:2], *scale.shape[1:])  # (batch, seq, 1, h, w)
        return scale

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        # x: (batch, seq, c, h, w)
        channel_scale = self.channel_att(x)
        x = x * channel_scale
        
        spatial_scale = self.spatial_att(x)
        x = x * spatial_scale
        return x

class CrossAttention(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.query_conv = nn.Conv2d(c1, c1, 1)
        self.key_conv = nn.Conv2d(c2, c2, 1)
        self.value_conv = nn.Conv2d(c2, c2, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, img_feat, depth_feat):
        batch, seq, c1, h, w = img_feat.shape
        batch, seq, c2, h, w = depth_feat.shape
        
        # Flatten seq into batch dim
        img_feat = img_feat.view(-1, c1, h, w)
        depth_feat = depth_feat.view(-1, c2, h, w)
        
        # Project features
        query = self.query_conv(img_feat).view(-1, c1, h*w) # (batch*seq, c1, h*w)
        key = self.key_conv(depth_feat).view(-1, c2, h*w).permute(0, 2, 1)  # (batch*seq, h*w, c2)
        value = self.value_conv(depth_feat).view(-1, c2, h*w)  # (batch*seq, c2, h*w)

        original_dtype = query.dtype
        with torch.autocast("cuda", enabled=False): 
            query = query.to(torch.float32)
            key = key.to(torch.float32)
            value = value.to(torch.float32)
            
            # Compute attention
            attention = torch.bmm(query, key)  # (batch*seq, c1, c2)

            # log_attention = torch.nn.functional.log_softmax(attention, dim=-1)
            # attention = torch.exp(log_attention)
            attention = self.softmax(attention)
            
            # Apply attention
            out = torch.bmm(attention, value)  # (batch*seq, c1, h*w)
            out = out.view(-1, c1, h, w)  # (batch*seq, c1, h, w)

            out = out.to(original_dtype)
        
        # Residual connection
        out = self.gamma * out + img_feat
        
        # Reshape back
        out = out.view(batch, seq, c1, h, w)
        return out

class HWC_SpatialAttention(nn.Module):
    def __init__(self, c1, c2, c):
        super().__init__()
        self.c = c
        self.query = nn.Linear(c1, c)  # 计算query
        self.key = nn.Linear(c2, c)    # 计算key
        self.value = nn.Linear(c2, c)  # 计算value

    def forward(self, img_feat, depth_feat):
        batch, seq, _, h, w = img_feat.shape
        img = img_feat.permute(0,1,3,4,2)  # [b,s,h,w,c1]
        depth = depth_feat.permute(0,1,3,4,2)  # [b,s,h,w,c2]

        # 展平空间维度 [b,s,h*w,c]
        q = self.query(img).view(batch, seq, h*w, -1)
        k = self.key(depth).view(batch, seq, h*w, -1)
        v = self.value(depth).view(batch, seq, h*w, -1)


        original_dtype = q.dtype
        with torch.autocast("cuda", enabled=False): 
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            # 计算空间注意力 [b,s,h*w,h*w]
            attn = torch.softmax((q @ k.transpose(-1,-2)) / (self.c**0.5), dim=-1)
            
            # 加权求和 [b,s,h*w,c1] -> [b,s,h,w,c]
            out = (attn @ v).view(batch, seq, h, w, -1)
            out = out + img  # 残差连接
            out = out.to(original_dtype)
        return out.permute(0,1,4,2,3)  # [b,s,c,h,w]

class FeatureFusionModule(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(FeatureFusionModule, self).__init__()
        
        # Cross attention between modalities
        self.cross_att = CrossAttention(channels, channels)
        
        # Channel and spatial attention (CBAM)
        self.cbam = CBAM(channels, reduction_ratio)
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2*channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, img_feat, depth_feat):
        # img_feat, depth_feat: (batch, seq, c, h, w)
        
        # Cross attention between image and depth features
        img_att = self.cross_att(img_feat, depth_feat)  # img attended by depth
        depth_att = self.cross_att(depth_feat, img_feat)  # depth attended by img
        
        # Concatenate features
        fused = torch.cat([img_att, depth_att], dim=2)  # (batch, seq, 2*c, h, w)
        
        # Flatten seq into batch dim for 2D operations
        batch, seq, c, h, w = fused.shape
        fused = fused.view(-1, c, h, w)  # (batch*seq, 2*c, h, w)
        
        # Fusion convolution
        fused = self.fusion_conv(fused)  # (batch*seq, c, h, w)
        
        # Apply CBAM attention
        fused = fused.view(batch, seq, *fused.shape[1:])  # (batch, seq, c, h, w)
        fused = self.cbam(fused)
        
        return fused