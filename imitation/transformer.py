from torch import nn
import torch
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads=8, max_seq_len=128):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4*input_dim),
            nn.GELU(),
            nn.Linear(4*input_dim, input_dim)
        )
        self.norm2 = nn.LayerNorm(input_dim)
        self.max_seq_len = max_seq_len
        
    def _generate_square_subsequent_mask(self, sz):
        """生成因果掩码，防止看到未来的信息"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(next(self.parameters()).device)
        
    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, _ = x.shape
        
        # 如果没有提供掩码，且序列长度大于1，则生成因果掩码
        if mask is None and seq_len > 1:
            mask = self._generate_square_subsequent_mask(seq_len)
        
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + attn_output
        x = self.norm1(x)
        
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x
    

class TransformerProcessor(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(input_dim, num_heads, max_seq_len) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x