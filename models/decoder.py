import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()

        d_ff = d_ff or 4*d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # 同样是从d_model到d_ff的一维卷积，后面再卷回来
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # Transformer一般用LayerNorm
        # 这里由于有3个子层，所以做3次norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 1. 第一层，先做decoder自己的自注意力
        # 注意：这里一般是带mask的自注意力！！！
        new_x = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0]
        # 别忘记每次add&norm前都要先dropout以下，下面同理
        new_x = self.dropout(new_x)
        # 第一次的add & norm
        x = x + new_x
        x = self.norm1(x)

        # 2. 第二层，是做正常的attention，其中queries来自decoder，而keys和values则来自encoder
        # 但是这里的x和cross的维度是不一样的，x为[32, 72, 512]，而cross则为[32, 48, 512]，所以维度相关信息需要在FullAttention中完成处理
        # 上述操作是没问题的，因为这里的72是query的个数n，48为key-value对的个数m，这二者本来就可以不相同的
        # 也就是Transformer里有两个保证，假设Q为[B, n, d_q]，K为[B, m, d_k]，Q为[B, m, d_v]
        # 要求一个是K和V的个数m必须相同；另一个是d_q和d_k必须相同。
        new_x = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0]
        new_x = self.dropout(new_x)
        # 第二次的add & norm
        x = x + new_x
        x = self.norm2(x)

        # 3. 最后也要做一次 positional-wise FFN（这和Encoder中的操作基本一致）
        # 一维卷积类似于MLP？
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        # 第三次的add & norm
        x = x + y
        x = self.norm3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # self.layers中一共会含有d_layers层的DecoderLayer层
        # 其中每个DecoderLayer层又会包括两个attention层，第一层为self-attention层，第二层为正常的attention层
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        # 这里再做一次norm，一般是LayerNorm
        if self.norm is not None:
            x = self.norm(x)

        return x