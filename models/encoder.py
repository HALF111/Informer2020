import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()

        padding = 1 if torch.__version__>='1.5.0' else 2

        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))  # 交换后两维，但是由于kernel=3、而padding=1，所以维度的数值没有变化
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)  # 由于max_pool的stride=2，所以maxpool的步长为2，这会导致x中的原来的长度96会减少一半变成48了
        x = x.transpose(1,2)  # 后两维再交换回来

        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()

        # d_ff为Encoder中的postional-wise FFN中，两个全连接层里面的隐藏层维度，默认为4*d_model
        d_ff = d_ff or 4*d_model

        self.attention = attention  # 传入的attention一般是一个AttentionLayer类的实例

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))

        # 从大的结构来看，Informer和Transformer中的Encoder部分的结构基本是类似的；
        # 主要区别就是attention计算方法做了替换，其他的残差连接是类似的
        # 也即 attetion -> add&norm -> positional-wise FFN -> add&norm

        # 这里就相当于在做自注意力，所以QKV三者均传入了x
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        new_x = self.dropout(new_x)

        # 这里相当于做第一个add & norm
        x = x + new_x
        x = self.norm1(x)

        # 然后下面也是做和transformer一样的positional-wise FFN，
        # 这其中包括两个全连接层和中间一个ReLU激活层，且全连接层的中间隐藏层维度为d_ff
        # 不过这里用了kernel=1的conv卷积操作来替代全连接层了，但本质上二者是一样的，都是一个全连接/MLP
        # PS：new_x、x和y的维度都是[32, 96, 512]
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))  # 注意y.transpose(-1, 1)交换了后两维，变成了[32, 512, 96]
        y = self.dropout(self.conv2(y).transpose(-1,1))

        # 这里相当于是做第二个add & norm
        x = x + y
        x = self.norm2(x)

        # 返回输出结果x和注意力attn
        return x, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()

        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            # 将attn_layers和conv_layers两两结对，形成许多个pair，并会遍历这些pair。长度按照较短的那个来计算
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)  # 经过attn_layer后，x的维度不变，仍然为[32, 96, 512]（但是随着层数向上，96这一维每过一层就会除2；会不断变成48，24，12……）
                x = conv_layer(x)  # conv_layer就是为了蒸馏操作啊，也即在Informer的Encoder层中，每层的维度是越往上越小的，每过一层都会除2
                attns.append(attn)
            
            # 由于一般attn_layer都要比conv_layer多一层（因为attn_layer层有e_layers个，而conv_layer层只有e_layers-1个），所以还要再用attn_layer[-1]额外做一次
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        # 最后再做一次norm，这里一般是要使用LayerNorm，而不是BatchNorm
        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
