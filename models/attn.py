import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # 此时的维度为[32, 72, 8, 64] -> 本来应该是[32, 72, 512]，但是在AttetionLayer层中做了multi-head的多头、以及每个头里的降维操作
        _, S, _, D = values.shape  # 此时的维度为[32, 48, 8, 64] -> 这里的48是因为encoder中每经过一层长度就会减少一半，所以开始的86变成了现在的48
        scale = self.scale or 1./sqrt(E)

        # torch.einsum简介：https://zhuanlan.zhihu.com/p/434232512
        # 也可以参考：https://zhuanlan.zhihu.com/p/547157325
        # 这里就是将blhe的queries和bshe的keys，在第长度的维度（第二维）上做点积
        # 按照原来的方法，需要先转置、再做matmul、再转置回来；而现在一个einsum函数直接解决问题了
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # 如果mask_flag为True，那么需要做个mask
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 计算好Q和K的点积之后，先除以scale，再在最后一维上计算softmax，
        # 再经过一个dropout，便得到了注意力权重矩阵A
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # 然后再用注意力权重A和值values一起计算，得到最终的注意力值V
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        # 再对V做contiguous之后再返回
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

# ProbAttention相当于不使用所有的Queries，而是从其中选出一些代表来参与注意力的计算，从而减少了计算量
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()

        self.factor = factor  # factor就是ProbAttention中对Q采样的因子，为可调的超参数
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention

        # PS：在ProbAttention中，其实是用不到dropout层的；只有FullAttention中才会用到
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q 和 K 的维度均为 [B, H, L, D]
        # 注意这里传入的Q和K都是已经transpose之后的向量了，所以H和L_K/L_Q的位置已经交换过了；
        # 另外，最后一维的E仍然是d_model//n_heads（如本例中即为：512//8 = 64）
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # 在K（keys）的H维和L_K维之间插入一个新的维度，并将其扩展成L_Q的大小；因此K_expand变成了五维向量
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # torch.randint参考：https://runebook.dev/zh-CN/docs/pytorch/generated/torch.randint
        # 这里的randint相当于是生成一个大小为[L_Q, sample_k]的tensor，tensor中的每个值都是在0-L_K之间的值
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part * L_q，其中U_part = factor * ln(L_k)

        # 然后，我们根据index_sample的值，在K_expand中随机挑选出smaple_k个keys来评价query的好坏
        # 下一句代码的操作其实是针对L_Q维的每一个query，用index_sample中的每行的25个值去随机选取对应的keys；（所以这里针对每个query所随机采样的keys都是不一样的！！！）
        # 所以假设K_expand的维度为[32, 8, 96, 96, 64]，那么K_sample则为[32, 8, 96, 25, 64]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # 然后就是计算Q和K_sample之间的值了，这里用的是直接矩阵乘法（点积注意力？？）（用Q * K^T来计算？所以要对K_smaple做transpose）
        # torch.matmul多维下的计算方法：https://blog.csdn.net/qsmx666/article/details/105783610
        # 所以Q.unsqueeze(-2)为[32, 8, 96, 1, 64]（加了一维），K_sample.transpose(-2, -1)为[32, 8, 96, 64, 25]，做矩阵乘法后为[32, 8, 96, 1, 25]，再squeeze之后就是[32, 8, 96, 25]
        Q_K_sample = torch.matmul( Q.unsqueeze(-2), K_sample.transpose(-2, -1) ).squeeze(-2)

        # find the Top_k query with sparisty measurement
        # 我们评价query的好坏指标是：针对当前query和那25个key做计算之后，得到的25个值中的最大值和这25个值的平均值，他们之间的差作为评判标准；这个差值越大，我们就认为这个query波动越大，特征也就越鲜明，也就更要选有特点的他
        # 公式为 $$max_{j}{q_i * k_j^T / sqrt(d)} - 1/L_K * sum_{j=1}^{L_K}{q_i * k_j^T / sqrt(d)}$$
        # PS：不过本代码实现中为了计算方便，不在这里计算attention的时候就带上sqrt(d)；而是在forward函数中等attention计算完成之后，再统一在scores_top的所有值上都除以这个sqrt(d)
        # PS：torch.max参考：https://blog.csdn.net/Z_lbj/article/details/79766690，可得Q_K_sample.max(-1)[0]为获得每一行的最大值们，其维度为[32, 8, 96]；然后torch.sum(-1)的维度也是[32, 8, 96]
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # PS：torch.topk函数解析：https://blog.csdn.net/qq_34914551/article/details/103738160。然后topk也会返回values和indices两个内容，我们则获取indices的那个即可
        # M的shape为[32, 8, 96]，而M_top的shape则变成了[32, 8, 25]，选出了其中最大的n_top个
        M_top = M.topk(n_top, sorted=False)[1]

        # select the reduced Q_reduce from original Q
        # X[:, None, None]表示在原来的tensor再加上两维，例如加入原来的X的维度为[10]，那么X[:, None, None]的维度则为[10, 1, 1]；同理，X[None, :, None]就会变成[1, 10, 1]
        # 所以这里就是在Q的L_Q维上取出最大的那些queries，并放进Q_reduce中（一共取出Q中的n_top个出来）
        # 所以输入时的Q的维度为[32, 8, 96, 64]，而Q_reduce的大小会变成[32, 8, 25, 64]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor * ln(L_q)
        
        # use the reduced Q to calculate Q_K
        # 最后将选取过后的包含n_top个项的Q_reduce，和*所有的*Keys来做计算，并得到最终的自注意力值Q_K
        # 由于Q_reduce为[32, 8, 25, 64]，K为[32, 8, 96, 64]、转置后K.transpose(-2, -1)为[32, 8, 64, 96]；二者做matmul之后为[32, 8, 25, 96]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor * ln(L_q) * L_k

        # 返回注意力权重值Q_K，以及在Q中的最大的那些queries所对应出现的index位置
        # PS：不过这里的Q_K还没做除以sqrt(d)的scale操作、以及归一化的softmax操作；但是在后面都是会做的。
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape

        # 只有decoder的最底下的第一个自注意力层需要做mask操作，其他的attention均不需要做mask
        if not self.mask_flag:
            # 这里为什么要做mean？？？
            # 我们在96个Q中选出了25个做attention，那么剩下的那些怎么办呢？
            # 因为我们的评判指标是波动程度，那我们干脆让波动程度小的那些queries们直接平庸到底。
            # 所以对那些queries，直接让他们每个位置的权重均为1/96（因为queries长度为96）
            # 这样，这个权重在和V一起做计算时，相当于直接给这些V以相同的权重值，也就等于对所有V做一个平均值。这也就是这里要对V做mean操作的原因。
            # 而之所以对dim=-2做，则是因为这一个维度为L_V维
            # 由于V为[32, 8, 96, 64]，所以V_sum为[32, 8, 64]

            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)

            # 我们又重新在V_sum的-2处插入一维，并扩展到L_Q大小，再做一个clone
            # 也即先对V中的向量做均值，再将这个均值复制L_Q(96)份，所以其大小又回到了[32, 8, 96, 64]
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert(L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        # 同样地，也只有deocder的最底下第一个自注意力层需要做mask操作
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 由于我们刚才计算的scores为原始注意力值，还没做softmax，所以这里需要补做softmax
        # 之所以在这里做，是因为如果是masked attention的话，那么需要先将被masked住的地方的值先变成-np.inf，然后再计算softmax的时候那些masked住的值就会变成0了
        # 因为此时score的大小是[32, 8, 25, 96]，所以是对dim=-1做softmax操作；并且做完后维度不变化
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        # index就是从_prob_QK中计算得到的、Q中的最大的那些queries出现的地方
        # 现在我们需要将这些部分的queries对应的attetion，用“attn”和“V”的乘积来替换掉，也即torch.matmul(attn, V)
        # 因为attn为[32, 8, 25, 96]，V为[32, 8, 96, 64]，所以二者的乘积结果为[32, 8, 25, 64]
        # 虽然context_in为[32, 8, 96, 64]，但是我们只对其中的context_in[..., ..., index, ...]做修改，所以其维度也是[32, 8, 25, 64]，所以维度是对上的
        # 也即指修改了25个queries，剩下的(96-25)个都没有修改，保持原来的V.mean的均值
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        
        # 是否返回注意力值
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape  # B为batch数，L_Q为queries的序列的长度/seq_len，H为head的个数，D为最后一维、一般是d_model//n_heads（如本例中即为512//8 = 64）
        _, L_K, _, _ = keys.shape  # 除了取出L_K作为keys的长度之外，其他参数和上一行是一样的

        # 这里对queries、keys和values做一个transpose；也即将 L_Q/L_K 和 H 这两个维度进行交换
        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # self.factor为ProbAttention中对Q采样的因子，为可调的超参数；在这里设置的factor为5。
        # 然后由于L_K和L_Q都是96，所以ln(96)=4.56，向上取整后为5；再乘以factor之后即为25。
        # 在这里，由于L_k和L_Q相等，所以U_part和u的值是相同的；但是二者的意思完全不同。
        # 1. U_part表示从keys中随机抽样出25个点，用来评估该query的波折程度大小，波折程度大的query表示这个query可能更有意义，更值得被关注。（相当于从96个keys中*随机*选出25个点，来评估query的好坏）
        # 2. 而u则表示，按照上面的方法评估了之后，会给所有的96个Q一个score；我们从按照这个score*从大到小*选出最大的前25个Q，之后计算自注意力时也只用这25个Q来计算了。（相当于从96个queries中*按score大小*、选出最有代表性的query参与注意力的计算，以减少计算量）
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c * ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()       # c * ln(L_q) 

        # 如果计算出来比原来的值还大，那么还是取原来的值
        U_part = U_part if U_part<L_K else L_K  # 或者 U_part = min(U_part, L_K)
        u = u if u<L_Q else L_Q                 # 或者 u = min(u, L_Q)
        
        # 调用上方的_prob_QK函数，计算了被选出的25个Q和全部96个K得到的attention
        # scores_top为注意力权重值Q_K，index为Q中的最大的那些queries所对应出现的index位置
        scores_top, index = self._prob_QK(queries, keys, 
                                            sample_k=U_part, n_top=u) 

        # add scale factor
        # 这里为了计算方便，我们不在计算attention的时候就带上sqrt(d)；
        # 而是等attention计算完成之后，再统一在scores_top的所有值上都除以sqrt(d)
        scale = self.scale or 1.0/sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        
        # get the context
        # 对整个context进行初始化，这里初始化的方式为：对V中的向量做均值，并复制L_Q份，相当于对于那些没被选中的(96-25)个Q我们假设他们对每个V的权重都是一样的
        # 但是这里对所有96个Q都采用了这样的初始化，所以接下来的_update_context函数中我们要把那25个Q重新改成我们在_prob_QK中计算出的attention值
        context = self._get_initial_context(values, L_Q)

        # update the context with selected top_k queries
        # _update_context函数中,我们要把那25个Q重新改成我们在_prob_QK中计算出的attention值
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        # 对context再做一次transpose，将 L_V 和 H 这两个维度再换回来，重新变成了[32, 96, 8, 64]或者说[B, L_V, H, D]
        # 然后再调用contiguous对当前tensor做一次强制拷贝，可以参考：https://blog.csdn.net/kdongyi/article/details/108180250
        # 将context和attn一起返回
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        # 如果d_keys和d_values的长度未指定的话，那么就把 d_model // n_heads 作为他们的长度
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape  # B为batch数，L为序列的长度/seq_len
        _, S, _ = keys.shape  # S为keys或values的个数？
        H = self.n_heads  # H为多头注意力的head的个数

        # 将QKV映射到对应的d_q、d_k、d_v维度上，并映射h次
        # 也即：对各个输入分别作用共h组的W^q、W^k、W^v三个权重参数后，得到了queries、keys和values三个向量
        # 然后将向量reshape为[bacth个数, 序列长度, head个数, d_keys或d_values]的一个四维的向量
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 计算得到注意力结果
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        if self.mix:
            out = out.transpose(2,1).contiguous()
        
        # 计算完注意力后的out的大小为[B, L, H, D]，然后再做view之后就又把多个头的注意力合并回了；变成一个三维的[B, L, d_values * n_heads]了
        out = out.view(B, L, -1)

        # 最后还有一个线性层：将多个头的输出concat后，映射回d_model维度。
        # 也即：将最后一维的d_values * n_heads维度重新映射回d_model？ 变成[B, L, d_model]了
        out = self.out_projection(out)

        return out, attn
