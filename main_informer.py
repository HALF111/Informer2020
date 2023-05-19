import argparse
import os
import torch

from exp.exp_informer import Exp_Informer
from exp.exp_informer_test import Exp_Informer_Test

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')  # target表示我们希望将数据中的哪一列拿来当作标签使用
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 因为在Informer中，在decoder中的pred_len之前会有一个label_len作为先验的知识，用来带一下pred中的值，从而更加方便预测。所以会多出来一个label_len
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# enc_in可以理解为输入数据的维度为7
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')  # 输入给encoder的特征维度数
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')  # 输入给decoder的特征维度数
parser.add_argument('--c_out', type=int, default=7, help='output size')  # c_out应该是需要预测的输出的特征维度数？
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')  # d_model也即各个隐藏层（如Embedding层）的特征维度
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')  # 一个encoder中包含几层multi-head注意力的层数
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')  # 一个decoder中包含几层multi-head注意力的层数
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')  # 这个是整个encoder模块层需要堆叠的层数？
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')  # FCN中的中间层的维度
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')  # ProbAttention中的对Q采样的因子
parser.add_argument('--padding', type=int, default=0, help='padding type')  
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)  # 是否使用distill，在Encoder中将样本的长度减半（默认为使用，加上参数后变为不使用）
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')  # attention类型
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')  # 是否输出attention（可用于可视化）
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')  # 是否需要做预测
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)  # mix attention？
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')  # 我们要读哪几栏数据
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')  # Windows必须为0，Linux可以设置多个
parser.add_argument('--itr', type=int, default=2, help='experiments times')  # 这个是比epoch更高的一个参数（例如：epoch为20，itr为2，那么会做完20个epoch的训练之后，再做一遍20个epoch的训练）
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')  # 用于EarlyStopping策略的一个参数
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')  # 默认loss函数为“mse”
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')  # 调整学习率，默认为type1（详见~/utils/tools.py介绍）
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)  # automatic mixed precision，也叫自动混合精度。和分布式有关？？？
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)  #反转输出结果？默认为False

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


# test_train_num
parser.add_argument('--test_train_num', type=int, default=10, help='how many samples to be trained during test')
parser.add_argument('--adapted_lr_times', type=float, default=1, help='the times of lr during adapted')  # adaptation时的lr是原来的lr的几倍？
parser.add_argument('--adapted_batch_size', type=int, default=32, help='the batch_size for adaptation use')  # adaptation时的数据集取的batch_size设置为多大
parser.add_argument('--test_train_epochs', type=int, default=1, help='the batch_size for adaptation use')  # adaptation时的数据集取的batch_size设置为多大
parser.add_argument('--run_train', action='store_true')
parser.add_argument('--run_test', action='store_true')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# 如果有多个GPU，那么获取其中的GPU:0
if args.use_gpu and args.use_multi_gpu: 
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# 所有数据集的集合&配置
# 'data'表示对应的数据
# 'T'表示target，也即我们希望将哪一列拿来作为标签
# 'M': multivariate predict multivariate, 'S':univariate predict univariate, 'MS':multivariate predict univariate
# [7,7,7]分别对应于args.enc_in, args.dec_in, args.c_out三个
# 分别表示输入给encoder的特征维度数、输入给decoder的特征维度数、和输出的特征维度数
# 之所以是7，是因为例如ETT数据集中包含date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT共8个参数；但是我们现在把OT拿来当作标签了，所以我们还剩下7个作为给输入的特征。
data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
}

# 将我们所需的数据集信息从data_parser中提取出来，并放入data_path、target、enc_in、dec_in和c_out等参数中
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)


# Exp = Exp_Informer
Exp = Exp_Informer_Test

for ii in range(args.itr):
    print(f"-------Start iteration {ii+1}--------------------------")

    # setting record of experiments
    # 别忘记加上test_train_num一项！！！
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args)  # set experiments
    
    if args.run_train:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

    if args.run_test:
        print('>>>>>>>normal testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting, flag="test")
        exp.test(setting, test=1, flag="test")
    
    # print('>>>>>>>normal testing but batch_size is 1 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting, test=1, flag="test_with_batchsize_1")

    # # 对整个模型进行fine-tuning
    # print('>>>>>>>my testing with all parameters trained : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.my_test(setting, is_training_part_params=False, use_adapted_model=True, test_train_epochs=3)

    # 只对最后的全连接层projection层进行fine-tuning
    print('>>>>>>>my testing with test-time training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.my_test(setting, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1)
    exp.my_test(setting, test=1, is_training_part_params=True, use_adapted_model=True, test_train_epochs=args.test_train_epochs)

    # print('>>>>>>>my testing but with original model : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.my_test(setting, is_training_part_params=True, use_adapted_model=False)


    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
    print()
