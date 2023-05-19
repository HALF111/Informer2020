from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from data.data_loader import Dataset_Custom_Test, Dataset_ETT_hour_Test
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import copy

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer_Test(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer_Test, self).__init__(args)
        
        # 这个可以作为超参数来设置
        self.test_train_num = self.args.test_train_num
    

    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        # 因为不同的csv数据，需要用不同的格式来处理（例如：有针对小时的，有针对分钟的，还有自定义的）
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,

            'exchange_rate': Dataset_Custom,
            'traffic': Dataset_Custom,
            'national_illness': Dataset_Custom,
        }

        # Data获取用于加载数据的dataset类
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        # 根据test/pred/train来设置相关参数
        # 这里shuffle不是将时序数据完全打乱，而是随机从序列中取出连续的一段出来
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag == 'test_with_batchsize_1':
            flag = "test"
            shuffle_flag = False; drop_last = True; batch_size = 1; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        
        # 定义我们的数据集实例
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        
        # 生成DataLoader
        # PS：DataLoader详解：https://www.zdaiot.com/MLFrameworks/Pytorch/Pytorch%20DataLoader%E8%AF%A6%E8%A7%A3/
        # 或者：https://zhuanlan.zhihu.com/p/381224748
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader


    def _get_data_at_test_time(self, flag):
        args = self.args

        timeenc = 0 if args.embed!='timeF' else 1

        # 根据test/pred/train来设置相关参数
        # 这里shuffle不是将时序数据完全打乱，而是随机从序列中取出连续的一段出来
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq

            # 注意：因为我们要做TTT/TTA，所以一定要把batch_size设置成1 ！！！
            batch_size = 32
            # batch_size = 1
            # batch_size = 4
            # batch_size = 2857

            # Data = Dataset_Custom_Test
            dataset_name = self.args.data
            if dataset_name == "ETTh1" or dataset_name == "ETTh2":
                Data = Dataset_ETT_hour_Test
            else:
                Data = Dataset_Custom_Test

            # 定义我们的数据集实例
            data_set = Data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                inverse=args.inverse,
                timeenc=timeenc,
                freq=freq,
                cols=args.cols,
                test_train_num = self.test_train_num
            )
        
        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        # vali或test时，需要将模型从train模式变成eval模式
        self.model.eval()
        total_loss = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            # 这一步和train一样，也是调用model在数据上跑（但是此时已经是eval模式、而不是train模式了）
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            # vali时，我们的loss都统一放回到cpu上计算
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        
        # 计算最后的平均loss
        total_loss = np.average(total_loss)
        # 模型重新变回train模式
        self.model.train()
        
        # 返回loss
        return total_loss

    def train(self, setting):
        # 获取train/val/test的data和DataLoader
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)  # 在checkpoints目录下新建文件
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)  # 注：len(train_data) / len(train_loader)的值基本等于我们一开始设置的batch_size的值
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 提前停止策略（如果模型连续多少次不更新，那么就提前停止训练）
        
        model_optim = self._select_optimizer()  # 使用Adam优化器
        criterion =  self._select_criterion()  # 使用MSELoss

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        for epoch in range(self.args.train_epochs):  # 跑train_epochs个epoch（默认为6）
            iter_count = 0
            train_loss = []
            
            self.model.train()  # 将模型设置为train模式
            epoch_time = time.time()

            # 遍历DataLoader时会去跑他的__getitem__函数
            # PS：取数据时一次会取一整个batch，所以这里的batch_x的维度为[batch_size, seq_len, feature_numbers]，如[32, 96, 12]
            # 类似地，batch_y的维度为[batch_size, label_len + pred_len, feature_numbers]，如[32, 48+24, 12]
            # batch_x_mark和batch_y_mark的前两维分别和batch_x、batch_y一样，最后一维则是时间维度的特征，例如[32, 96, 4]和[32, 48+24, 4]
            # 每个epoch里会遍历一遍所有的train数据
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()  # 先梯度清零
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)  # 生成模型的预测值和实际的真实值
                
                # 计算在训练集熵的loss
                loss = criterion(pred, true)  # 用MSE来计算loss
                train_loss.append(loss.item())  # 将每一次计算的loss存在数组中
                
                # 每过100个batch的数据后，就打印一次loss和speed信息
                if (i+1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ( (self.args.train_epochs - epoch) * train_steps - i )
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                # 反向传播，更新loss、optimizer和参数
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # 打印时间信息
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            # 计算train、val和test的loss
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)  # 调用self.vali函数计算val集的loss
            test_loss = self.vali(test_data, test_loader, criterion)  # 调用self.vali函数计算test集的loss

            # 打印loss信息
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # 如果能early_stopping，那么提前结束训练
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # 调整学习率
            adjust_learning_rate(model_optim, epoch+1, self.args)
        
        # 结束所有epoch的训练之后，我们将权重文件保存下来
        best_model_path = path + '/' + 'checkpoint.pth'

        # 这里为了防止异常，需要做一些修改，要在torch.load后加上map_location='cuda:0'
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))
        
        # 返回训练后的模型
        return self.model
    

    def test(self, setting, test=0, flag='test'):
        test_data, test_loader = self._get_data(flag=flag)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        
        self.model.eval()
        
        preds = []
        trues = []

        test_time_start = time.time()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        # 这一步是将preds和trues的前两维给合并成一维
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 这里分别计算了mae、mse、rmse、mape、mspe等个参数，且全都是自己实现的，在metrics.py文件里面
        # 而训练时采用的loss则是nn自带的MSELoss函数
        # 我们这边计算MSE则是使用np.mean((pred-true)**2)
        # 但实际上，这二者除了一个传入的是tensor、另一个传入的是np.array之外，并没有太大区别
        # 然后我们自己实现的方法其实也正好对应于在nn.MSELoss中使用reduction = 'mean'的结果
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        return


    def my_test(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))


        # test_data, test_loader = self._get_data(flag='test')

        # path = os.path.join(self.args.checkpoints, setting)  # 在checkpoints目录下新建文件
        # model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(model_path))

        # 加载原模型，并设置成eval模式
        # original_model = copy.deepcopy(self.model)
        # original_model.eval()
        
        self.model.eval()
        
        preds = []
        trues = []

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate/10)  # 使用Adam优化器
        criterion = nn.MSELoss()  # 使用MSELoss
        test_time_start = time.time()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # 从self.model拷贝下来cur_model，并设置为train模式
            cur_model = copy.deepcopy(self.model)
            cur_model.train()

            if is_training_part_params:
                params = []
                names = []
                cur_model.requires_grad_(False)
                for n_m, m in cur_model.named_modules():
                    if "projection" == n_m:
                        m.requires_grad_(True)
                        for n_p, p in m.named_parameters():
                            if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                                params.append(p)
                                names.append(f"{n_m}.{n_p}")


                # Adam优化器
                # model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # model_optim_norm = optim.Adam(params_norm, lr=self.args.learning_rate*1000 / (2**self.test_train_num))  # 使用Adam优化器
                
                # 普通的SGD优化器？
                model_optim = optim.SGD(params, lr=lr)
            else:
                self.model.requires_grad_(True)
                model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))


            # 开始训练
            # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
            for epoch in range(test_train_epochs):
                # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)

                import random
                is_random = False
                # is_random = True
                sample_order_list = list(range(self.test_train_num))
                # print("before random, sample_order_list is: ", sample_order_list)
                if is_random:
                    random.shuffle(sample_order_list)
                    # print("after random, sample_order_list is: ", sample_order_list)
                else:
                    # print("do not use random.")
                    pass


                for ii in sample_order_list:
                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    # if i == 3461:
                    #     pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    #         batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    #         batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])


                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])


                    # from functorch import vmap
                    # from functorch.experimental import replace_all_batch_norm_modules_
                    # replace_all_batch_norm_modules_(cur_model)
                    # tmp_batch_x = batch_x.unsqueeze(1)
                    # tmp_batch_x_mark = batch_x_mark.unsqueeze(1)
                    # vmap_func = vmap(self._process_one_batch_with_model, 
                    #                  in_dims=(None, None, 0, 0, 0, 0), out_dims=(0, 0), 
                    #                  randomness='different')
                    # pred, true = vmap_func(cur_model, test_data,
                    #     tmp_batch_x[:, :, ii : ii+seq_len, :], tmp_batch_x[:, :, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    #     tmp_batch_x_mark[:, :, ii : ii+seq_len, :], tmp_batch_x_mark[:, :, ii+seq_len-label_len : ii+seq_len+pred_len, :])

                    loss = criterion(pred, true)
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()
                    
                    # # cur_lr = cur_lr * 0.5
                    # cur_lr = cur_lr * 2
                    # for param_group in model_optim.param_groups:
                    #     param_group['lr'] = cur_lr
            

            # 完成test-time training，进入eval模式
            cur_model.eval()

            if use_adapted_model:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            else:
                pred, true = self._process_one_batch_with_model(self.model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            
            # # self.model.eval()
            # pred, true = self._process_one_batch(test_data,
            #     batch_x[:, -self.args.seq_len:, :], batch_y, 
            #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            if (i+1) % 100 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
            
            # if i > 3400:
            #     print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        # 这一步是将preds和trues的前两维给合并成一维
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 这里分别计算了mae、mse、rmse、mape、mspe等个参数，且全都是自己实现的，在metrics.py文件里面
        # 而训练时采用的loss则是nn自带的MSELoss函数
        # 我们这边计算MSE则是使用np.mean((pred-true)**2)
        # 但实际上，这二者除了一个传入的是tensor、另一个传入的是np.array之外，并没有太大区别
        # 然后我们自己实现的方法其实也正好对应于在nn.MSELoss中使用reduction = 'mean'的结果
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        return



    def subprocess_of_my_test_vmap(self, setting, is_training_part_params, use_adapted_model, test_train_epochs,
                                    test_data, batch_x, batch_x_mark, batch_y, batch_y_mark, 
                                    test_time_start, criterion, scaler, i, preds, trues,
                                    model, model_optim):
        # # 从self.model拷贝下来cur_model，并设置为train模式
        # cur_model = copy.deepcopy(self.model)
        # cur_model.train()

        # 从model拷贝下来cur_model，并设置为train模式
        cur_model = copy.deepcopy(model)
        cur_model.train()

        # from functorch.experimental import replace_all_batch_norm_modules_
        # replace_all_batch_norm_modules_(cur_model)

        # if is_training_part_params:
        #     params = []
        #     names = []
        #     cur_model.requires_grad_(False)
        #     for n_m, m in cur_model.named_modules():
        #         if "projection" == n_m:
        #             m.requires_grad_(True)
        #             for n_p, p in m.named_parameters():
        #                 if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
        #                     params.append(p)
        #                     names.append(f"{n_m}.{n_p}")


        #     model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
        # else:
        #     self.model.requires_grad_(True)
        #     model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))

        # # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))


        # 开始训练
        cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
        for epoch in range(test_train_epochs):
            cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
            for ii in range(self.test_train_num):
                model_optim.zero_grad()

                seq_len = self.args.seq_len
                label_len = self.args.label_len
                pred_len = self.args.pred_len

                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

                loss = criterion(pred, true)
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    pass
                
                # cur_lr = cur_lr * 0.5
                cur_lr = cur_lr * 2
                for param_group in model_optim.param_groups:
                    param_group['lr'] = cur_lr
        

        # 完成test-time training，进入eval模式
        cur_model.eval()

        if use_adapted_model:
            pred, true = self._process_one_batch_with_model(cur_model, test_data,
                batch_x[:, -self.args.seq_len:, :], batch_y, 
                batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
        else:
            pred, true = self._process_one_batch_with_model(self.model, test_data,
                batch_x[:, -self.args.seq_len:, :], batch_y, 
                batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
        
        # # self.model.eval()
        # pred, true = self._process_one_batch(test_data,
        #     batch_x[:, -self.args.seq_len:, :], batch_y, 
        #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
        
        # tmp_pred = torch._C._functorch.get_unwrapped(pred)[0]
        # tmp_true = torch._C._functorch.get_unwrapped(true)[0]
        # preds.append(tmp_pred.detach().cpu().numpy())
        # trues.append(tmp_true.detach().cpu().numpy())
        # # preds.append(pred.detach().cpu().numpy())
        # # trues.append(true.detach().cpu().numpy())

        if (i+1) % 100 == 0:
            print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
        
        # if i > 3400:
        #     print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))

        cur_model.eval()
        # cur_model.cpu()
        del cur_model
        torch.cuda.empty_cache()

        # return tmp_pred, tmp_true
        return pred, true

    
    # def my_test_vmap(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1):
    #     test_data, test_loader = self._get_data_at_test_time(flag='test')
    #     if test:
    #         print('loading model from checkpoint !!!')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))


    #     # test_data, test_loader = self._get_data(flag='test')

    #     # path = os.path.join(self.args.checkpoints, setting)  # 在checkpoints目录下新建文件
    #     # model_path = path + '/' + 'checkpoint.pth'
    #     # self.model.load_state_dict(torch.load(model_path))

    #     # 加载原模型，并设置成eval模式
    #     # original_model = copy.deepcopy(self.model)
    #     # original_model.eval()
        
    #     self.model.eval()
        
    #     preds = []
    #     trues = []

    #     if self.args.use_amp:
    #         scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？
    #     else:
    #         scaler = None

    #     # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate/10)  # 使用Adam优化器
    #     criterion = nn.MSELoss()  # 使用MSELoss
    #     test_time_start = time.time()
        
    #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #         from functorch import vmap
    #         # from functorch.experimental import replace_all_batch_norm_modules_
    #         # replace_all_batch_norm_modules_(cur_model)

    #         # 从self.model拷贝下来cur_model，并设置为train模式
    #         cur_model = copy.deepcopy(self.model)
    #         cur_model.train()

    #         from functorch.experimental import replace_all_batch_norm_modules_
    #         replace_all_batch_norm_modules_(cur_model)

    #         if is_training_part_params:
    #             params = []
    #             names = []
    #             cur_model.requires_grad_(False)
    #             for n_m, m in cur_model.named_modules():
    #                 if "projection" == n_m:
    #                     m.requires_grad_(True)
    #                     for n_p, p in m.named_parameters():
    #                         if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
    #                             params.append(p)
    #                             names.append(f"{n_m}.{n_p}")

    #             model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
    #         else:
    #             self.model.requires_grad_(True)
    #             model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))

    #         # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))

            
    #         # 这里一定一定不要忘记要给tmp_batch_x到tmp_batch_y_mark四个tensor均做一次unsqueeze(1)，
    #         # 否则后面做vmap的时候的维度会出现问题的
    #         tmp_batch_x = batch_x.unsqueeze(1)
    #         tmp_batch_x_mark = batch_x_mark.unsqueeze(1)
    #         tmp_batch_y = batch_y.unsqueeze(1)
    #         tmp_batch_y_mark = batch_y_mark.unsqueeze(1)
    #         vmap_func = vmap(self.subprocess_of_my_test_vmap, 
    #                          in_dims=(None, None, None, None, None,
    #                                   0, 0, 0, 0,
    #                                   None, None, None, None, None, None,
    #                                   None, None),
    #                          # out_dims=(0, 0), 
    #                          randomness='different')
    #         pred, true = vmap_func(
    #             setting, is_training_part_params, use_adapted_model, test_train_epochs, test_data,
    #             tmp_batch_x, tmp_batch_x_mark, tmp_batch_y, tmp_batch_y_mark,
    #             test_time_start, criterion, scaler, i, preds, trues,
    #             cur_model, model_optim)
            
    #         pred = pred[0]
    #         true = true[0]
    #         preds.append(pred.detach().cpu().numpy())
    #         trues.append(true.detach().cpu().numpy())


    #     preds = np.array(preds)
    #     trues = np.array(trues)
    #     print('test shape:', preds.shape, trues.shape)

    #     # 这一步是将preds和trues的前两维给合并成一维
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)

    #     # result save
    #     folder_path = './results/' + setting +'/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     # 这里分别计算了mae、mse、rmse、mape、mspe等个参数，且全都是自己实现的，在metrics.py文件里面
    #     # 而训练时采用的loss则是nn自带的MSELoss函数
    #     # 我们这边计算MSE则是使用np.mean((pred-true)**2)
    #     # 但实际上，这二者除了一个传入的是tensor、另一个传入的是np.array之外，并没有太大区别
    #     # 然后我们自己实现的方法其实也正好对应于在nn.MSELoss中使用reduction = 'mean'的结果
    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))

    #     # np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     # np.save(folder_path+'pred.npy', preds)
    #     # np.save(folder_path+'true.npy', trues)

    #     print(f"Test - cost time: {time.time() - test_time_start}s")

    #     return



    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # 将batch_x, batch_x_mark, batch_y_mark均送入device中；但是batch_y留在cpu就ok了
        # 但是最后在返回bacth_y的时候，也要将batch_y送入device中的
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        # 即我们在给decoder输入时，前面的label_len长的数据是从已有序列获取的；而后面pred_len长的数据则需要用padding去填充
        # 在Informer方法中，默认是全部都填充成0的，所以padding==0，会用torch.zeros将后半段的pred_len长的数据给填充上
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        
        # 后半段pred_len填充完成后，和前半段的label_len长度的数据concat在一起，从而作为给decoder的输入
        # 或者也可以理解成将batch_y的前半段保留，后半段都变成0或者1来填充了
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        # 这里则是正式地调用model的forward函数进行调用模型了
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            # 注意这里的第三个参数是dec_inp，而不是batch_y了！！！
            # 传进去的几个参数分别为x_enc, x_mark_enc, x_dec, x_mark_dec
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 这里返回的结果为[B, L, D]，例如[32, 24, 12]
        
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        
        f_dim = -1 if self.args.features=='MS' else 0

        # bacth_y其实就是对应的真实值，
        # 但是我们同样也取出后面的pred_len长度的内容，而将前面label_len长度的部分给扔掉
        # 另外，如果是MS/由多变量预测单变量的话，那么f_dim为-1，表示只取最后一维的特征返回
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # outputs为我们预测出的值pred，而batch_y则是对应的真实值true
        return outputs, batch_y

    
    def _process_one_batch_with_model(self, model, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 这里返回的结果为[B, L, D]，例如[32, 24, 12]
        
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        
        f_dim = -1 if self.args.features=='MS' else 0

        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # outputs为我们预测出的值pred，而batch_y则是对应的真实值true
        return outputs, batch_y
    

    def _process_with_vmap(self, model, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([self.args.pred_len, batch_y.shape[-1]]).float()
        
        dec_inp = torch.cat([batch_y[:self.args.label_len, :], dec_inp], dim=0).float().to(self.device)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 这里返回的结果为[B, L, D]，例如[32, 24, 12]
        
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        
        f_dim = -1 if self.args.features=='MS' else 0

        batch_y = batch_y[-self.args.pred_len:, f_dim:].to(self.device)

        # outputs为我们预测出的值pred，而batch_y则是对应的真实值true
        return outputs, batch_y
