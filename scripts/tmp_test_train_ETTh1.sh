if [ ! -d "./logs_test_train_updated" ]; then
    mkdir ./logs_test_train_updated
fi
 
model_name=informer

### M

# for test_train_num in 1 2 4 6 8 10 20 30 50
for test_train_num in 5
do

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2 --factor 3 --gpu 1 --test_train_num $test_train_num > logs_test_train_updated/$model_name'_'M_ETTh1_predlen_24_seqlen_48_labellen_48_ttn_$test_train_num.log

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2 --gpu 1 --test_train_num $test_train_num > logs_test_train_updated/$model_name'_'M_ETTh1_predlen_48_seqlen_96_labellen_48_ttn_$test_train_num.log

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2 --gpu 1 --test_train_num $test_train_num > logs_test_train_updated/$model_name'_'M_ETTh1_predlen_168_seqlen_168_labellen_168_ttn_$test_train_num.log

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2 --gpu 1 --test_train_num $test_train_num > logs_test_train_updated/$model_name'_'M_ETTh1_predlen_336_seqlen_168_labellen_168_ttn_$test_train_num.log

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 2 --gpu 1 --test_train_num $test_train_num > logs_test_train_updated/$model_name'_'M_ETTh1_predlen_720_seqlen_336_labellen_336_ttn_$test_train_num.log

done