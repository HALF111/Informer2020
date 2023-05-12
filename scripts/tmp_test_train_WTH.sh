if [ ! -d "./logs_test_train_WTH" ]; then
    mkdir ./logs_test_train_WTH
fi
 
model_name=informer

### M

# for test_train_num in 1 2 4 6 8 10 20 30 50
for test_train_num in 5
do

python -u main_informer.py --model $model_name --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --label_len 168 --pred_len 24 --seq_len 168 --des 'Exp' --itr 2 --gpu 1 --test_train_num $test_train_num > logs_test_train_WTH/$model_name'_'M_WTH_predlen_24_seqlen_168_labellen_168_ttn_$test_train_num.log

python -u main_informer.py --model $model_name --data WTH --features M --attn prob --d_layers 1 --e_layers 2 --label_len 96 --pred_len 48 --seq_len 96 --des 'Exp' --itr 2 --gpu 1 --test_train_num $test_train_num > logs_test_train_WTH/$model_name'_'M_WTH_predlen_48_seqlen_96_labellen_96_ttn_$test_train_num.log

python -u main_informer.py --model $model_name --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --label_len 168 --pred_len 168 --seq_len 336 --des 'Exp' --itr 2 --gpu 1 --test_train_num $test_train_num > logs_test_train_WTH/$model_name'_'M_WTH_predlen_168_seqlen_336_labellen_168_ttn_$test_train_num.log

python -u main_informer.py --model $model_name --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --label_len 168 --pred_len 336 --seq_len 720 --des 'Exp' --itr 2 --gpu 1 --test_train_num $test_train_num > logs_test_train_WTH/$model_name'_'M_WTH_predlen_336_seqlen_720_labellen_168_ttn_$test_train_num.log

python -u main_informer.py --model $model_name --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --label_len 336 --pred_len 720 --seq_len 720 --des 'Exp' --itr 2 --gpu 1 --test_train_num $test_train_num > logs_test_train_WTH/$model_name'_'M_WTH_predlen_720_seqlen_720_labellen_336_ttn_$test_train_num.log

done