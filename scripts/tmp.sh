# if [ ! -d "./logs_test" ]; then
#     mkdir ./logs_test
# fi
 
# model_name=informer

# ### M

# # python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3 > logs_test/$model_name'_'M_ETTh1_predlen_24_seqlen_48_labellen_48.log

# # python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs_test/$model_name'_'M_ETTh1_predlen_48_seqlen_96_labellen_48.log

# # python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs_test/$model_name'_'M_ETTh1_predlen_168_seqlen_168_labellen_168.log

# python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --gpu 1 > logs_test/$model_name'_'M_ETTh1_predlen_336_seqlen_168_labellen_168.log

# python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --gpu 1 > logs_test/$model_name'_'M_ETTh1_predlen_720_seqlen_336_labellen_336.log


python -u main_informer.py --model informer --data WTH --features M --attn prob --d_layers 2 --e_layers 3 --label_len 168 --pred_len 720 --seq_len 168 --des 'Exp'  --itr 1  --enc_in 21   --dec_in 21   --c_out 21 --gpu 1 --run_test
