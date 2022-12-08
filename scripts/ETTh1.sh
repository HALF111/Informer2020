if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=informer

### M

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 48 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 --factor 3 > logs/$model_name'_'M_ETTh1_seqlen_48_labellen_48_predlen_24.log

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs/$model_name'_'M_ETTh1_seqlen_96_labellen_48_predlen_48.log

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs/$model_name'_'M_ETTh1_seqlen_168_labellen_168_predlen_168.log

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs/$model_name'_'M_ETTh1_seqlen_168_labellen_168_predlen_336.log

python -u main_informer.py --model $model_name --data ETTh1 --features M --seq_len 336 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs/$model_name'_'M_ETTh1_seqlen_336_labellen_336_predlen_720.log

### S

python -u main_informer.py --model $model_name --data ETTh1 --features S --seq_len 720 --label_len 168 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs/$model_name'_'S_ETTh1_seqlen_720_labellen_168_predlen_24.log

python -u main_informer.py --model $model_name --data ETTh1 --features S --seq_len 720 --label_len 168 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs/$model_name'_'S_ETTh1_seqlen_720_labellen_168_predlen_48.log

python -u main_informer.py --model $model_name --data ETTh1 --features S --seq_len 720 --label_len 336 --pred_len 168 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs/$model_name'_'S_ETTh1_seqlen_720_labellen_336_predlen_168.log

python -u main_informer.py --model $model_name --data ETTh1 --features S --seq_len 720 --label_len 336 --pred_len 336 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs/$model_name'_'S_ETTh1_seqlen_720_labellen_336_predlen_336.log

python -u main_informer.py --model $model_name --data ETTh1 --features S --seq_len 720 --label_len 336 --pred_len 720 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5 > logs/$model_name'_'S_ETTh1_seqlen_720_labellen_336_predlen_720.log