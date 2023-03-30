export CUDA_VISIBLE_DEVICES=3

#cd ..
model=Transformer

for preLen in 48; do
  for seqlen in 48 96 144 192 240 288 336 384 432; do
    # ETT m1
    python -u run.py \
      --is_training 1 \
      --data_path ETTm1.csv \
      --task_id ETTm1 \
      --model $model \
      --data ETTm1 \
      --features M \
      --seq_len $seqlen \
      --label_len 48 \
      --pred_len $preLen \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 512

    # ETTh1
    python -u run.py \
      --is_training 1 \
      --data_path ETTh1.csv \
      --task_id ETTh1 \
      --model $model \
      --data ETTh1 \
      --features M \
      --seq_len $seqlen \
      --label_len 48 \
      --pred_len $preLen \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 512

    # ETTm2
    python -u run.py \
      --is_training 1 \
      --data_path ETTm2.csv \
      --task_id ETTm2 \
      --model $model \
      --data ETTm2 \
      --features M \
      --seq_len $seqlen \
      --label_len 48 \
      --pred_len $preLen \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 512

    # ETTh2
    python -u run.py \
      --is_training 1 \
      --data_path ETTh2.csv \
      --task_id ETTh2 \
      --model $model \
      --data ETTh2 \
      --features M \
      --seq_len $seqlen \
      --label_len 48 \
      --pred_len $preLen \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 512

    ## electricity
    python -u run.py \
      --is_training 1 \
      --data_path electricity.csv \
      --task_id ECL \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seqlen \
      --label_len 48 \
      --pred_len $preLen \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp'

    # exchange
    python -u run.py \
      --is_training 1 \
      --data_path exchange_rate.csv \
      --task_id Exchange \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seqlen \
      --label_len 48 \
      --pred_len $preLen \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp'

    # traffic
    python -u run.py \
      --is_training 1 \
      --data_path traffic.csv \
      --task_id traffic \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seqlen \
      --label_len 48 \
      --pred_len $preLen \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp'

    # weather
    python -u run.py \
      --is_training 1 \
      --data_path weather.csv \
      --task_id weather \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seqlen \
      --label_len 48 \
      --pred_len $preLen \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp'
  done
  for preLen in 24; do
    # illness
    python -u run.py \
      --is_training 1 \
      --data_path national_illness.csv \
      --task_id ili \
      --model $model \
      --data custom \
      --features M \
      --seq_len 36 \
      --label_len 18 \
      --pred_len $preLen \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp'
  done
done
