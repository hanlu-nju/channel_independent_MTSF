export CUDA_VISIBLE_DEVICES=2

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
      --des 'global_dataset' \
      --channel_independent \
      --batch_size 256 \
      --d_model 64

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
      --des 'global_dataset' \
      --channel_independent \
      --batch_size 256 \
      --d_model 64

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
      --des 'global_dataset' \
      --channel_independent \
      --batch_size 256 \
      --d_model 64

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
      --des 'global_dataset' \
      --channel_independent \
      --batch_size 256 \
      --d_model 64

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
      --des 'global_dataset' \
      --channel_independent \
      --batch_size 256 \
      --d_model 64

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
      --des 'global_dataset' \
      --channel_independent \
      --batch_size 256 \
      --d_model 64

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
      --des 'global_dataset' \
      --channel_independent \
      --batch_size 256 \
      --d_model 64

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
      --des 'global_dataset' \
      --channel_independent \
      --batch_size 256 \
      --d_model 64
  done
done

#for preLen in 24 36 48 60; do
#  # illness
#  python -u run.py \
#    --is_training 1 \
#    --data_path national_illness.csv \
#    --task_id ili \
#    --model $model \
#    --data custom \
#    --features M \
#    --seq_len 36 \
#    --label_len 18 \
#    --pred_len $preLen \
#    --e_layers 2 \
#    --d_layers 1 \
#    --factor 3 \
#    --enc_in 7 \
#    --dec_in 7 \
#    --c_out 7 \
#    --des 'global_dataset' \
#    --channel_independent \
#    --batch_size 64 \
#    --d_model 512
#done
