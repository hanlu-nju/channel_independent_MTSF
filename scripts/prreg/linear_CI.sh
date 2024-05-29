export CUDA_VISIBLE_DEVICES=3
model_name=Linear
for wd in 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1.0; do
  for pred_len in 48 96; do
    seq_len=96
    python -u run.py \
      --is_training 1 \
      --data_path ETTh1.csv \
      --task_id ETTh1_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'low rank experiment' \
      --tags weight_decay \
      --flat_input \
      --wd $wd \
      --pred_residual \
      --batch_size 32 --learning_rate 0.005

    python -u run.py \
      --is_training 1 \
      --data_path ETTh2.csv \
      --task_id ETTh2_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'low rank experiment' \
      --tags weight_decay \
      --flat_input \
      --wd $wd \
      --pred_residual \
      --batch_size 32 --learning_rate 0.05

    python -u run.py \
      --is_training 1 \
      --data_path ETTm1.csv \
      --task_id ETTm1_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'low rank experiment' \
      --tags weight_decay \
      --flat_input \
      --wd $wd \
      --pred_residual \
      --batch_size 8 --learning_rate 0.0001

    python -u run.py \
      --is_training 1 \
      --data_path ETTm2.csv \
      --task_id ETTm2_$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'low rank experiment' \
      --tags weight_decay \
      --flat_input \
      --wd $wd \
      --pred_residual \
      --batch_size 32 --learning_rate 0.001

    python -u run.py \
      --is_training 1 \
      --data_path exchange_rate.csv \
      --task_id Exchange_$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --des 'low rank experiment' \
      --tags weight_decay \
      --flat_input \
      --wd $wd \
      --pred_residual \
      --batch_size 32 --learning_rate 0.0005

    python -u run.py \
      --is_training 1 \
      --data_path weather.csv \
      --task_id weather_$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --des 'low rank experiment' \
      --tags weight_decay \
      --flat_input \
      --wd $wd \
      --pred_residual \
      --batch_size 32

  done

  for pred_len in 24 36; do
    python -u run.py \
      --is_training 1 \
      --data_path national_illness.csv \
      --task_id national_illness_$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 18 \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'low rank experiment' \
      --tags weight_decay \
      --flat_input \
      --wd $wd \
      --pred_residual \
      --batch_size 32 --learning_rate 0.01
  done
done
