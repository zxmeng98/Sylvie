python main.py \
  --dataset amazon \
  --dropout 0.1 \
  --lr 0.01 \
  --n-partitions 4 \
  --n-epochs 150 \
  --model dagnn \
  --sampling-rate 1 \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 10 \
  --datatype fp32 \
  --inductive \
  --use-pp \
  --fix-seed
  # --enable_pipeline 

