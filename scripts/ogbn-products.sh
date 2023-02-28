python main.py \
  --dataset ogbn-products \
  --dropout 0.5 \
  --lr 0.01 \
  --weight_decay 0.05 \
  --n-partitions 4 \
  --n-epochs 500 \
  --model dagnn \
  --sampling-rate 1 \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 10 \
  --datatype fp32 \
  --use-pp \
  --k 10 \
  --fix-seed \
  --save_testacc

