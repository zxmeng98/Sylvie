python main.py \
  --dataset ogbn-arxiv \
  --dropout 0.5 \
  --lr 0.01 \
  --n-partitions 4 \
  --n-epochs 1000 \
  --model dagnn \
  --sampling-rate 1 \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 10 \
  --use-pp \
  --fix-seed \
  --datatype fp32
