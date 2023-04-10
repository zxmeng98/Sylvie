# for FIX in 5 10 20 100
# do
# python main.py \
#   --dataset yelp \
#   --dropout 0.1 \
#   --lr 0.01 \
#   --n-partitions 8 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --sampling-rate 1 \
#   --n-layers 4 \
#   --n-linear 2 \
#   --n-hidden 512 \
#   --log-every 10 \
#   --inductive \
#   --datatype int1 \
#   --use-pp \
#   --enable_pipeline \
#   --fixed-synchro ${FIX} \
#   --save_csv \
#   --save_testacc
# done

python main.py \
  --dataset yelp \
  --dropout 0.1 \
  --lr 0.01 \
  --weight_decay 0 \
  --n-partitions 4 \
  --n-epochs 2000 \
  --model gcn \
  --n-layers 4 \
  --n-linear 2 \
  --n-hidden 512 \
  --log-every 10 \
  --inductive \
  --datatype fp32 \
  --use-pp \
  --k 16 
