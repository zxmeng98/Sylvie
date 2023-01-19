for MODEL in graphsage gcn gat
do
  for DATATYPE in fp32 int1
  do
    if [ ${MODEL} == gat ]
    then
    EPOCH=100
    LAYER=2
    HIDDEN=256
    LINEAR=0
    else
    EPOCH=100
    LAYER=4
    HIDDEN=512
    LINEAR=2
    fi
  echo -e "\033[1mclean python processes\033[0m"
  sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
  echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}\033[0m"
  export GLOO_SOCKET_IFNAME=eno2
  python main.py \
    --dataset yelp \
    --dropout 0.1 \
    --lr 0.01 \
    --n-partitions 8 \
    --n-epochs ${EPOCH} \
    --model ${MODEL} \
    --n-layers ${LAYER} \
    --n-linear ${LINEAR} \
    --n-hidden ${HIDDEN} \
    --master-addr 192.168.1.40 \
    --node-rank 0 \
    --parts-per-node 4 \
    --log-every 10 \
    --inductive \
    --datatype ${DATATYPE} \
    --fix-seed \
    --use-pp \
    --skip-partition \
    --n-class 100 \
    --n-feat 300 \
    --n-train 537635 \
    --no-eval \
    --save_csv
    # --enable_pipeline 

  echo -e "\033[1mclean python processes\033[0m"
  sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
  echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}\033[0m"
  export GLOO_SOCKET_IFNAME=eno2
  python main.py \
    --dataset yelp \
    --dropout 0.1 \
    --lr 0.01 \
    --n-partitions 8 \
    --n-epochs ${EPOCH} \
    --model ${MODEL} \
    --n-layers ${LAYER} \
    --n-linear ${LINEAR} \
    --n-hidden ${HIDDEN} \
    --master-addr 192.168.1.40 \
    --node-rank 0 \
    --parts-per-node 4 \
    --log-every 10 \
    --inductive \
    --datatype ${DATATYPE} \
    --fix-seed \
    --use-pp \
    --skip-partition \
    --n-class 100 \
    --n-feat 300 \
    --n-train 537635 \
    --no-eval \
    --save_csv \
    --enable_pipeline 
  done
done