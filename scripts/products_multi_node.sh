for MODEL in graphsage gcn gat
do
  for DATATYPE in fp32 int1
  do
    if [ ${MODEL} == gat ]
    then
    EPOCH=100
    LAYER=3
    HIDDEN=128
    LINEAR=0
    else
    EPOCH=100
    LAYER=3
    HIDDEN=128
    LINEAR=0
    fi
  echo -e "\033[1mclean python processes\033[0m"
  sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
  echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}\033[0m"
  export GLOO_SOCKET_IFNAME=eno2
  python main.py \
    --dataset ogbn-products \
    --dropout 0.3 \
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
    --datatype ${DATATYPE} \
    --fix-seed \
    --use-pp \
    --skip-partition \
    --n-class 47 \
    --n-feat 100 \
    --n-train 196615 \
    --no-eval \
    --save_csv
    # --enable_pipeline 

  echo -e "\033[1mclean python processes\033[0m"
  sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
  echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}, pipeline\033[0m"
  export GLOO_SOCKET_IFNAME=eno2
  python main.py \
    --dataset ogbn-products \
    --dropout 0.3 \
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
    --datatype ${DATATYPE} \
    --fix-seed \
    --use-pp \
    --skip-partition \
    --n-class 47 \
    --n-feat 100 \
    --n-train 196615 \
    --no-eval \
    --save_csv \
    --enable_pipeline 
  done
done