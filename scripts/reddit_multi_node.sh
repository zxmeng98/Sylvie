for MODEL in jknet
do
  for DATATYPE in adap
  do
  if [ ${MODEL} == gat ]
  then
    EPOCH=100
    LAYER=2
  else
    EPOCH=800
    LAYER=8
  fi
  echo -e "\033[1mclean python processes\033[0m"
  sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
  echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}\033[0m"
  export GLOO_SOCKET_IFNAME=eno2
  python main.py \
    --dataset reddit \
    --dropout 0.5 \
    --lr 0.01 \
    --weight_decay 0.0005 \
    --n-partitions 8 \
    --n-epochs ${EPOCH} \
    --model ${MODEL} \
    --n-layers ${LAYER} \
    --n-hidden 128 \
    --master-addr 192.168.1.40 \
    --node-rank 0 \
    --parts-per-node 4 \
    --log-every 10 \
    --inductive \
    --datatype ${DATATYPE} \
    --fix-seed \
    --use-pp \
    --skip-partition \
    --n-class 41 \
    --n-feat 602 \
    --n-train 153431 
    # --enable_pipeline 

  # echo -e "\033[1mclean python processes\033[0m"
  # sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
  # echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}\033[0m"
  # export GLOO_SOCKET_IFNAME=eno2
  # python main.py \
  #   --dataset reddit \
  #   --dropout 0.5 \
  #   --lr 0.01 \
  #   --n-partitions 8 \
  #   --n-epochs ${EPOCH} \
  #   --model ${MODEL} \
  #   --n-layers ${LAYER} \
  #   --n-hidden 256 \
  #   --master-addr 192.168.1.40 \
  #   --node-rank 0 \
  #   --parts-per-node 4 \
  #   --log-every 10 \
  #   --inductive \
  #   --datatype ${DATATYPE} \
  #   --fix-seed \
  #   --use-pp \
  #   --skip-partition \
  #   --n-class 41 \
  #   --n-feat 602 \
  #   --n-train 153431 \
  #   --no-eval \
  #   --enable_pipeline \
  #   --save_csv
  done
done
