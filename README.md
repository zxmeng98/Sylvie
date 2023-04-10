# Sylvie: 3D-adaptive System for Large-scale GNN Training

## Directory Structure

```
|-- checkpoint   # model checkpoints
|-- dataset
|-- helper       # auxiliary codes
|-- module       # PyTorch modules
|-- partitions   # partitions of input graphs
|-- results      # experiment outputs
|-- scripts      # example scripts
```

Note that `./checkpoint/`, `./dataset/`, `./partitions/` and `./results/` are empty folders at the beginning and will be created when training is launched.

## Setup

### Environment

#### Hardware Dependencies

- A CPU machine with at least 120 GB host memory 
- At least five Nvidia GPUs (at least 24 GB each)

#### Software Dependencies

- Ubuntu 20.04
- Python 3.9
- CUDA 11.7
- [PyTorch 1.10](https://github.com/pytorch/pytorch)
- [customized DGL 0.9.0](https://github.com/dmlc/dgl/tree/0.9.x)
- [OGB 1.3.4](https://ogb.stanford.edu/docs/home/)

### Installation

#### Run with Docker

We have prepared a Docker image for Sylvie.

```bash
docker pull zxmeng98/sylvie
docker run --gpus all -it zxmeng98/sylvie
```
<!-- 
#### Option 2: Install with Conda

Running the following command will install prerequisites from conda.

```bash
bash setup.sh
``` -->

### Datasets

We use Reddit, ogbn-products, Yelp and Amazon for evaluations. All datasets are supposed to be stored in `./dataset/`. Reddit, ogbn-products and ogbn-papers100M will be downloaded by DGL or OGB automatically. Yelp is preloaded in the Docker environment, and is available [here](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz).


## Basic Usage

### Core Training Options

- `--dataset`: the dataset you want to use
- `--model`: the GCN model (only GraphSAGE and GAT are supported at this moment)
- `--n-hidden`: the number of hidden units
- `--n-layers`: the number of GCN layers
- `--n-partitions`: the number of partitions
- `--master-addr`: the address of master server
- `--port`: the network port for communication

<!-- ### Reproduce experiments

Run `scripts/reddit.sh`, `scripts/ogbn-products.sh` and `scripts/yelp.sh` can reproduce Sylvie under the default settings. 


For example, after running `bash scripts/reddit.sh`, you will get the output like this

```
...
Process 000 | Epoch 02999 | Time(s) 0.3578 | Comm(s) 0.2267 | Reduce(s) 0.0108 | Loss 0.0716
Process 001 | Epoch 02999 | Time(s) 0.3600 | Comm(s) 0.2314 | Reduce(s) 0.0136 | Loss 0.0867
(rank 1) memory stats: current 562.96MB, peak 1997.89MB, reserved 2320.00MB
(rank 0) memory stats: current 557.01MB, peak 2087.31MB, reserved 2296.00MB
Epoch 02999 | Accuracy 96.55%
model saved
Max Validation Accuracy 96.68%
Test Result | Accuracy 97.21%
``` -->

### Run Experiments

To reproduce experiments of our paper (e.g., throughput and accuracy in Table 4 and 5), please run `scripts/reddit.sh`,  `scripts/ogbn-products.sh` or  `scripts/yelp.sh`. Users can adjust the options to reproduce results of other settings. The outputs will be saved to `./results/` directory.

<!-- ### Run Customized Settings

You may adjust `--n-partitions` and `--sampling-rate` to reproduce the results of BNS-GCN under other settings. To verify the exact throughput or time breakdown of BNS-GCN, please add `--no-eval` argument to skip the evaluation step. You may also use the argument `--partition-method=random` to explore the performance of BNS-GCN with random partition.

### Run with Multiple Compute Nodes

Our code base also supports distributed GCN training with multiple compute nodes. To achieve this, you should specify `--master-addr`, `--node-rank` and `--parts-per-node` for each compute node. An example is provided in `scripts/reddit_multi_node.sh` where we train the Reddit graph over 4 compute nodes, each of which contains 10 GPUs, with 40 partitions in total. You should run the command on each node and specify the corresponding node rank. **Please turn on `--fix-seed` argument** so that all nodes initialize the same model weights.

If the compute nodes do not share storage, you should partition the graph in a single device first and manually distribute the partitions to other compute nodes. When run the training script, please enable `--skip-partition` argument. -->


