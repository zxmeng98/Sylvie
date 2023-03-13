import os
import scipy
import dgl
from dgl.data import RedditDataset
from dgl.distributed import partition_graph
from helper.context import *
from ogb.nodeproppred import DglNodePropPredDataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from myconfig import *
import pandas as pd



class TransferTag:
    NODE = 0
    FEAT = 1


def load_ogb_dataset(name):
    dataset = DglNodePropPredDataset(name=name, root=data_dir)
    split_idx = dataset.get_idx_split()
    g, label = dataset[0]
    n_node = g.num_nodes()
    node_data = g.ndata
    node_data['label'] = label.view(-1).long()
    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][split_idx["train"]] = True
    node_data['val_mask'][split_idx["valid"]] = True
    node_data['test_mask'][split_idx["test"]] = True
    return g


def load_yelp():

    # prefix = './dataset/yelp/'
    prefix = data_dir + 'yelp/'

    with open(prefix + 'class_map.json') as f:
        class_map = json.load(f)
    with open(prefix + 'role.json') as f:
        role = json.load(f)

    adj_full = scipy.sparse.load_npz(prefix + 'adj_full.npz')
    feats = np.load(prefix + 'feats.npy')
    n_node = feats.shape[0]

    g = dgl.from_scipy(adj_full)
    node_data = g.ndata

    label = list(class_map.values())
    node_data['label'] = torch.tensor(label)

    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][role['tr']] = True
    node_data['val_mask'][role['va']] = True
    node_data['test_mask'][role['te']] = True

    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['val_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['test_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['val_mask'], node_data['test_mask'])))
    assert torch.all(torch.logical_or(torch.logical_or(node_data['train_mask'], node_data['val_mask']), node_data['test_mask']))

    train_feats = feats[node_data['train_mask']]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    node_data['feat'] = torch.tensor(feats, dtype=torch.float)

    return g    


def load_amazon():

    # prefix = './dataset/yelp/'
    prefix = data_dir + 'amazon/'

    with open(prefix + 'class_map.json') as f:
        class_map = json.load(f)
    class_map = {int(k): v for k, v in class_map.items()}
    
    with open(prefix + 'role.json') as f:
        role = json.load(f)

    adj_full = scipy.sparse.load_npz(prefix + 'adj_full.npz')
    feats = np.load(prefix + 'feats.npy')
    n_node = feats.shape[0]

    g = dgl.from_scipy(adj_full)
    node_data = g.ndata

    num_classes = len(list(class_map.values())[0])
    class_arr = np.zeros((g.num_nodes(), num_classes))
    for k, v in class_map.items():
        class_arr[k] = v

    node_data['label'] = torch.tensor(class_arr, dtype=torch.float)

    node_data['train_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['val_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['test_mask'] = torch.zeros(n_node, dtype=torch.bool)
    node_data['train_mask'][role['tr']] = True
    node_data['val_mask'][role['va']] = True
    node_data['test_mask'][role['te']] = True

    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['val_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['train_mask'], node_data['test_mask'])))
    assert torch.all(torch.logical_not(torch.logical_and(node_data['val_mask'], node_data['test_mask'])))
    assert torch.all(torch.logical_or(torch.logical_or(node_data['train_mask'], node_data['val_mask']), node_data['test_mask']))

    train_feats = feats[node_data['train_mask']]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    node_data['feat'] = torch.tensor(feats, dtype=torch.float)

    return g

def load_data(dataset):
    if dataset == 'reddit':
        data = RedditDataset(raw_dir=data_dir)
        g = data[0]
    elif dataset == 'ogbn-products':
        g = load_ogb_dataset('ogbn-products')
    elif dataset == 'ogbn-papers100m':
        g = load_ogb_dataset('ogbn-papers100M')
    elif dataset == 'ogbn-arxiv':
        g = load_ogb_dataset('ogbn-arxiv')
    elif dataset == 'yelp':
        g = load_yelp()
    elif dataset == 'amazon':
        g = load_amazon()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

    n_feat = g.ndata['feat'].shape[1]
    if g.ndata['label'].dim() == 1:
        n_class = g.ndata['label'].max().item() + 1
    else:
        n_class = g.ndata['label'].shape[1]

    g.edata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, n_feat, n_class


def graph_partition(g, args):

    graph_dir = partition_dir + 'partitions/' + args.graph_name + '/'
    part_config = graph_dir + args.graph_name + '.json'

    # TODO: after being saved, a bool tensor becomes a uint8 tensor (including 'inner_node')
    if not os.path.exists(part_config):
        with g.local_scope():
            if args.inductive:
                g.ndata.pop('val_mask')
                g.ndata.pop('test_mask')
            g.ndata['in_degree'] = g.in_degrees()
            partition_graph(g, args.graph_name, args.n_partitions, graph_dir,  part_method=args.partition_method,
                            reshuffle=True, balance_edges=False, objtype=args.partition_obj)


def load_partition(args, rank):

    graph_dir = partition_dir + 'partitions/' + args.graph_name + '/'
    part_config = graph_dir + args.graph_name + '.json'

    print('loading partitions')

    subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, rank)
    node_type = node_type[0]
    node_feat[dgl.NID] = subg.ndata[dgl.NID]
    if 'part_id' in subg.ndata:
        node_feat['part_id'] = subg.ndata['part_id']
    node_feat['inner_node'] = subg.ndata['inner_node'].bool()
    node_feat['label'] = node_feat[node_type + '/label']
    node_feat['feat'] = node_feat[node_type + '/feat']
    node_feat['in_degree'] = node_feat[node_type + '/in_degree']
    node_feat['train_mask'] = node_feat[node_type + '/train_mask'].bool()
    node_feat.pop(node_type + '/label')
    node_feat.pop(node_type + '/feat')
    node_feat.pop(node_type + '/in_degree')
    node_feat.pop(node_type + '/train_mask')
    if not args.inductive:
        node_feat['val_mask'] = node_feat[node_type + '/val_mask'].bool()
        node_feat['test_mask'] = node_feat[node_type + '/test_mask'].bool()
        node_feat.pop(node_type + '/val_mask')
        node_feat.pop(node_type + '/test_mask')
    if args.dataset == 'ogbn-papers100m':
        node_feat.pop(node_type + '/year')
    subg.ndata.clear()
    subg.edata.clear()

    return subg, node_feat, gpb


def get_layer_size(n_feat, n_hidden, n_class, n_layers):
    layer_size = [n_feat]
    layer_size.extend([n_hidden] * (n_layers - 1))
    layer_size.append(n_class)
    return layer_size


def get_boundary(node_dict, gpb):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right) # node_dict['part_id']一个tesor，size=这个partition所有node数量，标记每个node属于哪个partition
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start # (在子图中通过dgl.NID或者dgl.EID来获取父图中节点和边的映射), - start = 获取相对于子图的local node ID
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long)
        else:
            u = torch.zeros(num_left, dtype=torch.long, device=device)
        req.wait()
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.cuda()
        else:
            boundary[left] = u
        req.wait()

    return boundary


def data_transfer(data, recv_shape, backend, dtype=torch.float, tag=0):
    rank, size = dist.get_rank(), dist.get_world_size()
    res = [None] * size

    for i in range(1, size):
        left = (rank - i + size) % size
        if backend == 'gloo':
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype)
        else:
            res[left] = torch.zeros(torch.Size([recv_shape[left], data[left].shape[1]]), dtype=dtype, device='cuda')

    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        if backend == 'gloo':
            req = dist.isend(data[right].cpu(), dst=right, tag=tag)
        else:
            req = dist.isend(data[right], dst=right, tag=tag)
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda()
        req.wait()

    return res


def merge_feature(feat, recv):
    size = len(recv)
    for i in range(size - 1, 0, -1):
        if recv[i] is None:
            recv[i] = recv[i - 1]
            recv[i - 1] = None
    recv[0] = feat
    return torch.cat(recv)


def inductive_split(g):
    g_train = g.subgraph(g.ndata['train_mask'])
    g_val = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    g_test = g
    return g_train, g_val, g_test


def minus_one_tensor(size, device=None):
    if device is not None:
        return torch.zeros(size, dtype=torch.long, device=device) - 1
    else:
        return torch.zeros(size, dtype=torch.long) - 1


def nonzero_idx(x):
    return torch.nonzero(x, as_tuple=True)[0]


def print_memory(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.cuda.synchronize()
    print('(rank %d) ' % rank + s +
          ': current {:.2f}MB, peak {:.2f}MB, reserved {:.2f}MB'.format(torch.cuda.memory_allocated() / 1024 / 1024,
                                                                   torch.cuda.max_memory_allocated() / 1024 / 1024,
                                                                   torch.cuda.memory_reserved() / 1024 / 1024))

@contextmanager
def timer(s):
    rank, size = dist.get_rank(), dist.get_world_size()
    t = time.time()
    yield
    print('(rank %d) running time of %s: %.3f seconds' % (rank, s, time.time() - t))


def boundary_deg_group(boundary, node_dict):
    # 按照degree分组boundary nodes ID
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary_devide = [None] * size
    for i in range(size):
        if i == rank:
            continue
        else:
            degree_bdry = node_dict['in_degree'][boundary[i]]
            df_degree_bdry = pd.DataFrame({'idx': boundary[i].cpu(), 'deg':degree_bdry.cpu()})
            df_degree_bdry.sort_values(by=['deg'], inplace=True)
            
            # TODO: Choose nodes (degree = 3) to be quantized to 1 bit
            bdry1 = torch.tensor(np.array(df_degree_bdry.loc[df_degree_bdry['deg'] == 3]['idx']), device=device)
            bdry2 = torch.tensor(np.array(df_degree_bdry.loc[df_degree_bdry['deg'] != 3]['idx']), device=device)
            
            boundary_devide[i] = [bdry1, bdry2]
            
    return boundary_devide


def get_recv_info_ada(boundary_group, hidden_size, recv_shape, assign_bits):
    # According to the assigned node bits, prepare the quant group buffer size  
    rank, size = dist.get_rank(), dist.get_world_size()
    
    qgroup_size = [] # [None, [], [], []]
    for j in range(size):
        tmp = []
        if j == rank:
            qgroup_size.append(None)
            continue
        for k in range(len(assign_bits)):
            fake_send = torch.randn(len(boundary_group[j][k]), hidden_size).cuda()
            quant_feat_send, _, _ = quantize_and_pack(fake_send, assign_bits[k])
            s = quant_feat_send.shape[0]
            tmp.append(s)
        
        tmp = torch.tensor(tmp) 
        qgroup_size.append(tmp)
    
    qgroup_recv = [None] * size
    group_recv_id = [None] * size # nodes get from other partitions
    group_recv_size = [None] * size 
    # Send after quant group buffer size, grouped node IDs, grouped node size
    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        
        if dist.get_backend() == 'gloo':
            qgroup_recv_left = torch.zeros(len(assign_bits), dtype=torch.long)
        req = dist.isend(qgroup_size[right], dst=right) # ! send tensor must be on cpu
        dist.recv(qgroup_recv_left, src=left)
        qgroup_recv[left] = qgroup_recv_left
        if dist.get_backend() == 'gloo':
            shuffled_ids = torch.zeros(recv_shape[left], dtype=torch.long)
            tmp = torch.cat(boundary_group[right]).cpu()
        req.wait()
        req = dist.isend(tmp, dst=right)
        dist.recv(shuffled_ids, src=left)
        group_recv_id[left] = shuffled_ids 
        if dist.get_backend() == 'gloo':
            group_size = torch.zeros(len(assign_bits), dtype=torch.long)
        req.wait()
        req = dist.isend(torch.tensor([boundary_group[right][k].shape[0] for k in range(len(assign_bits))]), dst=right)
        dist.recv(group_size, src=left)
        group_recv_size[left] = group_size
        req.wait()

    return qgroup_size, qgroup_recv, group_recv_id, group_recv_size

    
        
        
        
            
        
        

    
                
                
