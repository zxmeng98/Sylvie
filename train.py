import torch.nn.functional as F
from module.model import *
from helper.utils import *
import torch.distributed as dist
import time
import copy
from multiprocessing.pool import ThreadPool
from sklearn.metrics import f1_score
import pandas as pd
from module.gin import *


def calc_acc(logits, labels):
    if labels.dim() == 1:
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]
    else:
        return f1_score(labels, logits > 0, average='micro')


@torch.no_grad()
def evaluate_induc(name, model, g, mode, result_file_name=None):
    """
    mode: 'val' or 'test'
    """
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    mask = g.ndata[mode + '_mask']
    logits = model(g, feat)
    logits = logits[mask]
    labels = labels[mask]
    acc = calc_acc(logits, labels)
    buf = "{:s} | Accuracy {:.2%}".format(name, acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, acc


@torch.no_grad()
def evaluate_trans(name, model, g, result_file_name=None):
    model.eval()
    model.cpu()
    feat, labels = g.ndata['feat'], g.ndata['label']
    val_mask, test_mask = g.ndata['val_mask'], g.ndata['test_mask']
    logits = model(g, feat)
    val_logits, test_logits = logits[val_mask], logits[test_mask]
    val_labels, test_labels = labels[val_mask], labels[test_mask]
    val_acc = calc_acc(val_logits, val_labels)
    test_acc = calc_acc(test_logits, test_labels)
    buf = "{:s} | Validation Accuracy {:.2%} | Test Accuracy {:.2%}".format(name, val_acc, test_acc)
    if result_file_name is not None:
        with open(result_file_name, 'a+') as f:
            f.write(buf + '\n')
            print(buf)
    else:
        print(buf)
    return model, val_acc, test_acc


def average_gradients(model, n_train):
    reduce_time = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        t0 = time.time()
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= n_train
        reduce_time += time.time() - t0
    return reduce_time


def move_to_cuda(graph, part, node_dict):

    for key in node_dict.keys():
        node_dict[key] = node_dict[key].cuda()
    graph = graph.int().to(torch.device('cuda'))
    part = part.int().to(torch.device('cuda'))

    return graph, part, node_dict


def get_pos(node_dict, gpb):
    pos = []
    rank, size = dist.get_rank(), dist.get_world_size()
    for i in range(size):
        if i == rank:
            pos.append(None)
        else:
            part_size = gpb.partid2nids(i).shape[0]
            start = gpb.partid2nids(i)[0].item()
            p = minus_one_tensor(part_size, 'cuda')
            in_idx = nonzero_idx(node_dict['part_id'] == i)
            out_idx = node_dict[dgl.NID][in_idx] - start
            p[out_idx] = in_idx
            pos.append(p)
    return pos


def get_recv_shape(node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
        else:
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape


def create_inner_graph(graph, node_dict):
    u, v = graph.edges()
    sel = torch.logical_and(node_dict['inner_node'].bool()[u], node_dict['inner_node'].bool()[v])
    u, v = u[sel], v[sel]
    return dgl.graph((u, v))


def order_graph(part, graph, gpb, node_dict, pos):
    rank, size = dist.get_rank(), dist.get_world_size()
    one_hops = []
    for i in range(size):
        if i == rank:
            one_hops.append(None)
            continue
        start = gpb.partid2nids(i)[0].item()
        nodes = node_dict[dgl.NID][node_dict['part_id'] == i] - start
        nodes, _ = torch.sort(nodes)
        one_hops.append(nodes)
    return construct(part, graph, pos, one_hops), one_hops


def move_train_first(graph, node_dict, boundary):
    train_mask = node_dict['train_mask']
    num_train = torch.count_nonzero(train_mask).item()
    num_tot = graph.num_nodes('_V')

    new_id = torch.zeros(num_tot, dtype=torch.int, device='cuda')
    new_id[train_mask] = torch.arange(num_train, dtype=torch.int, device='cuda')
    new_id[torch.logical_not(train_mask)] = torch.arange(num_train, num_tot, dtype=torch.int, device='cuda')

    u, v = graph.edges()
    u[u < num_tot] = new_id[u[u < num_tot].long()]
    v = new_id[v.long()]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})

    for key in node_dict:
        node_dict[key][new_id.long()] = node_dict[key][0:num_tot].clone()

    for i in range(len(boundary)):
        if boundary[i] is not None:
            boundary[i] = new_id[boundary[i]].long()

    return graph, node_dict, boundary


def create_graph_train(graph, node_dict):
    u, v = graph.edges()
    num_u = graph.num_nodes('_U')
    sel = nonzero_idx(node_dict['train_mask'][v.long()])
    u, v = u[sel], v[sel]
    graph = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if graph.num_nodes('_U') < num_u:
        graph.add_nodes(num_u - graph.num_nodes('_U'), ntype='_U')
    return graph, node_dict['in_degree'][node_dict['train_mask']]


def precompute(graph, node_dict, boundary, recv_shape, args):
    rank, size = dist.get_rank(), dist.get_world_size()
    in_size = node_dict['inner_node'].bool().sum()
    feat = node_dict['feat']
    send_info = []
    for i, b in enumerate(boundary):
        if i == rank:
            send_info.append(None)
        else:
            send_info.append(feat[b])
    recv_feat = data_transfer(send_info, recv_shape, args.backend, dtype=torch.float)
    if args.model == 'graphsage':
        with graph.local_scope():
            graph.nodes['_U'].data['h'] = merge_feature(feat, recv_feat)
            graph['_E'].update_all(fn.copy_src(src='h', out='m'),
                                   fn.sum(msg='m', out='h'),
                                   etype='_E')
            mean_feat = graph.nodes['_V'].data['h'] / node_dict['in_degree'][0:in_size].unsqueeze(1)
        return torch.cat([feat, mean_feat[0:in_size]], dim=1)
    elif args.model == 'gcn' or args.model == 'gin' or args.model == 'gat':
        # raise NotImplementedError
        return merge_feature(feat, recv_feat)
    elif args.model == 'appnp':
        return feat
    elif args.model == 'dagnn':
        return feat
    else:
        raise Exception


def create_model(layer_size, args):
    if args.model == 'graphsage':
        return GraphSAGE(layer_size, F.relu, norm=args.norm, use_pp=args.use_pp, dropout=args.dropout,
                         train_size=args.n_train, n_linear=args.n_linear)
    elif args.model == 'gat':
        heads = [args.heads] * (args.n_layers-1) + [1]
        return GAT(layer_size, F.relu, use_pp=True, heads=heads, norm=args.norm, dropout=args.dropout, n_linear=args.n_linear)
    elif args.model == 'gcn':
        return GCN(layer_size, F.relu, norm=args.norm, use_pp=args.use_pp, dropout=args.dropout,
                         train_size=args.n_train, n_linear=args.n_linear)
    elif args.model == 'gin':
        return GIN(args.n_layers, 2, layer_size[0], layer_size[1], layer_size[-1], args.dropout, False, 'sum', 'sum')
    elif args.model == 'appnp':
        return APPNP(
        layer_size[0],
        [args.n_hidden],
        args.n_class,
        F.relu,
        0,
        args.dropout,
        0.1,
        args.k,
    )
    elif args.model == 'dagnn':
        return DAGNN(
        k=args.k,
        in_dim=layer_size[0],
        hid_dim=args.n_hidden,
        out_dim=args.n_class,
        dropout=args.dropout, # 0.8 / 0.5
    )


def reduce_hook(param, name, n_train):
    def fn(grad):
        ctx.reducer.reduce(param, name, grad, n_train)
    return fn


def construct(part, graph, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    tot = part.num_nodes()
    u, v = part.edges()
    u_list, v_list = [u], [v]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            u_ = torch.repeat_interleave(graph.out_degrees(u.int()).long()) + tot
            tot += u.shape[0]
            _, v = graph.out_edges(u.int())
            u_list.append(u_.int())
            v_list.append(v)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
    if g.num_nodes('_U') < tot:
        g.add_nodes(tot - g.num_nodes('_U'), ntype='_U')
    return g


def extract(graph, node_dict):
    rank, size = dist.get_rank(), dist.get_world_size()
    sel = (node_dict['part_id'] < size)
    for key in node_dict.keys():
        if node_dict[key].shape[0] == sel.shape[0]:
            node_dict[key] = node_dict[key][sel]
    graph = dgl.node_subgraph(graph, sel, store_ids=False)
    return graph, node_dict


def get_send_size(boundary, prob):
    rank, size = dist.get_rank(), dist.get_world_size()
    res, ratio = [], []
    for i, b in enumerate(boundary):
        if i == rank:
            res.append(0)
            ratio.append(0)
            continue
        s = int(prob * b.shape[0])
        res.append(s)
        # TODO: ratio.append(1 if args.model == 'gat' else s / b.shape[0])
        ratio.append(s / b.shape[0])
    return res, ratio


def construct_feat(num, feat, pos, one_hops):
    rank, size = dist.get_rank(), dist.get_world_size()
    res = [feat[0:num]]
    for i in range(size):
        if i == rank:
            continue
        else:
            u = one_hops[i]
            if u.shape[0] == 0:
                continue
            u = pos[i][u]
            res.append(feat[u])

    return torch.cat(res)


def run(graph, node_dict, gpb, args):
    
    rank, size = dist.get_rank(), dist.get_world_size()

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    if rank == 0 and args.eval:
        full_g, n_feat, n_class = load_data(args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            val_g, test_g = full_g.clone(), full_g.clone()
        del full_g

    if rank == 0:
        os.makedirs('checkpoint/', exist_ok=True)
        os.makedirs('results/speed_pcie4', exist_ok=True)
        os.makedirs('results/speed_pcie4_oregon2', exist_ok=True)
        os.makedirs('results/testacc_curve_pcie3', exist_ok=True)
        os.makedirs('results/testacc_curve_pcie4', exist_ok=True)
        

    part = create_inner_graph(graph.clone(), node_dict)
    num_in = node_dict['inner_node'].bool().sum().item()
    part.ndata.clear()
    part.edata.clear()

    print(f'Process {rank} has {graph.num_nodes()} nodes, {graph.num_edges()} edges '
          f'{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')

    graph, part, node_dict = move_to_cuda(graph, part, node_dict)
    boundary = get_boundary(node_dict, gpb)

    layer_size = get_layer_size(args.n_feat, args.n_hidden, args.n_class, args.n_layers)
    # print(layer_size, layer_size[:args.n_layers - args.n_linear])

    pos = get_pos(node_dict, gpb)
    graph, one_hops = order_graph(part, graph, gpb, node_dict, pos)
    in_deg = node_dict['in_degree']

    graph, node_dict, boundary = move_train_first(graph, node_dict, boundary)

    recv_shape = get_recv_shape(node_dict)
    send_size, ratio = get_send_size(boundary, 1)
    
    # '_U'是包含 boundary nodes
    # [args.n_hidden]*(args.k+1)
    ctx.buffer.init_buffer(num_in, graph.num_nodes('_U'), send_size, recv_shape, [args.n_class]*(args.k+1),
                           use_pp=args.use_pp, backend=args.backend, dtype=args.datatype, pipeline=args.enable_pipeline, corr_feat=args.feat_corr, corr_grad=args.grad_corr, corr_momentum=args.corr_momentum, fixed_synchro=args.fixed_synchro)
    # ctx.buffer2.init_buffer(num_in, graph.num_nodes('_U'), send_size, recv_shape, layer_size[:args.n_layers - args.n_linear],
    #                        use_pp=args.use_pp, backend=args.backend, dtype='int8', pipeline=args.enable_pipeline, corr_feat=args.feat_corr, corr_grad=args.grad_corr, corr_momentum=args.corr_momentum, fixed_synchro=args.fixed_synchro)
    ctx.buffer.set_selected(boundary)
    # ctx.buffer2.set_selected(boundary)

    if args.use_pp:
        node_dict['feat'] = precompute(graph, node_dict, boundary, recv_shape, args)

    labels = node_dict['label'][node_dict['train_mask']]
    train_mask = node_dict['train_mask']
    part_train = train_mask.int().sum().item()

    del boundary
    del part
    del pos

    torch.manual_seed(args.seed)
    model = create_model(layer_size, args)
    model.cuda()

    ctx.reducer.init(model)

    for i, (name, param) in enumerate(model.named_parameters()):
        param.register_hook(reduce_hook(param, name, args.n_train))

    best_model, best_acc = None, 0

    if args.dataset == 'yelp' or args.dataset == 'amazon':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    train_dur, comm_dur, reduce_dur, extra_dur = [], [], [], []
    quant_dur, dequant_dur = [], []

    torch.cuda.reset_peak_memory_stats()
    thread = None
    pool = ThreadPool(processes=1)

    feat = node_dict['feat']

    node_dict.pop('train_mask')
    node_dict.pop('inner_node')
    node_dict.pop('part_id')
    node_dict.pop(dgl.NID)

    # if not args.eval:
    #     node_dict.pop('val_mask')
    #     node_dict.pop('test_mask')

    print(f'Process {rank} start training')
    f_abs_err = []
    f_relative_err = []
    grad_abs_err = []
    grad_relative_err = []
    for epoch in range(args.n_epochs):
        # if epoch == 100:
        #     ctx.buffer.change_epoch_bit()
            
        ctx.buffer.set_pipeline()

        t0 = time.time()
        model.train()
        
        if args.model == 'graphsage' or args.model == 'gcn' or args.model == 'gin':
            logits = model(graph, feat, in_deg)
            
            # if rank == 0:
            #     print(epoch, abs_err)
            # f_abs_err.append(abs_err.item()/3)
            # f_relative_err.append(rel_err.item()/3)
        elif args.model == 'gat':
            logits = model(graph, feat)
        elif args.model == 'appnp':
            logits = model(graph, feat)
        elif args.model == 'dagnn':
            logits = model(graph, feat, in_deg)
        else:
            raise NotImplementedError

        if args.inductive:
            loss = loss_fcn(logits, labels)
        else:
            loss = loss_fcn(logits[train_mask], labels)
        del logits
        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        # print(f'rank: {rank}, total: {ctx.pipe_buffer.commu_volume} MB')
        # grad_abs_err.append(ctx.buffer2.grad_abs_err.item()/3)
        # if rank == 0:
        #     dict = {'grad abs err': ctx.buffer2.grad_abs_err.item()/3}
        #     df = pd.DataFrame([dict])
        #     err_file_csv = 'results/g_abs_err.csv'
        #     if os.path.exists(err_file_csv):
        #         df.to_csv(err_file_csv, mode='a', header=False, index=False)
        #     else:
        #         df.to_csv(err_file_csv, mode='a', index=False)
        # grad_relative_err.append(ctx.buffer2.grad_rel_err.item()/3)
        # ctx.buffer2.grad_abs_err = 0
        ctx.buffer.next_epoch()
        # ctx.buffer2.next_epoch()

        pre_reduce = time.time()
        ctx.reducer.synchronize()
        reduce_time = time.time() - pre_reduce
        optimizer.step()

        if epoch >= 5 and epoch % args.log_every != 0:
            train_dur.append(time.time() - t0)
            comm_dur.append(ctx.comm_timer.tot_time())
            reduce_dur.append(reduce_time)
            extra_dur.append(ctx.quant_timer.tot_time())

            # Process the quant & dequant time 
            quant_t, dequant_t = 0, 0
            for (k, (t0, t1)) in quant_timer._time.items():
                str_list = k.split('_')
                if str_list[0] == 'fquant' or str_list[0] == 'bquant':
                    quant_t += t1 - t0
                else:
                    dequant_t += t1 - t0
            quant_dur.append(quant_t)
            dequant_dur.append(dequant_t)

        if (epoch + 1) % 10 == 0:
            # compute_t = np.mean(train_dur) - np.mean(comm_dur) - np.mean(reduce_dur)
            compute_t = np.mean(train_dur) - np.mean(comm_dur) - np.mean(reduce_dur) - np.mean(dequant_dur)
            # print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Compute(s) {:.4f} | Loss {:.4f}".format(
            #       rank, epoch, np.mean(train_dur), np.mean(comm_dur), np.mean(reduce_dur), compute_t, loss.item() / part_train))
            print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Compute(s) {:.4f} | Extra(s) {:.4f}, quant {:.4f}, dequant {:.4f} | Loss {:.4f}".format(
                  rank, epoch, np.mean(train_dur), np.mean(comm_dur)-np.mean(quant_dur), np.mean(reduce_dur), compute_t, np.mean(extra_dur), np.mean(quant_dur), np.mean(dequant_dur), loss.item() / part_train))

        ctx.comm_timer.clear()
        ctx.quant_timer.clear()
        
        # del loss
        if rank == 0 and args.eval and (epoch + 1) % args.log_every == 0:
            if thread is not None:
                if args.inductive:
                    model_copy, val_acc = thread.get()
                else:
                    model_copy, val_acc, _ = thread.get()
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = model_copy
            model_copy = copy.deepcopy(model)
            if not args.inductive:
                thread = pool.apply_async(evaluate_trans, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, None))
            else:
                thread = pool.apply_async(evaluate_induc, args=('Epoch %05d' % epoch, model_copy,
                                                                val_g, 'val', None))
            if args.save_testacc:
                if args.inductive:
                    _, acc = thread.get()
                else:
                    _, _, acc = thread.get()
                if args.fixed_synchro is not None:
                    acc_file_csv = 'results/testacc_curve_products/%s_n%d_%s_%s_%d.csv' % (args.dataset, args.n_partitions, args.model, args.datatype, args.fixed_synchro)
                else:
                    # acc_file_csv = 'results/%s_n%d_%s_%s_%s_test.csv' % (args.dataset, args.n_partitions, args.model, args.datatype, args.enable_pipeline)
                    acc_file_csv = 'results/%s_%s_100e1_300e32.csv' % (args.dataset, args.model)
                dict = {'epoch': epoch, 'acc': acc, 'loss': loss.item()}
                df = pd.DataFrame([dict])
                if os.path.exists(acc_file_csv):
                    df.to_csv(acc_file_csv, mode='a', header=False, index=False)
                else: 
                    df.to_csv(acc_file_csv, mode='a', index=False)
       

    # print(f'rank {rank}, f abs: {np.mean(f_abs_err)}, f rel: {np.mean(f_relative_err)}, grad abs: {np.mean(grad_abs_err)}, grad rel: {np.mean(grad_relative_err)}')
    # print_memory("memory stats")
    # print(np.mean(grad_abs_err))
    
    if args.eval and rank == 0:
        if thread is not None:
            if args.inductive:
                model_copy, val_acc = thread.get()
            else:
                model_copy, val_acc, _ = thread.get()
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model_copy
        # torch.save(best_model.state_dict(), 'model/' + args.graph_name + '_final.pth.tar')
        print('model saved')
        print("Validation accuracy {:.2%}".format(best_acc))
        model_copy = copy.deepcopy(model)
        _, acc = evaluate_induc('Test Result', model_copy, test_g, 'test')

    if args.save_csv and rank == 0:
        # save epoch time to csv
        if args.fixed_synchro is not None:
            dict = {'dataset': args.dataset, 'model': [args.model, args.n_layers, args.n_hidden], 
                    'epoch': args.n_epochs, 'datatype': args.datatype, 'pipeline': args.enable_pipeline, 'fixed synchro': args.fixed_synchro, 'rank': rank,
                    'epoch time':np.mean(train_dur), 'commu': np.mean(comm_dur), 'reduce': np.mean(reduce_dur),
                    'compute': np.mean(train_dur) - np.mean(comm_dur) - np.mean(reduce_dur), 'accuracy': 'NA', 
                    'peak memory': torch.cuda.max_memory_allocated() / 1024 / 1024,
                    }
        if args.fixed_synchro is None:
            dict = {'dataset': args.dataset, 'model': [args.model, args.n_layers, args.n_hidden], 
                    'epoch': args.n_epochs, 'datatype': args.datatype, 'pipeline': args.enable_pipeline, 'fixed synchro': 'NA', 'rank': rank,
                    'epoch time':np.mean(train_dur), 'commu': np.mean(comm_dur), 'reduce': np.mean(reduce_dur),
                    'compute': np.mean(train_dur) - np.mean(comm_dur) - np.mean(reduce_dur), 'accuracy': 'NA', 
                    'peak memory': torch.cuda.max_memory_allocated() / 1024 / 1024,
                    }
        df = pd.DataFrame([dict])
        file_csv = 'results/speed_pcie4_v2/%s_n%d_2node.csv' % (args.dataset, args.n_partitions)
        if os.path.exists(file_csv):
            df.to_csv(file_csv, mode='a', header=False)
        else:
            df.to_csv(file_csv, mode='a')
    


def check_parser(args):
    if args.norm == 'none':
        args.norm = None


def init_processes(rank, size, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '%d' % args.port
    dist.init_process_group(args.backend, rank=rank, world_size=size)
    rank, size = dist.get_rank(), dist.get_world_size()
    check_parser(args)
    g, node_dict, gpb = load_partition(args, rank)
    run(g, node_dict, gpb, args)
