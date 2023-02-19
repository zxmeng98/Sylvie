from module.layer import *
import dgl
from torch import nn
import torch.nn.functional as F
from module.sync_bn import SyncBatchNorm
from helper import context as ctx
# from qtorch.quant import float_quantize
from myconfig import *
import torch.distributed as dist
from dgl.nn.pytorch.conv import APPNPConv



class GNNBase(nn.Module):

    def __init__(self, layer_size, activation, use_pp=False, dropout=0.5, norm='layer', n_linear=0):
        super(GNNBase, self).__init__()
        self.n_layers = len(layer_size) - 1
        self.layers = nn.ModuleList()
        self.activation = activation
        self.use_pp = use_pp
        self.n_linear = n_linear

        if norm is None:
            self.use_norm = False
        else:
            self.use_norm = True
            self.norm = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)


class GraphSAGE(GNNBase):

    def __init__(self, layer_size, activation, use_pp, dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GraphSAGE, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(GraphSAGELayer(layer_size[i], layer_size[i + 1], use_pp=use_pp))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_deg=None):
        rank = dist.get_rank()
        abs_err = 0
        rel_err = 0
        h = feat
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    # bits=1
                    # output, scale, mn = ctx.quantize_and_pack(h, bits)
                    # h = ctx.dequantize_and_unpack(output, bits, h.shape, scale, mn)
                    h, commu_part32 = ctx.buffer.update(i, h)
                        
                    # h, commu_part32 = ctx.buffer.update(i, h)
                        
                    # abs_err += torch.sqrt(((commu_part32-commu_part1)**2).sum(1)).mean()
                    # err_tmp = torch.sqrt(((commu_part32-commu_part1)**2).sum(1)) / torch.norm(commu_part32, dim=1, p=2)
                    # rel_err += err_tmp.mean()
                
                # If only communicate the last layer
                # if self.training and (i == self.n_layers - self.n_linear - 1 or not self.use_pp):
                #     h = ctx.buffer.update(i, h)
                # elif self.training and (i < self.n_layers - self.n_linear - 1 or not self.use_pp) and i > 0:
                #     h = ctx.buffer.fetchdata(i, h)               
                
                # If only communicate the second layer
                # if self.training and (i == 1 or not self.use_pp):
                #     h = ctx.buffer.update(i, h)
                # elif self.training and (i > 1 or not self.use_pp):
                #     h = ctx.buffer.fetchdata(i, h)             
                    
                h = self.dropout(h)
                h = self.layers[i](g, h, in_deg)
                # if rank == 0:
                #     print(f'after 1st layer: {h.shape}')
            else:
                h = self.dropout(h)
                h = self.layers[i](h)

            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)
        return h


class GAT(GNNBase):

    def __init__(self, layer_size, activation, use_pp, heads=1, dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GAT, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(dgl.nn.GATConv(layer_size[i], layer_size[i + 1], heads[i]))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))

    def forward(self, g, feat):
        rank = dist.get_rank()
        h = feat
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.training:
                    if i > 0 or not self.use_pp:
                        # bits=1
                        # output, scale, mn = ctx.quantize_and_pack(h, bits)
                        # h = ctx.dequantize_and_unpack(output, bits, h.shape, scale, mn)
                        # h1 = ctx.buffer.update(i, h)
                        if self.training and (i == 1 or not self.use_pp):
                            h = ctx.buffer.update(i, h)
                        elif self.training and (i > 1 or not self.use_pp):
                            h = ctx.buffer.fetchdata(i, h) 
                    else:
                        h1 = h
                        h = h[0:g.num_nodes('_V')]
                    h = self.layers[i](g, (h1, h))
                else:
                    h = self.layers[i](g, h)
                h = h.mean(1)
            else:
                h = self.dropout(h)
                h = self.layers[i](h)
            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)
        return h


class GCN(GNNBase):

    def __init__(self, layer_size, activation, use_pp, dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GCN, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                self.layers.append(dgl.nn.GraphConv(layer_size[i], layer_size[i + 1]))
            else:
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))
            use_pp = False

    def forward(self, g, feat, in_deg=None):
        rank = dist.get_rank()
        h = feat
        for i in range(self.n_layers):
            h = self.dropout(h)
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    # bits=1
                    # output, scale, mn = ctx.quantize_and_pack(h, bits)
                    # h = ctx.dequantize_and_unpack(output, bits, h.shape, scale, mn)

                    h, commu_part1 = ctx.buffer.update(i, h)
                        
                    # h, commu_part32 = ctx.buffer.update(i, h)
                # if self.training and (i == 1 or not self.use_pp):
                #     h = ctx.buffer.update(i, h)
                # elif self.training and (i > 1 or not self.use_pp):
                #     h = ctx.buffer.fetchdata(i, h)    
                
                # # If only communicate the last layer
                # if self.training and (i == self.n_layers - self.n_linear - 1 or not self.use_pp):
                #     h = ctx.buffer.update(i, h)
                # elif self.training and (i < self.n_layers - self.n_linear - 1 or not self.use_pp) and i > 0:
                #     h = ctx.buffer.fetchdata(i, h)    

                    
                h = self.layers[i](g, h)

            else:
                h = self.layers[i](h)

            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)

        return h


class APPNP(nn.Module):
    
    def __init__(
        self,
        in_feats,
        hiddens,
        n_classes,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        rank = dist.get_rank()
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.propagate(g, h)
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        # if rank == 0:
        #     print(h.shape)
        # h, _ = ctx.buffer.update(1, h)
        # if rank == 0:
        #     print(h.shape)
        # exit(0)
        
        return h


class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block

        self.checkpoint_grad = False

        in_channels = args.in_channels
        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        norm = args.norm
        mlp_layers = args.mlp_layers

        if aggr in ['softmax_sg', 'softmax', 'power'] and self.num_layers > 3:
            self.checkpoint_grad = False
            self.ckp_k = self.num_layers // 2

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.node_features_encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

    def forward(self,  x, edge_index):

        h = self.node_features_encoder(x)

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index)

            if self.checkpoint_grad:

                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

                    if layer % self.ckp_k != 0:
                        res = checkpoint(self.gcns[layer], h2, edge_index)
                        h = res + h
                    else:
                        h = self.gcns[layer](h2, edge_index) + h

            else:
                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    h = self.gcns[layer](h2, edge_index) + h

            h = F.relu(self.norms[self.num_layers - 1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index)
                h2 = self.norms[layer](h1)
                h = F.relu(h2)
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')

        h = self.node_pred_linear(h)

        return torch.log_softmax(h, dim=-1)

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))