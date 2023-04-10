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
import time
from dgl.nn import GraphConv, JumpingKnowledge
import dgl.function as fn


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
        self.abs_err = []
        self.commu_part = None
        self.commu_part32 = None
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
        rel_err = 0
        h = feat
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    # bits=1
                    # output, scale, mn = ctx.quantize_and_pack(h, bits)
                    # h = ctx.dequantize_and_unpack(output, bits, h.shape, scale, mn)

                    h = ctx.dbuffer.update(i, h)
                    
                    # Test error-bit
                        # a = torch.sqrt(((commu_part32-commu_part)**2).sum(1))
                        # self.abs_err.append(a.view(1, -1))
                    # abs_err += torch.sqrt(((commu_part32-commu_part1)**2).sum(1)).mean()
                    # err_tmp = torch.sqrt(((commu_part32-commu_part1)**2).sum(1)) / torch.norm(commu_part32, dim=1, p=2)
                    # rel_err += err_tmp.mean()
                
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


class GraphSAGE2(GNNBase):

    def __init__(self, layer_size, activation, use_pp, dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GraphSAGE2, self).__init__(layer_size, activation, use_pp, dropout, norm, n_linear)
        self.commu_part = None
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
        rel_err = 0
        h = feat
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.training and (i > 0 or not self.use_pp):
                    # bits=1
                    # output, scale, mn = ctx.quantize_and_pack(h, bits)
                    # h = ctx.dequantize_and_unpack(output, bits, h.shape, scale, mn)
                    
                    # ctx.buffer.layer_pos = 3
                    h, commu_part, commu_part32 = ctx.buffer2.update(i, h)
                    
                    # Test error-bit
                    if i == self.n_layers - self.n_linear - 1:
                        self.commu_part = commu_part
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
                        h1 = ctx.volume_buffer.update(i, h)
                        # if self.training and (i == 1 or not self.use_pp):
                        #     h = ctx.buffer.update(i, h)
                        # elif self.training and (i > 1 or not self.use_pp):
                        #     h = ctx.buffer.fetchdata(i, h) 
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
                    # ctx.buffer.layer_pos = 4
                    h = ctx.volume_buffer.update(i, h)
                        
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


# DAGNN
class DAGNNConv(nn.Module):
    def __init__(self, in_dim, k):
        super(DAGNNConv, self).__init__()

        self.s = Parameter(torch.FloatTensor(in_dim, 1))
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):
        rank = dist.get_rank()
        with graph.local_scope():
            if self.training:
                results = [feats]

                # degs = graph.in_degrees().float()
                # norm = torch.pow(degs, -0.5)
                # norm = norm.to(feats.device).unsqueeze(1)
                
                # TODO
                dst_norm = torch.pow(
                    graph.in_degrees().float().clamp(min=1), -0.5)
                shp = dst_norm.shape + (1,) * (feats.dim() - 1)
                dst_norm = torch.reshape(dst_norm, shp).to(feats.device)

                src_norm = torch.pow(
                    graph.out_degrees().float().clamp(min=1), -0.5)
                shp = src_norm.shape + (1,) * (feats.dim() - 1)
                print('in deg', graph.in_degrees().float().shape, 'out deg', graph.out_degrees().float().shape)
                exit(0)
                src_norm = torch.reshape(src_norm, shp).to(feats.device)
                
                for i in range(1, self.k+1):
                    feats, _ = ctx.buffer.update(i, feats)
                    feats = feats * src_norm
                    graph.nodes['_U'].data['h'] = feats
                    graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                    feats = graph.nodes['_V'].data["h"]
                    feats = feats * dst_norm
                    results.append(feats)

                H = torch.stack(results, dim=1)
                S = torch.sigmoid(torch.matmul(H, self.s))
                S = S.permute(0, 2, 1)
                H = torch.matmul(S, H).squeeze()

            else:
                results = [feats]

                dst_norm = torch.pow(
                    graph.in_degrees().float().clamp(min=1), -0.5)
                shp = dst_norm.shape + (1,) * (feats.dim() - 1)
                dst_norm = torch.reshape(dst_norm, shp).to(feats.device)

                src_norm = torch.pow(
                    graph.out_degrees().float().clamp(min=1), -0.5)
                shp = src_norm.shape + (1,) * (feats.dim() - 1)
                src_norm = torch.reshape(src_norm, shp).to(feats.device)

                for _ in range(self.k):
                    feats = feats * src_norm
                    graph.ndata["h"] = feats
                    graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                    feats = graph.ndata["h"]
                    feats = feats * dst_norm
                    results.append(feats)

                H = torch.stack(results, dim=1)
                S = torch.sigmoid(torch.matmul(H, self.s))
                S = S.permute(0, 2, 1)
                H = torch.matmul(S, H).squeeze()

            return H


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, dropout=0):
        super(MLPLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        if self.activation is F.relu:
            gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):
        feats = self.dropout(feats)
        feats = self.linear(feats)
        if self.activation:
            feats = self.activation(feats)

        return feats
    

class DAGNN(nn.Module):
    def __init__(
        self,
        k,
        in_dim,
        hid_dim,
        out_dim,
        bias=True,
        activation=F.relu,
        dropout=0,
        
    ):
        super(DAGNN, self).__init__()
        self.mlp = nn.ModuleList()
        self.mlp.append(
            MLPLayer(
                in_dim=in_dim,
                out_dim=hid_dim,
                bias=bias,
                activation=activation,
                dropout=dropout,
            )
        )
        self.mlp.append(
            MLPLayer(
                in_dim=hid_dim,
                out_dim=out_dim,
                bias=bias,
                activation=None,
                dropout=dropout,
            )
        )
        self.dagnn = DAGNNConv(in_dim=out_dim, k=k)

    def forward(self, graph, feats):
        for layer in self.mlp:
            feats = layer(feats)
        feats = self.dagnn(graph, feats)
        return feats
    

# JKNet
class JKNet(nn.Module):
    def __init__(
        self, in_dim, hid_dim, out_dim, num_layers=1, mode="cat", dropout=0.0
    ):
        super(JKNet, self).__init__()

        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        for _ in range(num_layers):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))

        if self.mode == "lstm":
            self.jump = JumpingKnowledge(mode, hid_dim, num_layers)
        else:
            self.jump = JumpingKnowledge(mode)

        if self.mode == "cat":
            hid_dim = hid_dim * (num_layers + 1)

        self.output = nn.Linear(hid_dim, out_dim)
        self.reset_params()

    def reset_params(self):
        self.output.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()
        self.jump.reset_parameters()

    def forward(self, g, feats):
        if self.training:
            feat_lst = []
            layer_idx = 0
            for layer in self.layers:
                if layer_idx > 0:
                    # feats = ctx.dbuffer.update(layer_idx, feats)
                    feats = ctx.volume_buffer.update(layer_idx, feats)
                    feat_lst.append(feats)
                feats = self.dropout(layer(g, feats))
                layer_idx += 1
                
            # feats = ctx.dbuffer.update(layer_idx, feats)
            feats = ctx.volume_buffer.update(layer_idx, feats)
            feat_lst.append(feats)
            
            g.nodes['_U'].data['h'] = self.jump(feat_lst)
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            
            return self.output(g.nodes['_V'].data['h'])
    
        else:
            feat_lst = []
            for layer in self.layers:
                feats = self.dropout(layer(g, feats))
                feat_lst.append(feats)

            g.ndata["h"] = self.jump(feat_lst)
            g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))

            return self.output(g.ndata["h"])