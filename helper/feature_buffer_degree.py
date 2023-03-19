import torch
from multiprocessing.pool import ThreadPool
from multiprocessing import Event
from helper.timer.timer import *
import queue
import numpy as np
import re
import exact.cpp_extension.quantization as ext_quantization
import torch.nn.functional as F
import pandas as pd
from collections import deque


def quantize_and_pack(data, bits):
    N = data.shape[0]
    input_flatten = data.view(N, -1)
    mn, mx = torch.min(input_flatten, 1)[0], torch.max(input_flatten, 1)[0]

    # Pack to bitstream
    assert type(bits) == int
    pack_func = ext_quantization.pack_single_precision
    scale = (2 ** bits - 1) / (mx - mn) 
    output = pack_func(data, mn, mx, scale.to(data.dtype), bits, True)
    return output, scale, mn


def dequantize_and_unpack(data, bits, shape, scale, mn):
    N = shape[0]
    num_features = int(np.prod(shape[1:]))
    # Unpack bitstream
    assert type(bits) == int
    unpack_func = ext_quantization.unpack_single_precision
    data = unpack_func(data, bits, scale, mn, N, num_features)
    return data


class DegreeBuffer(object):

    def __init__(self):
        super(DegreeBuffer, self).__init__()
        self._num_in = None
        self._n_layers = 0
        self._layer_size = []
        self._pipeline = False
        self._epoch = 0
        self._feat_cpu, self._grad_cpu, self._sm_cpu, self._gsm_cpu = [], [], [], []
        self._f_buf = []
        self._f_recv, self._b_recv, self._sm_recv, self._gsm_recv = [], [], [], []
        self._f_recv_cpu, self._b_recv_cpu, self._sm_recv_cpu, self._gsm_recv_cpu = [], [], [], []
        self._f_avg, self._b_avg = [], []
        self._send_shape = []
        self._recv_shape = []
        self._pool = None
        self._comm_stream = None
        self._f_cpu_event, self._b_cpu_event = [], []
        self._f_cuda_event, self._b_cuda_event = [], []
        self._backend = None
        self._corr_momentum = 0
        self._corr_feat, self._corr_grad = False, False
        self._pl, self._pr = [], []
        self.dtype = 'int1'
        self.dtype2 = 'fp32' # the datatype changed to 
        self.nbits = []
        self.nbits2 = None
        self._selected = []
        self._grad_selected = None
        self._group_send_size = None
        self._fixed_synchro = None
        self.grad_abs_err = 0
        self.grad_rel_err = 0
        self.change_layer_b = False
        self.change_epoch_b = False
        self.layer_pos = 50


    def init_buffer(self, num_in, num_all, f_send_shape, f_recv_shape, layer_size, start_bits, qgroup_size_send_tot, qgroup_size_recv_tot, group_size_recv_tot, bdry_idx_recv_tot, 
                    use_pp=False, backend='gloo', pipeline=False, fixed_synchro=None):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._backend = backend
        self._num_in = num_in
        self._num_all = num_all
        self._n_layers = len(layer_size)
        self._layer_size = layer_size
        self._pipeline = pipeline
        self._send_shape = f_send_shape
        self._recv_shape = f_recv_shape
        self._fixed_synchro = fixed_synchro
        self._start_bits = start_bits
        self._qgroup_size_send_tot = qgroup_size_send_tot
        self._qgroup_size_recv_tot = qgroup_size_recv_tot
        self._group_size_recv_tot = group_size_recv_tot 
        self._bdry_idx_recv_tot = bdry_idx_recv_tot
        self._group_num = len(self._qgroup_size_send_tot)
        
        assign_bits = deque(start_bits)
        if backend == 'gloo':
            # CPU part buffer
            for g in range(self._group_num):
                qgroup_size_send = self._qgroup_size_send_tot[g]
                qgroup_size_recv = self._qgroup_size_recv_tot[g]

                feat_cpu_sub, grad_cpu_sub = [None] * self._n_layers, [None] * self._n_layers
                f_recv_cpu_sub, b_recv_cpu_sub, = [None] * self._n_layers, [None] * self._n_layers
                sm_cpu_sub = [None] * self._n_layers
                sm_recv_cpu_sub = [None] * self._n_layers
                gsm_cpu_sub = [None] * self._n_layers
                gsm_recv_cpu_sub = [None] * self._n_layers
                for i in range(self._n_layers):
                    if i == 0 and use_pp:
                        continue
                    tmp1, tmp2, tmp3, tmp4 = [], [], [], []
                    tmp5, tmp6, tmp7, tmp8 = [], [], [], []

                    for j in range(size):
                        if j == rank:
                            tmp1.append(None); tmp2.append(None); tmp3.append(None); tmp4.append(None)
                            tmp5.append(None); tmp6.append(None); tmp7.append(None); tmp8.append(None) 
                        else:
                            s1 = torch.Size([qgroup_size_send[j].sum().item()])
                            s2 = torch.Size([qgroup_size_recv[j].sum().item()])
                            # feature buffer init
                            tmp1.append(torch.zeros(s1, pin_memory=True, dtype=torch.int8))
                            tmp3.append(torch.zeros(s2, pin_memory=True, dtype=torch.int8))
                            # fscale & fmn buffer init (scale and mn have the same size, concatenate them together to commu)
                            s3 = torch.Size([self._send_shape[j]])
                            s4 = torch.Size([self._recv_shape[j]])
                            tmp5.append(torch.cat((torch.zeros(s3, pin_memory=True, dtype=torch.half), torch.zeros(s3, pin_memory=True, dtype=torch.half))))
                            tmp6.append(torch.cat((torch.zeros(s4, pin_memory=True, dtype=torch.half), torch.zeros(s4, pin_memory=True, dtype=torch.half))))
                            # gradient buffer init
                            tmp2.append(torch.zeros(s2, pin_memory=True, dtype=torch.int8))
                            tmp4.append(torch.zeros(s1, pin_memory=True, dtype=torch.int8))
                            # gscale & gmn buffer init
                            tmp7.append(torch.cat((torch.zeros(s4, pin_memory=True, dtype=torch.half), torch.zeros(s4, pin_memory=True, dtype=torch.half))))
                            tmp8.append(torch.cat((torch.zeros(s3, pin_memory=True, dtype=torch.half), torch.zeros(s3, pin_memory=True, dtype=torch.half))))
                    feat_cpu_sub[i] = tmp1
                    f_recv_cpu_sub[i] = tmp3
                    grad_cpu_sub[i] = tmp2
                    b_recv_cpu_sub[i] = tmp4

                    sm_cpu_sub[i] = tmp5
                    sm_recv_cpu_sub[i] = tmp6
                    gsm_cpu_sub[i] = tmp7
                    gsm_recv_cpu_sub[i] = tmp8

                self._feat_cpu.append(feat_cpu_sub)
                self._f_recv_cpu.append(f_recv_cpu_sub)
                self._grad_cpu.append(grad_cpu_sub)
                self._b_recv_cpu.append(b_recv_cpu_sub)

                self._sm_cpu.append(sm_cpu_sub)
                self._sm_recv_cpu.append(sm_recv_cpu_sub)
                self._gsm_cpu.append(gsm_cpu_sub)
                self._gsm_recv_cpu.append(gsm_recv_cpu_sub)
                
        # GPU part buffer, events, stream init
        for g in range(self._group_num):
            qgroup_size_send = self._qgroup_size_send_tot[g]
            qgroup_size_recv = self._qgroup_size_recv_tot[g]

            f_recv_sub, b_recv_sub = [None] * self._n_layers, [None] * self._n_layers
            sm_recv_sub = [None] * self._n_layers
            gsm_recv_sub = [None] * self._n_layers
            self._f_buf = [None] * self._n_layers
            self._f_cpu_event, self._b_cpu_event = [None] * self._n_layers, [None] * self._n_layers
            self._f_cuda_event, self._b_cuda_event = [None] * self._n_layers, [None] * self._n_layers
            self._comm_stream, self._corr_stream = torch.cuda.Stream(), torch.cuda.Stream()

            for i in range(self._n_layers):
                if i == 0 and use_pp:
                    continue
                self._f_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda')
                tmp1, tmp2 = [], []
                tmp3, tmp4 = [], []
                for j in range(size):
                    if j == rank:
                        tmp1.append(None); tmp2.append(None)
                        tmp3.append(None); tmp4.append(None)
                    else:
                        if self.dtype == 'fp32' or self.dtype == 'fp16':
                            s1 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                            s2 = torch.Size([f_send_shape[j], self._layer_size[i]])
                            if self.dtype == 'fp32':
                                dtype = torch.float
                            elif self.dtype == 'fp16':
                                dtype = torch.half 
                            tmp1.append(torch.zeros(s1, device='cuda', dtype=dtype))
                            tmp2.append(torch.zeros(s2, device='cuda', dtype=dtype))
                            tmp3.append(torch.zeros(s1, device='cuda'))
                            tmp4.append(torch.zeros(s2, device='cuda'))
                        
                        else:
                            s1 = torch.Size([qgroup_size_recv[j].sum().item()])
                            s2 = torch.Size([qgroup_size_send[j].sum().item()])

                            # feature gpu buffer init
                            tmp1.append(torch.zeros(s1, device='cuda', dtype=torch.int8))
                            # fscale & fmn buffer init
                            s3 = torch.Size([self._recv_shape[j]])
                            tmp3.append(torch.cat((torch.zeros(s3, device='cuda', dtype=torch.half), torch.zeros(s3, device='cuda', dtype=torch.half))))

                            # gradient gpu buffer init
                            tmp2.append(torch.zeros(s2, device='cuda', dtype=torch.int8))
                            # gscale & gmn buffer init
                            s4 = torch.Size([self._send_shape[j]])
                            tmp4.append(torch.cat((torch.zeros(s4, device='cuda', dtype=torch.half), torch.zeros(s4, device='cuda', dtype=torch.half))))
                            
                f_recv_sub[i] = tmp1
                b_recv_sub[i] = tmp2
                sm_recv_sub[i] = tmp3
                gsm_recv_sub[i] = tmp4

                self._f_cpu_event[i] = Event()
                self._b_cpu_event[i] = Event()
                self._f_cuda_event[i] = torch.cuda.Event()
                self._b_cuda_event[i] = torch.cuda.Event()
            self._f_recv.append(f_recv_sub)
            self._b_recv.append(b_recv_sub)
            self._sm_recv.append(sm_recv_sub)
            self._gsm_recv.append(gsm_recv_sub)

        self._pool = ThreadPool(processes=2*self._n_layers)
        self.__init_pl_pr()
            

    def set_selected(self, selected):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._selected = selected
        
        self._grad_selected = []
        self._group_size_send_tot = []
        for g in range(self._group_num):
            selected_sub = self._selected[g]
            grad_selected_sub = [None] * size
            group_size_send = [None] * size
            pos_idx = self._num_in
            for i in range(size):
                if i == rank:
                    continue
                bdry_real_idx = torch.split(self._bdry_idx_recv_tot[g][i]+pos_idx, self._group_size_recv_tot[g][i].numpy().tolist(), dim=0)
                grad_selected_sub[i] = list(bdry_real_idx)
                group_size_send[i] = torch.tensor([part_id.shape[0] for part_id in selected_sub[i]])
                pos_idx += self._recv_shape[i]
            self._grad_selected.append(grad_selected_sub)
            self._group_size_send_tot.append(group_size_send)


    def adjust_buffer(self, base_bit):
        self.curr_g = torch.nonzero(torch.tensor(self._start_bits)==base_bit, as_tuple=True)[0].item()


    def change_epoch_bit(self):
        self.change_epoch_b = True


    def __init_pl_pr(self):
        self._pl, self._pr = [], []
        tot = self._num_in
        for s in self._recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)


    def next_epoch(self):
        self._epoch += 1


    def set_pipeline(self):
        if self._fixed_synchro is not None:
            if (self._epoch+1) % self._fixed_synchro == 0:
                self._pipeline = False
            else:
                self._pipeline = True


    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        t_dequant = 0
        if not self.change_layer_b and not self.change_epoch_b:
            with quant_timer.timer(f'fdequant_{layer}'):
                for i in range(size):
                    if i != rank:
                        shape = torch.Size([self._recv_shape[i], self._layer_size[layer]])
                        if self._pipeline and self._epoch == 0:
                            tmp.append(torch.zeros(shape, device='cuda'))
                        else:
                            # Decode data, scale, mn
                            data_tot = torch.split(self._f_recv[layer][i], self._qgroup_recv_size[i].numpy().tolist(), dim=0)
                            sm = torch.chunk(self._sm_recv[layer][i], 2, dim=0)
                            scale_tot = torch.split(sm[0].float(), self._group_recv_size[i].numpy().tolist(), dim=0)
                            mn_tot = torch.split(sm[1].float(), self._group_recv_size[i].numpy().tolist(), dim=0)
                            
                            data_tmp = []
                            bdry_feat = torch.zeros(shape, device='cuda')
                            for k in range(len(self.nbits)): 
                                group_shape = torch.Size([self._group_recv_size[i][k], self._layer_size[layer]])
                                sub_data = dequantize_and_unpack(data_tot[k], self.nbits[k], group_shape, 
                                            scale_tot[k], mn_tot[k])
                                data_tmp.append(sub_data)

                            # Reorder feature back
                            bdry_feat[self._bdry_idx_recv[i]] = torch.cat(data_tmp)
                            tmp.append(bdry_feat)

        else:
            with quant_timer.timer(f'fdequant_{layer}'):
                for i in range(size):
                    if i != rank:
                        if self.dtype2 == 'fp32' or self.dtype2 == 'fp16':
                            tmp.append(self._f_recv_e1[layer][i].float())

                        else:
                            shape = torch.Size([self._recv_shape[i], self._layer_size[layer]])
                            if self._pipeline and self._epoch == 0:
                                tmp.append(torch.zeros(shape, device='cuda'))
                            else:
                                # tmp.append(dequantize_and_unpack(self._f_recv[layer][i], self.nbits, shape, 
                                #             self._scale_recv[layer][i].float(), self._mn_recv[layer][i].float()))
                                sm = torch.chunk(self._sm_recv_e1[layer][i], 2, dim=0)
                                subpart = dequantize_and_unpack(self._f_recv_e1[layer][i], self.nbits2, shape, 
                                            sm[0].float(), sm[1].float())
                                tmp.append(subpart)
        return torch.cat(tmp)


    def update(self, layer, feat):
        torch.cuda.current_stream().synchronize()
        if layer == self.layer_pos:
            self.change_layer_b = True
        if self._pipeline is False:
            with comm_timer.timer(f'forward_{layer}'):
                self.__feat_transfer(self._epoch, layer, feat)
                torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
            self._f_buf[layer] = self.__feat_concat(layer, feat)
            
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
            return self._f_buf[layer]
        else:
            if self._epoch > 0:
                with comm_timer.timer(f'forward_{layer}'):
                    self._f_cpu_event[layer].wait()
                    torch.cuda.current_stream().wait_event(self._f_cuda_event[layer])
                    self._f_cpu_event[layer].clear()
            self._f_buf[layer], commu_part = self.__feat_concat(layer, feat)
            self._pool.apply_async(self.__feat_transfer, args=(self._epoch, layer, feat))
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
            return self._f_buf[layer]


    def __gloo_all_to_all(self, send_gpu, send_cpu, recv_cpu, recv_gpu, tag, forward=True):
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()
        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size
            r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
            req2.put((r2, left))
   
            send_cpu[right].copy_(send_gpu[right])    
            r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
            req1.append(r1)
        while not req2.empty():
            r, idx = req2.get()
            # TODO: if r.is_completed() run following lines else next r (see issue #30723)
            r.wait()
            recv_gpu[idx].copy_(recv_cpu[idx], non_blocking=True)
            
        # 如果删掉，pipeline模式会卡
        for r in req1:
            r.wait()


    def __feat_transfer(self, epoch, layer, feat):
        # 卡住的话可能是某一步/子进程里面有报错
        tag = epoch * 2 * self._n_layers + layer
        if not self.change_layer_b and not self.change_epoch_b:
            with quant_timer.timer(f'fquant_{layer}'):
                quant_feat, feat_sm = self.__quant_data_ada(feat, forward=True)
            if self._backend == 'gloo':
                self._comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._comm_stream):
                    if feat_sm is None:
                        self.__gloo_all_to_all(quant_feat, self._feat_cpu[layer], self._f_recv_cpu[layer], self._f_recv[layer],
                                                tag, forward=True)
                    else:
                        self.__gloo_all_to_all(quant_feat, self._feat_cpu[layer], self._f_recv_cpu[layer], 
                                                self._f_recv[layer], tag, forward=True)
                        self.__gloo_all_to_all(feat_sm, self._sm_cpu[layer], self._sm_recv_cpu[layer], 
                                                self._sm_recv[layer], (tag+100), forward=True)
                
                self._f_cuda_event[layer].record(self._comm_stream)
                if self._corr_feat:
                    self._f_cuda_event[layer].record(self._corr_stream)
            else:
                raise NotImplementedError
        else:     
            with quant_timer.timer(f'fquant_{layer}'):
                quant_feat, feat_sm = self.__quant_data(feat, forward=True, data_type=self.dtype2, nbit=self.nbits2)
            if self._backend == 'gloo':
                self._comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._comm_stream):
                    if feat_sm is None:
                        self.__gloo_all_to_all(self.dtype2, quant_feat, self._feat_cpu_e1[layer], self._f_recv_cpu_e1[layer], self._f_recv_e1[layer],
                                                tag, self._corr_feat, self._f_avg[layer], forward=True)
                    else:
                        self.__gloo_all_to_all(self.dtype2, quant_feat, self._feat_cpu_e1[layer], self._f_recv_cpu_e1[layer], 
                                                self._f_recv_e1[layer], tag, self._corr_feat, self._f_avg[layer], forward=True)
                        self.__gloo_all_to_all(self.dtype, feat_sm, self._sm_cpu_e1[layer], self._sm_recv_cpu_e1[layer], 
                                                self._sm_recv_e1[layer], (tag+100), self._corr_feat, self._f_avg[layer], forward=True)
                        
                self._f_cuda_event[layer].record(self._comm_stream)
                if self._corr_feat:
                    self._f_cuda_event[layer].record(self._corr_stream)
            else:
                raise NotImplementedError
        self._f_cpu_event[layer].set()
        

    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        with quant_timer.timer(f'bdequant_{layer}'):
            if not self.change_epoch_b and not self.change_layer_b:
                for i in range(size):
                    if i != rank:
                        if self._pipeline and self._epoch == 0:
                            shape = torch.Size([self._send_shape[i], self._layer_size[layer]])
                            grad[self._selected[i]] += torch.zeros(shape, device='cuda')
                        else:
                            # Decode gradient, scale, mn
                            grad_tot = torch.split(self._b_recv[layer][i], self._qgroup_send_size[i].numpy().tolist(), dim=0)
                            gsm = torch.chunk(self._gsm_recv[layer][i], 2, dim=0)
                            gscale_tot = torch.split(gsm[0].float(), self._group_send_size[i].numpy().tolist(), dim=0)
                            gmn_tot = torch.split(gsm[1].float(), self._group_send_size[i].numpy().tolist(), dim=0)
                            
                            grad_tmp = []
                            for k in range(len(self.nbits)):
                                shape = torch.Size([self._group_send_size[i][k], self._layer_size[layer]])
                                sub_grad = dequantize_and_unpack(grad_tot[k], self.nbits[k], shape, 
                                            gscale_tot[k], gmn_tot[k])
                                grad_tmp.append(sub_grad)

                            # Reorder gradients back
                            grad[torch.cat(self._selected[i])] += torch.cat(grad_tmp)
     
            else:
                for i in range(size):
                    if i != rank:
                        if self.dtype2 == 'fp32' or self.dtype2 == 'fp16':
                            grad[self._selected[i]] += self._b_recv_e1[layer][i].float()
                        else:
                            shape = torch.Size([self._send_shape[i], self._layer_size[layer]])
                            if self._pipeline and self._epoch == 0:
                                grad[self._selected[i]] += torch.zeros(shape, device='cuda')
                            else:
                                gsm = torch.chunk(self._gsm_recv_e1[layer][i], 2, dim=0)
                                commu_part = dequantize_and_unpack(self._b_recv_e1[layer][i], self.nbits2, 
                                                                            shape, gsm[0].float(), gsm[1].float())
                                grad[self._selected[i]] += commu_part


    def __grad_hook(self, epoch, layer):
        def fn(grad):
            grad = grad.contiguous()
            if layer < self.layer_pos:
                self.change_layer_b = False
            torch.cuda.current_stream().synchronize()
            if self._pipeline is False:
                with comm_timer.timer(f'backward_{layer}'):
                    self.__grad_transfer(epoch, layer, grad)
                    torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                self.__update_grad(layer, grad)
                return grad
            else:
                if self._epoch > 0:
                    with comm_timer.timer(f'backward_{layer}'):
                        self._b_cpu_event[layer].wait()
                        torch.cuda.current_stream().wait_event(self._b_cuda_event[layer])
                        self._b_cpu_event[layer].clear()
                self.__update_grad(layer, grad)
                self._pool.apply_async(self.__grad_transfer, args=(epoch, layer, grad))
                return grad
        return fn


    def __grad_transfer(self, epoch, layer, grad):
        tag = epoch * 2 * self._n_layers + layer + self._n_layers
        if not self.change_layer_b and not self.change_epoch_b:
            data_type = self.dtype
            nbit = self.nbits
        else:
            data_type = self.dtype2
            nbit = self.nbits2
            
        with quant_timer.timer(f'bquant_{layer}'):
            quant_grad, grad_sm = self.__quant_data_ada(grad, forward=False)
        if self._backend == 'gloo':
            self._comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._comm_stream):
                if not self.change_layer_b and not self.change_epoch_b:
                    if grad_sm is None:
                        self.__gloo_all_to_all(quant_grad, self._grad_cpu[layer], self._b_recv_cpu[layer], self._b_recv[layer],
                                        tag, forward=False)
                    else:
                        self.__gloo_all_to_all(quant_grad, self._grad_cpu[layer], self._b_recv_cpu[layer], 
                                                self._b_recv[layer], tag, forward=False)
                        self.__gloo_all_to_all(grad_sm, self._gsm_cpu[layer], self._gsm_recv_cpu[layer],
                                                self._gsm_recv[layer], (tag+100), forward=False)
                    # transfer one more fp32 gradient
                    # self.__gloo_all_to_all(grad, self._grad_cpu32[layer], self._b_recv_cpu32[layer], self._b_recv32[layer],
                    #                     tag, self._corr_grad, self._b_avg[layer], forward=False)
                else:
                    if grad_sm is None:
                        self.__gloo_all_to_all(data_type, quant_grad, self._grad_cpu_e1[layer], self._b_recv_cpu_e1[layer], self._b_recv_e1[layer],
                                        tag, self._corr_grad, self._b_avg[layer], forward=False)
                    else:
                        self.__gloo_all_to_all(data_type, quant_grad, self._grad_cpu_e1[layer], self._b_recv_cpu_e1[layer], 
                                                self._b_recv_e1[layer], tag, self._corr_grad, self._b_avg[layer], forward=False)
                        self.__gloo_all_to_all(data_type, grad_sm, self._gsm_cpu_e1[layer], self._gsm_recv_cpu_e1[layer],
                                                self._gsm_recv_e1[layer], (tag+100), self._corr_grad, self._b_avg[layer], forward=False)
                
            self._b_cuda_event[layer].record(self._comm_stream)
            if self._corr_grad:
                self._b_cuda_event[layer].record(self._corr_stream)
        else:
            raise NotImplementedError
        self._b_cpu_event[layer].set()


    def __quant_data(self, data, forward, data_type, nbit):
        rank, size = dist.get_rank(), dist.get_world_size()
        if data_type == 'fp32':
            quant_data = data
            data_sm = None
        elif data_type == 'fp16':
            quant_data = data.half()
            data_sm = None
        # elif data_type == 'int8':
        #     quant_data= data * 100
        #     quant_data = quant_data.char()
        #     data_scale = None
        #     data_mn = None
        else:
            quant_data, data_sm = [], []
            for i in range(size):
                if i == rank:
                    quant_data.append(None)
                    data_sm.append(None)
                else:
                    if forward:
                        data_part, scale, mn = quantize_and_pack(data[self._selected[i]], nbit)
                    else:
                        data_part, scale, mn = quantize_and_pack(data[self._pl[i]:self._pr[i]], nbit)
                    quant_data.append(data_part)
                    data_sm.append(torch.cat((scale.half(), mn.half())))
        
        return quant_data, data_sm
    
    
    def fetchdata(self, layer, h):
        # Complete features when only tranfer last layer 
        tmp = [h]
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i != rank:
                shape = torch.Size([self._recv_shape[i], self._layer_size[layer]])
                tmp.append(torch.zeros(shape, device='cuda'))

        return torch.cat(tmp)
    
    
    def __quant_data_ada(self, data, forward):
        rank, size = dist.get_rank(), dist.get_world_size()

        quant_data, data_sm = [], []
        for i in range(size):
            if i == rank:
                quant_data.append(None)
                data_sm.append(None)
            else:
                data_tmp = []
                scale_tmp = []
                mn_tmp = []
                for k in range(len(self.nbits)):
                    nbit = self.nbits[k]
                    if forward:
                        data_part, scale, mn = quantize_and_pack(data[self._selected[i][k]], nbit)
                    else:
                        data_part, scale, mn = quantize_and_pack(data[self._grad_selected[i][k]], nbit)
                    data_tmp.append(data_part)
                    scale_tmp.append(scale.half())
                    mn_tmp.append(mn.half())
                    
                sm_tmp = scale_tmp + mn_tmp
                quant_data.append(torch.cat(data_tmp))
                data_sm.append(torch.cat(sm_tmp))
        
        return quant_data, data_sm