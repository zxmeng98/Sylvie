# Add communication volume record
from matplotlib.pyplot import sca
import torch
from multiprocessing.pool import ThreadPool
from multiprocessing import Event
from helper.timer.timer import *
import queue
import numpy as np
import re
import exact.cpp_extension.quantization as ext_quantization


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



class CvolumeBuffer(object):

    def __init__(self):
        super(CvolumeBuffer, self).__init__()
        self._num_in = None
        self._n_layers = 0
        self._layer_size = []
        self._pipeline = False
        self._epoch = 0
        self._feat_cpu, self._grad_cpu, self._scale_cpu, self._mn_cpu, self._gscale_cpu, self._gmn_cpu = None, None, None, None, None, None
        self._f_buf = []
        self._f_recv, self._b_recv, self._scale_recv, self._mn_recv, self._gscale_recv, self._gmn_recv = None, None, None, None, None, None
        self._f_recv_cpu, self._b_recv_cpu, self._scale_recv_cpu, self._mn_recv_cpu, self._gscale_recv_cpu, self._gmn_recv_cpu = None, None, None, None, None, None
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
        self.dtype = torch.float
        self.nbits = None
        self._selected = []
        self._fixed_synchro = None
        self.commu_volume = 0


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


    def init_buffer(self, num_in, num_all, f_send_shape, f_recv_shape, layer_size, use_pp=False, backend='gloo',
                    dtype=torch.float, pipeline=False, corr_feat=False, corr_grad=False, corr_momentum=0, fixed_synchro=None):
        rank, size = dist.get_rank(), dist.get_world_size()
        self._num_in = num_in
        self._n_layers = len(layer_size)
        self._layer_size = layer_size
        self._pipeline = pipeline
        self._epoch = 0
        self._send_shape = f_send_shape
        self._recv_shape = f_recv_shape
        self.dtype = dtype
        self._fixed_synchro = fixed_synchro

        if backend == 'gloo':
            self._feat_cpu, self._grad_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._f_recv_cpu, self._b_recv_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._scale_cpu, self._mn_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._scale_recv_cpu, self._mn_recv_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._gscale_cpu, self._gmn_cpu = [None] * self._n_layers, [None] * self._n_layers
            self._gscale_recv_cpu, self._gmn_recv_cpu = [None] * self._n_layers, [None] * self._n_layers
            for i in range(self._n_layers):
                if i == 0 and use_pp:
                    continue
                tmp1, tmp2, tmp3, tmp4 = [], [], [], []
                tmp5, tmp6, tmp7, tmp8 = [], [], [], []
                tmp9, tmp10, tmp11, tmp12 = [], [], [], []
                for j in range(size):
                    if j == rank:
                        tmp1.append(None); tmp2.append(None); tmp3.append(None); tmp4.append(None)
                        tmp5.append(None); tmp6.append(None); tmp7.append(None); tmp8.append(None)
                        tmp9.append(None); tmp10.append(None); tmp11.append(None); tmp12.append(None)
                    else:
                        s1 = torch.Size([f_send_shape[j], self._layer_size[i]])
                        s2 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                        if self.dtype == 'int4' or self.dtype == 'int2' or self.dtype == 'int1':
                            self.nbits = int(re.findall(r"\d+", self.dtype)[0])
                            fake_send = torch.randn(f_send_shape[j], self._layer_size[i]).cuda()
                            fake_recv = torch.randn(f_recv_shape[j], self._layer_size[i]).cuda()
                            quant_feat_send, scale_send, mn_send = quantize_and_pack(fake_send, self.nbits)
                            quant_feat_recv, scale_recv, mn_recv = quantize_and_pack(fake_recv, self.nbits)
                            s1 = quant_feat_send.shape
                            s2 = quant_feat_recv.shape
                            # feature buffer init
                            tmp1.append(torch.zeros(s1, pin_memory=True, dtype=torch.int8))
                            tmp3.append(torch.zeros(s2, pin_memory=True, dtype=torch.int8))
                            # fcale buffer init
                            s3 = scale_send.shape
                            s4 = scale_recv.shape
                            tmp5.append(torch.zeros(s3, pin_memory=True, dtype=torch.half))
                            tmp6.append(torch.zeros(s4, pin_memory=True, dtype=torch.half))
                            # fmn buffer init
                            tmp7.append(torch.zeros(s3, pin_memory=True, dtype=torch.half))
                            tmp8.append(torch.zeros(s4, pin_memory=True, dtype=torch.half))
                            # gradient buffer init
                            tmp2.append(torch.zeros(s2, pin_memory=True, dtype=torch.int8))
                            tmp4.append(torch.zeros(s1, pin_memory=True, dtype=torch.int8))
                            # gscale buffer init
                            tmp9.append(torch.zeros(s4, pin_memory=True, dtype=torch.half))
                            tmp10.append(torch.zeros(s3, pin_memory=True, dtype=torch.half))
                            # gmn buffer init
                            tmp11.append(torch.zeros(s4, pin_memory=True, dtype=torch.half))
                            tmp12.append(torch.zeros(s3, pin_memory=True, dtype=torch.half))
                            self._scale_cpu[i] = tmp5
                            self._scale_recv_cpu[i] = tmp6
                            self._mn_cpu[i] = tmp7
                            self._mn_recv_cpu[i] = tmp8
                            self._gscale_cpu[i] = tmp9
                            self._gscale_recv_cpu[i] = tmp10
                            self._gmn_cpu[i] = tmp11
                            self._gmn_recv_cpu[i] = tmp12
                        else:
                            if self.dtype == 'fp32':
                                dtype = torch.float
                            elif self.dtype == 'fp16':
                                dtype = torch.half
                            elif self.dtype == 'int8':
                                dtype = torch.int8
                            tmp1.append(torch.zeros(s1, pin_memory=True, dtype=dtype))
                            tmp2.append(torch.zeros(s2, pin_memory=True, dtype=dtype))
                            tmp3.append(torch.zeros(s2, pin_memory=True, dtype=dtype))
                            tmp4.append(torch.zeros(s1, pin_memory=True, dtype=dtype))
                self._feat_cpu[i] = tmp1
                self._f_recv_cpu[i] = tmp3
                self._grad_cpu[i] = tmp2
                self._b_recv_cpu[i] = tmp4

        
        # Prepare list for feature buffer, events and streams
        self._backend = backend
        self._f_buf = [None] * self._n_layers
        self._f_avg, self._b_avg = [None] * self._n_layers, [None] * self._n_layers
        self._f_recv, self._b_recv = [None] * self._n_layers, [None] * self._n_layers
        self._scale_recv, self._mn_recv = [None] * self._n_layers, [None] * self._n_layers
        self._gscale_recv, self._gmn_recv = [None] * self._n_layers, [None] * self._n_layers
        self._f_cpu_event, self._b_cpu_event = [None] * self._n_layers, [None] * self._n_layers
        self._f_cuda_event, self._b_cuda_event = [None] * self._n_layers, [None] * self._n_layers
        self._comm_stream, self._corr_stream = torch.cuda.Stream(), torch.cuda.Stream()

        for i in range(self._n_layers):
            if i == 0 and use_pp:
                continue
            self._f_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda')
            tmp1, tmp2 = [], []
            tmp3, tmp4 = [], []
            tmp5, tmp6 = [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None); tmp2.append(None)
                    tmp3.append(None); tmp4.append(None)
                    tmp5.append(None); tmp6.append(None)
                else:
                    s1 = torch.Size([f_recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([f_send_shape[j], self._layer_size[i]])
                    if self.dtype == 'int4' or self.dtype == 'int2' or self.dtype == 'int1':
                        fake_send = torch.randn(f_send_shape[j], self._layer_size[i]).cuda()
                        fake_recv = torch.randn(f_recv_shape[j], self._layer_size[i]).cuda()
                        quant_feat_send, scale_send, mn_send = quantize_and_pack(fake_send, self.nbits)
                        quant_feat_recv, scale_recv, mn_recv = quantize_and_pack(fake_recv, self.nbits)
                        s1 = quant_feat_recv.shape
                        # feature gpu buffer init
                        tmp1.append(torch.zeros(s1, device='cuda', dtype=torch.int8))
                        # scale gpu buffer init
                        s2 = scale_recv.shape
                        tmp3.append(torch.zeros(s2, device='cuda', dtype=torch.half))
                        # mn gpu buffer init
                        tmp4.append(torch.zeros(s2, device='cuda', dtype=torch.half))
                        # gradient gpu buffer init
                        s3 = quant_feat_send.shape
                        tmp2.append(torch.zeros(s3, device='cuda', dtype=torch.int8))
                        # gscale gpu buffer init
                        s4 = scale_send.shape
                        tmp5.append(torch.zeros(s4, device='cuda', dtype=torch.half))
                        # gmn gpu buffer init
                        tmp6.append(torch.zeros(s4, device='cuda', dtype=torch.half))
                        self._scale_recv[i] = tmp3
                        self._mn_recv[i] = tmp4
                        self._gscale_recv[i] = tmp5
                        self._gmn_recv[i] = tmp6
                    else:
                        if self.dtype == 'fp32':
                            dtype = torch.float
                        elif self.dtype == 'fp16':
                            dtype = torch.half
                        elif self.dtype == 'int8':
                            dtype = torch.int8
                        tmp1.append(torch.zeros(s1, device='cuda', dtype=dtype))
                        tmp2.append(torch.zeros(s2, device='cuda', dtype=dtype))
                        tmp3.append(torch.zeros(s1, device='cuda'))
                        tmp4.append(torch.zeros(s2, device='cuda'))
            self._f_recv[i] = tmp1
            self._b_recv[i] = tmp2

            if corr_feat:
                self._f_avg[i] = tmp3
            if corr_grad and i > 0:
                self._b_avg[i] = tmp4

            self._f_cpu_event[i] = Event()
            self._b_cpu_event[i] = Event()
            self._f_cuda_event[i] = torch.cuda.Event()
            self._b_cuda_event[i] = torch.cuda.Event()
        self._corr_momentum = corr_momentum
        self._corr_feat, self._corr_grad = corr_feat, corr_grad
        self._pool = ThreadPool(processes=2*self._n_layers)
        self.__init_pl_pr()


    def set_selected(self, selected):
        self._selected = selected


    def next_epoch(self):
        self._epoch += 1
        self.commu_volume = 0


    def set_pipeline(self):
        if self._fixed_synchro is not None:
            if (self._epoch+1) % self._fixed_synchro == 0:
                self._pipeline = False
            else:
                self._pipeline = True


    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                if self.dtype == 'int4' or self.dtype == 'int2' or self.dtype == 'int1':
                    shape = torch.Size([self._recv_shape[i], self._layer_size[layer]])
                    if self._epoch == 0:
                        tmp.append(torch.zeros(shape, device='cuda'))
                    else:
                        tmp.append(dequantize_and_unpack(self._f_recv[layer][i], self.nbits, shape, 
                                    self._scale_recv[layer][i].float(), self._mn_recv[layer][i].float()))
                else: 
                    if self.dtype == 'int8':
                        tmp.append(self._f_recv[layer][i].float() * 0.01)
                    else:
                        tmp.append(self._f_recv[layer][i].float())
        return torch.cat(tmp)


    def update(self, layer, feat):
        torch.cuda.current_stream().synchronize()
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
            self._f_buf[layer] = self.__feat_concat(layer, feat)
            self._pool.apply_async(self.__feat_transfer, args=(self._epoch, layer, feat))
            if self._f_buf[layer].requires_grad:
                self._f_buf[layer].register_hook(self.__grad_hook(self._epoch, layer))
            return self._f_buf[layer]


    def __gloo_all_to_all(self, send_gpu, send_cpu, recv_cpu, recv_gpu, tag, corr, avg=None, forward=True):
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()
        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size
            r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
            req2.put((r2, left))
            if forward:
                if self.dtype == 'int4' or self.dtype == 'int2' or self.dtype == 'int1':
                    send_cpu[right].copy_(send_gpu[right]) 
                else:
                    send_cpu[right].copy_(send_gpu[self._selected[right]])
            else:
                if self.dtype == 'int4' or self.dtype == 'int2' or self .dtype == 'int1':
                    send_cpu[right].copy_(send_gpu[right])
                else:
                    send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]])
            r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
            req1.append(r1)
        while not req2.empty():
            r, idx = req2.get()
            # TODO: if r.is_completed() run following lines else next r (see issue #30723)
            r.wait()
            recv_gpu[idx].copy_(recv_cpu[idx], non_blocking=True)
            
            if corr:
                with torch.cuda.stream(self._corr_stream):
                    self._corr_stream.wait_stream(self._comm_stream)
                    t = avg[idx]
                    t *= self._corr_momentum
                    t += (1 - self._corr_momentum) * recv_gpu[idx]
        # 如果删掉，pipeline模式会卡
        for r in req1:
            r.wait()


    def __feat_transfer(self, epoch, layer, feat):
        # 卡住的话可能是某一步/子进程里面有报错
        tag = epoch * 2 * self._n_layers + layer
        quant_feat, feat_scale, feat_mn = self.__quant_data(feat, forward=True)
        if self._backend == 'gloo':
            self._comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._comm_stream):
                if feat_scale is None and feat_mn is None:
                    self.__gloo_all_to_all(quant_feat, self._feat_cpu[layer], self._f_recv_cpu[layer], self._f_recv[layer],
                                            tag, self._corr_feat, self._f_avg[layer], forward=True)
                else:
                    self.__gloo_all_to_all(quant_feat, self._feat_cpu[layer], self._f_recv_cpu[layer], 
                                            self._f_recv[layer], tag, self._corr_feat, self._f_avg[layer], forward=True)
                    self.__gloo_all_to_all(feat_scale, self._scale_cpu[layer], self._scale_recv_cpu[layer], 
                                            self._scale_recv[layer], (tag+100), self._corr_feat, self._f_avg[layer], forward=True)
                    self.__gloo_all_to_all(feat_mn, self._mn_cpu[layer], self._mn_recv_cpu[layer], 
                                            self._mn_recv[layer], (tag+200), self._corr_feat, self._f_avg[layer], forward=True)
            self._f_cuda_event[layer].record(self._comm_stream)
            if self._corr_feat:
                self._f_cuda_event[layer].record(self._corr_stream)
        else:
            raise NotImplementedError
        self._f_cpu_event[layer].set()


    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i != rank:
                if self.dtype == 'int4' or self.dtype == 'int2' or self.dtype == 'int1':
                    shape = torch.Size([self._send_shape[i], self._layer_size[layer]])
                    if self._epoch == 0:
                        grad[self._selected[i]] += torch.zeros(shape, device='cuda')
                    else:
                        grad[self._selected[i]] += dequantize_and_unpack(self._b_recv[layer][i], self.nbits, 
                                                                    shape, self._gscale_recv[layer][i].float(), self._gmn_recv[layer][i].float()) 
                else:
                    if self.dtype == 'int8':
                        grad[self._selected[i]] += self._b_recv[layer][i].float() * 0.01
                    else:
                        grad[self._selected[i]] += self._b_recv[layer][i].float()


    def __grad_hook(self, epoch, layer):
        def fn(grad):
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
        quant_grad, grad_scale, grad_mn = self.__quant_data(grad, forward=False)
        if self._backend == 'gloo':
            self._comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._comm_stream):
                if grad_scale is None and grad_mn is None:
                    self.__gloo_all_to_all(quant_grad, self._grad_cpu[layer], self._b_recv_cpu[layer], self._b_recv[layer],
                                    tag, self._corr_grad, self._b_avg[layer], forward=False)
                else:
                    self.__gloo_all_to_all(quant_grad, self._grad_cpu[layer], self._b_recv_cpu[layer], 
                                            self._b_recv[layer], tag, self._corr_grad, self._b_avg[layer], forward=False)
                    self.__gloo_all_to_all(grad_scale, self._gscale_cpu[layer], self._gscale_recv_cpu[layer],
                                            self._gscale_recv[layer], (tag+100), self._corr_grad, self._b_avg[layer], forward=False)
                    self.__gloo_all_to_all(grad_mn, self._gmn_cpu[layer], self._gmn_recv_cpu[layer], 
                                            self._gmn_recv[layer], (tag+200), self._corr_grad, self._b_avg[layer], forward=False)
            self._b_cuda_event[layer].record(self._comm_stream)
            if self._corr_grad:
                self._b_cuda_event[layer].record(self._corr_stream)
        else:
            raise NotImplementedError
        self._b_cpu_event[layer].set()


    def __quant_data(self, data, forward):
        rank, size = dist.get_rank(), dist.get_world_size()
        # Data quantilization
        if self.dtype == 'fp16':
            quant_data = data.half()
            data_scale = None
            data_mn = None
        elif self.dtype == 'int8':
            quant_data= data * 100
            quant_data = quant_data.char()
            data_scale = None
            data_mn = None
        elif self.dtype == 'int4' or self.dtype == 'int2' or self.dtype == 'int1':
            quant_data, data_scale, data_mn = [], [], []
            for i in range(size):
                if i == rank:
                    quant_data.append(None)
                    data_scale.append(None)
                    data_mn.append(None)
                else:
                    if forward:
                        data_part, scale, mn = quantize_and_pack(data[self._selected[i]], self.nbits)
                    else:
                        data_part, scale, mn = quantize_and_pack(data[self._pl[i]:self._pr[i]], self.nbits)
                    cshape = data_part.shape
                    self.commu_volume += cshape[0]/1000000
                    # cshape = scale.shape
                    # self.commu_volume += cshape[0]*2/1000000
                    # cshape = mn.shape
                    # self.commu_volume += cshape[0]*2/1000000

                    quant_data.append(data_part)
                    data_scale.append(scale.half())
                    data_mn.append(mn.half())
        else:
            quant_data = data
            data_scale = None
            data_mn = None
            for i in range(size):
                if i == rank:
                    pass
                else:
                    if forward:
                        cshape = data[self._selected[i]].shape
                        self.commu_volume += cshape[0]*cshape[1]*4/1000000
                        # print(f'rank: {rank}, forward size: {cshape[0]*cshape[1]*4/1000000} MB, total: {self.commu_volume} MB')
                    else:
                        cshape = data[self._pl[i]:self._pr[i]].shape
                        self.commu_volume += cshape[0]*cshape[1]*4/1000000
                        # print(f'rank: {rank}, backward size: {cshape[0]*cshape[1]*4/1000000} MB, total: {self.commu_volume} MB')
                
        return quant_data, data_scale, data_mn 