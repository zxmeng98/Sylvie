from re import A
import torch
from scipy.spatial.distance import cdist
import numpy as np
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


# m0 = torch.cuda.memory_allocated()
# a = torch.tensor([2, 3])
# c = torch.split(b, a.numpy().tolist(), dim=0)
# b = torch.tensor([True, False, True, False, False])
start_bits = [1,1,1,1]
a = torch.nonzero(torch.tensor(start_bits)==2, as_tuple=True)[0]
print(a)

# torch.manual_seed(0)
# a = torch.randn((3, 6)).cuda()
# print(a)
# a += 2
# a_sub = a[:8]
# data, scale, mn = quantize_and_pack(a, 16)
# aa = dequantize_and_unpack(data, 16, a.shape, scale, mn)
# data_sub, scale_sub, mn_sub = quantize_and_pack(a_sub, 1)
# aa_sub = dequantize_and_unpack(data_sub, 1, a_sub.shape, scale_sub, mn_sub)
# a = torch.tensor([i for i in range(5)])
# s1 = torch.Size([32])
# c = torch.cat((a, b))
# m1 = torch.cuda.memory_allocated()
# print(f'size: {a.shape}, type: {a.type()}, before quant: {(m1-m0)/(1024*1024)} MiB')


# quant_a,scale,mn = quantize_and_pack(a, 4)

# print(f'size: {quant_a.shape}, type: {quant_a.type()}, after quant: {(torch.cuda.memory_allocated()-m1)/(1024*1024)} MiB')
# print(quant_a.shape, scale.shape, mn.shape)

# print(f"Full Precision: {full_precision_tensor}, type: {full_precision_tensor.type()}")
# # a = full_precision_tensor * 10
# # print(a)
# a = full_precision_tensor.half()
# print(a)
# b = a.float()
# print(b.shape)
# print(np.mean(a[2:]))


# low_precision_tensor = float_quantize(full_precision_tensor, exp=5, man=2, rounding="nearest")
# print(f"Low Precision: {low_precision_tensor}, type: {low_precision_tensor.type()}")

# low_precision_tensor = t1.char()
# t2 = torch.mul(low_precision_tensor.float(), 0.01)

# print(low_precision_tensor)

# a = [None] * 3
# print(a)
