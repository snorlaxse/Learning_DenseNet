# -*- coding:utf-8 -*-
# [pytorch方法测试——卷积（一维）](https://blog.csdn.net/tmk_01/article/details/80669906)
# https://pytorch.org/docs/stable/nn.html#conv1d

import torch
import torch.nn as nn

m = nn.Conv1d(2, 5, 2, stride=2)
print(m)
input = torch.randn(2, 2, 5) 
print(input)  # size: (2, 2, 5)
output = m(input)  # size: (2, 5, 2)  
print(m.weight)  # size: (5, 2, 2)    # (out_channels, in_channels, kernel_size)
print(m.bias)  # size: (1, 5)  #  (out_channels)
print(output)  
print(output.size()) 
print(m.weight[0][0][0] * input[0][0][0] + m.weight[0][0][1] * input[0][0][1] 
+ m.weight[0][1][0] * input[0][1][0] + m.weight[0][1][1] * input[0][1][1] + m.bias[0])  # stride=2
'''
Conv1d(2, 5, kernel_size=(2,), stride=(2,))
tensor([[[-0.0407, -1.1735,  0.3089,  0.3836, -0.6918],
         [ 1.6295, -0.8931, -0.1777,  1.8637,  1.7639]],

        [[-0.1234, -0.8543, -0.7096,  1.7037,  0.6126],
         [ 1.5456, -0.0939, -2.3423, -0.3782, -0.4182]]])
Parameter containing:
tensor([[[-0.2984,  0.0122],
         [ 0.2986, -0.4655]],

        [[-0.1491,  0.4130],
         [ 0.3814,  0.3395]],

        [[-0.1723,  0.1665],
         [-0.2019, -0.0494]],

        [[-0.1965, -0.3058],
         [ 0.3797, -0.0790]],

        [[-0.0091,  0.3376],
         [ 0.2992,  0.0556]]], requires_grad=True)
Parameter containing:
tensor([-0.0668,  0.2757,  0.1555,  0.3930,  0.2633], requires_grad=True)
tensor([[[ 8.3332e-01, -1.0750e+00],
         [ 1.1540e-01,  9.5303e-01],
         [-3.1776e-01,  1.0990e-01],
         [ 1.4490e+00,  3.4058e-04],
         [ 3.0538e-01,  4.4034e-01]],

        [[ 4.6483e-01, -3.5756e-01],
         [ 4.9881e-01,  6.3339e-02],
         [-2.7292e-01,  1.0531e+00],
         [ 1.2727e+00, -8.4795e-01],
         [ 4.3316e-01,  1.2312e-01]]], grad_fn=<SqueezeBackward1>)
(2, 5, 2)
tensor(0.8333, grad_fn=<AddBackward0>)
'''
'''
结论：一维卷积第一个输出 = 
m.weight[0][0][0] * input[0][0][0] + m.weight[0][0][1] * input[0][0][1] 
+ m.weight[0][1][0] * input[0][1][0] + m.weight[0][1][1] * input[0][1][1] + m.bias[0]
'''
