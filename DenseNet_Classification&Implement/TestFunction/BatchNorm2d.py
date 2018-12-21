# -*- coding: utf-8 -*-
# [code](https://blog.csdn.net/tmk_01/article/details/80679549)
# [nn.BatchNorm2d](https://pytorch.org/docs/stable/nn.html#batchnorm2d)

import torch

import torch.nn as nn

m = nn.BatchNorm2d(2,affine=True) #权重w和偏重将被使用
input = torch.randn(1,2,3,4)
output = m(input)

print("输入图片：")
print(input)
print("归一化权重：")
print(m.weight)
print("归一化的偏重：")
print(m.bias)

print("归一化的输出：")
print(output)
print("输出的尺度：")
print(output.size())

# i = torch.randn(1,1,2)
print("输入的第一个维度：")
print(input[0][0])
firstDimenMean = torch.Tensor.mean(input[0][0])
firstDimenVar= torch.Tensor.var(input[0][0],False) #Bessel's Correction贝塞尔校正不会被使用

print(m.eps)
print("输入的第一个维度平均值：")
print(firstDimenMean)
print("输入的第一个维度方差：")
print(firstDimenVar)

bacthnormone = \
    ((input[0][0][0][0] - firstDimenMean)/(torch.pow(firstDimenVar+m.eps,0.5) ))\
               * m.weight[0] + m.bias[0]
print(bacthnormone)

'''
输入图片：
tensor([[[[ 0.7660,  1.4679, -0.6296, -0.0719],
          [-2.1087,  1.9102, -2.0127,  0.7351],
          [ 0.1641, -0.9963, -0.7625, -0.8602]],

         [[-1.5979, -1.0518, -0.0357, -0.1237],
          [-0.3339, -0.3031, -1.7708, -0.5184],
          [-0.1858, -0.5217,  1.1664,  1.1214]]]])
归一化权重：
Parameter containing:
tensor([0.9051, 0.7635], requires_grad=True)
归一化的偏重：
Parameter containing:
tensor([0., 0.], requires_grad=True)
归一化的输出：
tensor([[[[ 0.7200,  1.2432, -0.3203,  0.0954],
          [-1.4228,  1.5728, -1.3513,  0.6969],
          [ 0.2713, -0.5937, -0.4194, -0.4922]],

         [[-1.1178, -0.6301,  0.2774,  0.1988],
          [ 0.0111,  0.0385, -1.2723, -0.1537],
          [ 0.1433, -0.1567,  1.3509,  1.3107]]]],
       grad_fn=<NativeBatchNormBackward>)
输出的尺度：
(1, 2, 3, 4)
输入的第一个维度：
tensor([[ 0.7660,  1.4679, -0.6296, -0.0719],
        [-2.1087,  1.9102, -2.0127,  0.7351],
        [ 0.1641, -0.9963, -0.7625, -0.8602]])
1e-05
输入的第一个维度平均值：
tensor(-0.1999)
输入的第一个维度方差：
tensor(1.4743)
tensor(0.7200, grad_fn=<AddBackward0>)
'''