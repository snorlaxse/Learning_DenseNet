# -*- coding: utf-8 -*-
# 原文：https://blog.csdn.net/tmk_01/article/details/80679991 
# [nn.ReLU](https://pytorch.org/docs/stable/nn.html?highlight=relu#torch.nn.ReLU)

import torch
import torch.nn as nn

#inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
m = nn.ReLU(inplace=True)
input = torch.randn(7)

print("输入处理前图片：")
print(input)

output = m(input)

print("ReLU输出：")
print(output)
print("输出的尺度：")
print(output.size())

print("输入处理后图片：")
print(input)

'''
输入处理前图片：
tensor([-1.0190, -0.5473,  0.6421,  1.3420, -0.8580,  0.2715, -0.0938])
ReLU输出：
tensor([0.0000, 0.0000, 0.6421, 1.3420, 0.0000, 0.2715, 0.0000])
输出的尺度：
(7,)
输入处理后图片：
tensor([0.0000, 0.0000, 0.6421, 1.3420, 0.0000, 0.2715, 0.0000])
'''