from thop import profile, clever_format

class FL(object):
    def __init__(self,model):

        self.model = model

    # 计算每一层的访存量
    def calculate_memory_access(self, model, input_data):
        total_memory = 0

        # 计算每一层的输入数据、输出数据和参数的大小
        def memory_hooks(module, input, output):
            nonlocal total_memory

            # 输入数据大小
            input_size = input[0].element_size() * input[0].numel()
            total_memory += input_size

            # 输出数据大小
            output_size = output.element_size() * output.numel()
            total_memory += output_size

            # 参数大小
            if hasattr(module, 'weight'):
                parameter_size = module.weight.element_size() * module.weight.numel()
                total_memory += parameter_size

        # 注册 hook 并执行模型
        hooks = []
        for layer in model.children():
            hooks.append(layer.register_forward_hook(memory_hooks))

        # 执行模型
        model(input_data)

        # 移除 hook
        for hook in hooks:
            hook.remove()

        return total_memory

    # 计算参数量和flops
    def flops(self, input_data):
        flops, params = profile(self.model, inputs=(input_data,))
        flops, params = clever_format([flops, params], "%.3f")

        return flops, params

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
import logging

# 设置 Thop 库的日志级别为 ERROR
logging.getLogger('thop').setLevel(logging.ERROR)

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]))
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.flag = 0
        self.T = T
        self.memory = self.memory

    def forward(self, x, y):
        out = torch.mm(x, self.memory.t()) / self.T
        return out

    def update_weight(self, features, index):
        if not self.flag:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(0.0)
            weight_pos.add_(torch.mul(features.data, 1.0))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
            self.flag = 1
        else:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory)

    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)

