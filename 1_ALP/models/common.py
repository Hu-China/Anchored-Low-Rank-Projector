import numpy as np
import torch
import torch.nn as nn


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


def act(act_fun='LeakyReLU'):
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return nn.SiLU()  # PyTorch built-in Swish
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Identity()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


# -------------------------------------------------------
# 标准卷积模块 (用于输入层、输出层、1x1卷积)
# -------------------------------------------------------
def standard_conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False  # 简化其他极少用的下采样
        stride = 1

    padder = None
    to_pad = (kernel_size - 1) // 2
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


# -------------------------------------------------------
# 可分离非对称卷积 (用于中间深层，保持轻量化)
# 结构: ReflectionPad -> DW(kx1) -> DW(1xk) -> PW(1x1)
# -------------------------------------------------------
def separable_conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False
        stride = 1

    padder = None
    to_pad = (kernel_size - 1) // 2
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    # 非对称深度卷积 (Asymmetric Depthwise)
    # 3x3 -> 3x1 followed by 1x3
    depthwise_v = nn.Conv2d(in_f, in_f, kernel_size=(kernel_size, 1),
                            stride=(stride, 1), padding=(to_pad, 0),
                            groups=in_f, bias=False)  # 通常DW不带bias
    depthwise_h = nn.Conv2d(in_f, in_f, kernel_size=(1, kernel_size),
                            stride=(1, stride), padding=(0, to_pad),
                            groups=in_f, bias=False)

    # 点卷积 (Pointwise)
    pointwise = nn.Conv2d(in_f, out_f, 1, 1, 0, bias=bias)

    layers = filter(lambda x: x is not None,
                    [padder, depthwise_v, depthwise_h, pointwise, downsampler])
    return nn.Sequential(*layers)
