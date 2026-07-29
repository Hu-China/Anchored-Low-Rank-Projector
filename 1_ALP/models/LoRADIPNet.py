import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *


# ==========================================
# 核心 LoRA 卷积组件
# ==========================================
class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, rank=4, lora_alpha=1.0, pad_mode='zero'):
        super(LoRAConv2d, self).__init__()

        self.padding = padding
        self.active = True  # 核心开关：控制是否启用 LoRA 分支计算

        # 1. 填充处理
        if pad_mode == 'reflection':
            self.pad_layer = nn.ReflectionPad2d(padding)
            conv_padding = 0
        else:
            self.pad_layer = nn.Identity()
            conv_padding = padding

        # 2. 主干权重 (Pre-train 阶段的目标)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=conv_padding, bias=bias)

        # 3. LoRA 参数 (Fine-tune 阶段的目标)
        self.rank = rank
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, in_channels * kernel_size * kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))  # 初始化为0，确保初态一致

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        # 基础主干计算
        x = self.pad_layer(x)
        base_out = self.conv(x)

        # 如果处于微调阶段且开关开启，计算 LoRA 增量
        if self.active:
            delta_w = (self.lora_B @ self.lora_A).view(self.conv.weight.shape)
            lora_out = F.conv2d(x, delta_w, stride=self.conv.stride, padding=0, bias=None)
            return base_out + lora_out * self.scaling

        return base_out


# ==========================================
# 主体网络类
# ==========================================
class LoRASkipNet(nn.Module):
    def __init__(self,
                 num_input_channels=3, num_output_channels=3,
                 num_channels_down=[16, 32, 64, 128],
                 num_channels_up=[16, 32, 64, 128],
                 num_channels_skip=[4, 4, 4, 4],
                 filter_size_down=3, filter_size_up=3,
                 need_sigmoid=True, need_bias=True,
                 pad='zero', upsample_mode='bilinear', act_fun='LeakyReLU',
                 use_lora=True, lora_rank=8):
        super(LoRASkipNet, self).__init__()

        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.n_scales = len(num_channels_down)

        def to_list(x): return x if isinstance(x, (list, tuple)) else [x] * self.n_scales

        self.f_down = to_list(filter_size_down)
        self.f_up = to_list(filter_size_up)

        self.model = nn.Sequential()
        self._build_network(self.model, num_input_channels, num_channels_down,
                            num_channels_up, num_channels_skip, need_bias,
                            pad, act_fun, upsample_mode)

        self.output_layer = nn.Sequential(
            standard_conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad)
        )
        if need_sigmoid:
            self.output_layer.add_module('sigmoid', nn.Sigmoid())

    def _get_conv(self, in_f, out_f, k, s=1, bias=True, pad='zero'):
        p = (k - 1) // 2
        if not self.use_lora:
            return standard_conv(in_f, out_f, k, s, bias, pad)
        return LoRAConv2d(in_f, out_f, k, stride=s, padding=p, bias=bias, rank=self.lora_rank, pad_mode=pad)

    def _build_network(self, model_tmp, input_depth, down, up, skip, bias, pad, act_fun, up_mode):
        curr_ref = model_tmp
        in_d = input_depth
        for i in range(self.n_scales):
            deeper = nn.Sequential()
            skip_layer = nn.Sequential()

            if skip[i] != 0:
                curr_ref.add_module(f'concat_{i}', Concat(1, skip_layer, deeper))
            else:
                curr_ref.add_module(f'deeper_{i}', deeper)

            # 下采样路径
            deeper.add_module('d_conv1', self._get_conv(in_d, down[i], self.f_down[i], 2, bias, pad))
            deeper.add_module('d_bn1', bn(down[i]))
            deeper.add_module('d_act1', act(act_fun))
            deeper.add_module('d_conv2', self._get_conv(down[i], down[i], self.f_down[i], 1, bias, pad))
            deeper.add_module('d_bn2', bn(down[i]))
            deeper.add_module('d_act2', act(act_fun))

            # Skip 分支
            if skip[i] != 0:
                skip_layer.add_module('s_conv', self._get_conv(in_d, skip[i], 1, 1, bias, pad))
                skip_layer.add_module('s_bn', bn(skip[i]))
                skip_layer.add_module('s_act', act(act_fun))

            deeper_main = nn.Sequential()
            if i < self.n_scales - 1:
                deeper.add_module('main', deeper_main)
                k_val = up[i + 1]
            else:
                k_val = down[i]

            deeper.add_module('up', nn.Upsample(scale_factor=2, mode=up_mode))

            # 上采样合并路径
            curr_ref.add_module(f'bn_merge_{i}', bn(skip[i] + (up[i + 1] if i < self.n_scales - 1 else down[i])))
            curr_ref.add_module(f'u_conv', self._get_conv(skip[i] + k_val, up[i], self.f_up[i], 1, bias, pad))
            curr_ref.add_module(f'u_bn', bn(up[i]))
            curr_ref.add_module(f'u_act', act(act_fun))

            in_d = down[i]
            curr_ref = deeper_main

    def set_stage(self, stage="pretrain"):
        """管理模型阶段：pretrain 或 finetune"""
        if stage == "pretrain":
            print(">>> Stage: PRE-TRAIN (LoRA Off)")
            for m in self.modules():
                if isinstance(m, LoRAConv2d): m.active = False
            for p in self.parameters(): p.requires_grad = True
        elif stage == "finetune":
            print(">>> Stage: FINE-TUNE (LoRA Only)")
            for m in self.modules():
                if isinstance(m, LoRAConv2d): m.active = True
            for n, p in self.named_parameters():
                p.requires_grad = True if 'lora_' in n else False

    def forward(self, x):
        return self.output_layer(self.model(x))


# ==========================================
# 执行流程演示
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_img = torch.randn(1, 3, 512, 512).to(device)

    # 1. 预训练阶段
    net = LoRASkipNet(use_lora=True, lora_rank=8, pad='reflection').to(device)
    net.set_stage("pretrain")

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # 模拟训练 1 步并保存
    out = net(dummy_img)
    torch.save(net.state_dict(), 'base_model.pth')
    print("Pre-train model saved.")

    # 2. 微调阶段 (DIP 任务)
    print("\n--- Starting Fine-tuning ---")
    new_net = LoRASkipNet(use_lora=True, lora_rank=8, pad='reflection').to(device)
    new_net.load_state_dict(torch.load('base_model.pth'))
    new_net.set_stage("finetune")

    # 只优化 LoRA 参数
    lora_params = [p for p in new_net.parameters() if p.requires_grad]
    optimizer_dip = torch.optim.Adam(lora_params, lr=1e-2)

    for i in range(5):
        optimizer_dip.zero_grad()
        output = new_net(dummy_img)
        loss = F.mse_loss(output, dummy_img)  # 模拟拟合
        loss.backward()
        optimizer_dip.step()
        print(f"DIP Step {i + 1}, Loss: {loss.item():.6f}")