from .common import *


def get_conv_layer(in_c, out_c, k_size, stride=1, bias=True, pad='zero', downsample_mode='stride',
                   force_standard=False):
    """
    智能选择卷积层工厂函数:
    1. 如果 kernel_size == 1 (1x1卷积)，强制使用标准卷积 (separable无意义且更慢)。
    2. 如果 input_channels <= 4 (输入层)，强制使用标准卷积 (避免3通道被过度拆解)。
    3. 否则 (深层网络)，使用 separable_conv (非对称可分离卷积) 以节省参数。
    """
    if k_size == 1 or in_c <= 4 or force_standard:
        return standard_conv(in_c, out_c, k_size, stride, bias, pad, downsample_mode)
    else:
        return separable_conv(in_c, out_c, k_size, stride, bias, pad, downsample_mode)


def low_rank_skip(
        num_input_channels=3, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='reflection', upsample_mode='bilinear', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    # 规范化参数格式
    if not isinstance(upsample_mode, (list, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not isinstance(downsample_mode, (list, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    if not isinstance(filter_size_down, (list, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    if not isinstance(filter_size_up, (list, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1
    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels

    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        # --------------------------------
        # 1. 构建 Skip Connection (Branch 0)
        # --------------------------------
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))

            # 简化点：Skip层通常是1x1，get_conv_layer会自动选择标准卷积
            # 即使 filter_skip_size > 1，如果是第一层(in=3)，也会选标准卷积
            skip.add(get_conv_layer(input_depth, num_channels_skip[i], filter_skip_size,
                                    bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
        else:
            model_tmp.add(deeper)

        # Concat 后的 BN
        # 注意：这里计算了 concat 后的总通道数
        concat_channels = num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])
        model_tmp.add(bn(concat_channels))

        # --------------------------------
        # 2. 构建下采样路径 (Branch 1 - Encoder)
        # --------------------------------
        # 简化点：如果是 Layer 0 (input_depth=3), get_conv_layer 会返回标准 Conv2d(3, ..., stride=2)
        # 避免了原先的 3x1 -> 1x3 拆解
        deeper.add(get_conv_layer(input_depth, num_channels_down[i], filter_size_down[i], stride=2,
                                  bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        # 中间处理层
        deeper.add(get_conv_layer(num_channels_down[i], num_channels_down[i], filter_size_down[i],
                                  bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        # --------------------------------
        # 3. 递归构建下一层
        # --------------------------------
        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # 最底层 (The deepest)
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        # --------------------------------
        # 4. 上采样 (Upsample)
        # --------------------------------
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        # --------------------------------
        # 5. 特征融合处理 (Mix)
        # --------------------------------
        # 这里的输入通道是 skip + deep_feature
        in_ch_up = num_channels_skip[i] + k
        model_tmp.add(get_conv_layer(in_ch_up, num_channels_up[i], filter_size_up[i], 1,
                                     bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        # --------------------------------
        # 6. 1x1 变换 (Post-processing per scale)
        # --------------------------------
        if need1x1_up:
            # 简化点：因为 k=1，get_conv_layer 自动选择标准卷积，避免 separable overhead
            model_tmp.add(get_conv_layer(num_channels_up[i], num_channels_up[i], 1,
                                         bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    # --------------------------------
    # 7. 最终输出层
    # --------------------------------
    # 简化点：输出层直接映射回 num_output_channels (RGB)，使用标准卷积
    model.add(standard_conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model