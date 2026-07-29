import torch.nn as nn


# 1. 定义一个辅助函数来获取所有 Conv2d 参数
def get_conv_params(model):
    conv_params = []
    # model.modules() 会递归地返回网络中的所有模块
    for name, module in model.named_modules():
        # print(name)
        if isinstance(module, nn.Conv2d):
            print(f"Adding parameters from layer: {name}")
            # 将该层的 weight 和 bias (如果存在) 加入列表
            for param, _ in module.named_parameters():
                # print(name)
                conv_params.append(f"{name}.{param}")
    return conv_params


def init_method(model_type, image_color):
    path_str = ''

    CNN_weight_name = [
    ]

    if image_color == 'L':
        model_path = f'../pre_models/{model_type}_clean_gray/best_model.pth'
    else:
        model_path = f'../pre_models/{model_type}_clean/best_model.pth'

    return path_str, CNN_weight_name, model_path
