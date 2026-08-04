import torch
from torch import nn


def count_parameters(model, trainable_only=True):
    # total learnable values
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def parameters_by_layer(model):
    return [
        (name, tuple(parameter.shape), parameter.numel())
        for name, parameter in model.named_parameters()
    ]


def count_flops(model, input_shape=(3, 32, 32)):
    totals = {"flops": 0}
    handles = []

    def convolution_hook(module, inputs, output):
        output_height, output_width = output.shape[2], output.shape[3]
        kernel_height, kernel_width = module.kernel_size
        per_output_value = (module.in_channels // module.groups) * kernel_height * kernel_width
        totals["flops"] += output_height * output_width * module.out_channels * per_output_value

    def linear_hook(module, inputs, output):
        totals["flops"] += module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, *input_shape))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return totals["flops"]


def describe(model, input_shape=(3, 32, 32)):
    parameters = count_parameters(model)
    flops = count_flops(model, input_shape)
    return f"{parameters:,} parameters, {flops / 1e6:.1f}M FLOPs at {input_shape}" # mr claude formatted this
