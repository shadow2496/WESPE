import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn


def gaussian_kernel(kernel_size, nsig, channels):
    """
    kernel_size : filter width and height length
    nsig        : range of gaussian distribution
    channels    : choose how many channel you use, default is 3
    """
    interval = (2 * nsig + 1) / kernel_size
    x = np.linspace(start=-nsig-interval / 2, stop=nsig + interval / 2, num=kernel_size+1)

    new_x = st.norm.cdf(x)
    kernel1d = np.diff(new_x)

    kernel_raw = np.sqrt(np.outer(kernel1d, kernel1d))
    kernel = kernel_raw / np.sum(kernel_raw)  # normalize

    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = np.reshape(out_filter, newshape=(1, 1, kernel_size, kernel_size))  # 4-dimensional shape 21, 21, 1, 1
    out_filter = np.repeat(out_filter, channels, axis=0)
    out_filter = torch.from_numpy(out_filter)

    return out_filter


def filter_forward(image, filters):
    # image = torch.from_numpy(image)
    # image = np.reshape(image, newshape=(-1, channels, height, width))
    return filters(image)


def kernel_to_conv2d(kernel_size, nsig, channels):
    """
    kernel_size : filter width and height length
    nsig        : range of gaussian distribution
    channels    : choose how many channel you use, default is 3
    """
    out_filter = gaussian_kernel(kernel_size, nsig, channels)
    gaussian_filter = nn.Conv2d(channels, channels, kernel_size, groups=channels, bias=False, padding=10)
    gaussian_filter.weight.data = out_filter
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def gaussian_blur(image, kernel_size, sigma, channels, device):
    out_filter = kernel_to_conv2d(kernel_size, sigma, channels).to(device)
    tensor_image = filter_forward(image, out_filter)

    return tensor_image


def gray_scale(image, channels, height, width, device):
    """
    image  : (batch_size, image_size), image_size = image_width * image_height * channels
    height : refers to image_height
    width  : refers to image_width

    return : (batch_size, image_size with one channel)
    """

    # rgb_image = np.reshape(image, newshape=(-1, channels, height, width)) # 3 channel which is rgb
    # gray_image = np.reshape(gray_image, newshape=(-1, 1, height, width))
    # gray_image = torch.from_numpy(gray_image)

    rgb_image = image.view(-1, channels)
    gray_image = torch.matmul(rgb_image, torch.Tensor([0.299, 0.587, 0.114]).to(device))
    gray_image = gray_image.view(-1, 1, height, width)

    return gray_image
