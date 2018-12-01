import torch.nn as nn
import torchvision.models as models

from config import *
from model import *
from load_dataset import *
from utils import *

USE_CUDA = False
FEATURE_ID = 29


class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def add_layer(self, name, layer):
        self.add_module(name, layer)

    def forward(self, x, feature_id):
        for idx, module in enumerate(self._modules):
            x = self._modules[module](x)
            if idx == feature_id:
                return x


def get_feature_extractor():
    vgg_temp = models.vgg19(pretrained=True).features
    model = FeatureExtractor()

    conv_counter = 1
    relu_counter = 1
    block_counter = 1

    for i, layer in enumerate(list(vgg_temp)):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(block_counter) + "_" + str(conv_counter)
            conv_counter += 1
            model.add_layer(name, layer)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(block_counter) + "_" + str(relu_counter)
            relu_counter += 1
            model.add_layer(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(block_counter)
            relu_counter = conv_counter = 1
            block_counter += 1
            model.add_layer(name, layer)  # Is nn.AvgPool2d((2,2)) better than nn.MaxPool2d?

    if USE_CUDA:
        model.cuda('cuda:2')
    return model


def get_feature(model, img_tensor, feature_id):
    if USE_CUDA:
        img_tensor = img_tensor.cuda('cuda:2')

    feature_tensor = model(img_tensor, feature_id)
    feature = feature_tensor.data.squeeze().cpu().numpy().transpose(1, 2, 0)

    return feature


def main():
    # dataset load, default = 'blackberry'
    train_phone, train_dslr = load_train_dataset(config.model_type['0'], config.data_path, config.batch_size,
                                                config.height * config.width * config.channels)
    #test_phone, test_dslr   = load_test_dataset(config.model_type['0'], config.data_path,
    #                                            config.height * config.width * config.channels)

    # numpy array to torch tensor
    train_phone = torch.from_numpy(train_phone)
    train_dslr  = torch.from_numpy(train_dslr)
    #test_phone  = torch.from_numpy(test_phone)
    #test_dslr   = torch.from_numpy(test_dslr)

    train_phone = train_phone.view(-1, config.channels, config.height, config.width)
    train_dslr  = train_dslr.view(-1, config.channels, config.height, config.width)
    print ('Check Train and Test data shape')
    print ('Train phone shape : ', train_phone.shape)
    print ('Train canon shape : ', train_dslr.shape)
    #print ('Test phone shape : ', test_phone.shape)
    #print ('Test canon shape : ', test_dslr.shape)

    training = True

    wespe = WESPE(config, USE_CUDA, training)

    train_iter = 1 # 20000
    epochs = 1

    model = get_feature_extractor()
    for e in range(epochs):
        for idx in range(train_iter):
            # 주의해야할 점 : 다시 train data load하면 view 함수로 2 dimension -> 4 dimension으로 만들어줘야한다.

            # 추후에 고칠 예정
            y_hat = wespe.generator_g(train_phone)
            x_hat = wespe.generator_f(y_hat)

            # content loss
            print ("y_hat shape ", y_hat.shape)
            print ("x_hat shape ", x_hat.shape)
            feat_x = get_feature(model, train_phone, FEATURE_ID)
            feat_x_rec = get_feature(model, x_hat, FEATURE_ID)

            # color loss
            # gaussian blur image for discriminator_c
            fake_blur = gaussian_blur(y_hat, config.kernel_size, config.sigma, config.channels,
                                    config.height, config.width)
            real_blur = gaussian_blur(train_dslr, config.kernel_size, config.sigma, config.channels,
                                    config.height, config.width)
            print ("fake blur image shape ", fake_blur.shape)
            print ("real blur image shape ", real_blur.shape)
            # texture loss
            # grayscale image for discriminator_t
            fake_gray = grayscale(y_hat, config.channels, config.height, config.width)
            real_gray = grayscale(train_dslr, config.channels, config.height, config.width)
            print ("fake grayscale image shape ", fake_gray.shape)
            print ("real grayscale image shape ", real_gray.shape)
            # total variation loss

            # all loss sum


if __name__ == '__main__':
    main()
