from config import config
from load_dataset import *
from utils import *
from model import *

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


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


def get_feature_extractor(device):
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

    model.to(device)
    return model


def get_feature(model, img_tensor, feature_id):
    img_normalized = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
    print("img_normalized mean : ", img_normalized.view(-1, config.channels, config.height * config.width).mean(2))
    print("img_normalized std : ", img_normalized.view(-1, config.channels, config.height * config.width).std(2))
    feature = model(img_normalized, feature_id)
    # feature = feature.data.squeeze().cpu().numpy().transpose(1, 2, 0)
    return feature


def main():
    # dataset load, default = 'blackberry'
    train_phone, train_dslr = load_train_dataset(config.model_type['0'], config.data_path, config.batch_size,
                                                 config.height * config.width * config.channels)
    # test_phone, test_dslr = load_test_dataset(config.model_type['0'], config.data_path,
    #                                           config.height * config.width * config.channels)

    # numpy array to torch tensor
    train_phone = torch.from_numpy(train_phone)
    train_dslr = torch.from_numpy(train_dslr)
    # test_phone = torch.from_numpy(test_phone)
    # test_dslr = torch.from_numpy(test_dslr)

    train_phone = train_phone.view(-1, config.channels, config.height, config.width)
    train_dslr = train_dslr.view(-1, config.channels, config.height, config.width)
    print('Check Train and Test data shape')
    print('Train phone shape : ', train_phone.size())
    print('Train canon shape : ', train_dslr.size())
    # print('Test phone shape : ', test_phone.size())
    # print('Test canon shape : ', test_dslr.size())

    device = torch.device('cuda:3' if config.use_cuda else 'cpu')
    model = WESPE(config, device)
    extractor = get_feature_extractor(device)

    true_labels = torch.ones(config.batch_size, dtype=torch.long).to(device)
    false_labels = torch.zeros(config.batch_size, dtype=torch.long).to(device)
    for idx in range(config.train_iters):
        train_phone, train_dslr = load_train_dataset(config.model_type['0'], config.data_path, config.batch_size,
                                                     config.height * config.width * config.channels)
        x = torch.from_numpy(train_phone).view(-1, config.channels, config.height, config.width)
        y_real = torch.from_numpy(train_dslr).view(-1, config.channels, config.height, config.width)
        x = x.to(device)
        y_real = y_real.to(device)

        # 추후에 고칠 예정
        y_fake = model.gen_g(train_phone)
        x_rec = model.gen_f(y_fake)
        print("y_fake shape : ", y_fake.size())
        print("x_rec shape : ", x_rec.size())

        # content loss
        feat_x = get_feature(extractor, x, config.feature_id).detach()
        feat_x_rec = get_feature(extractor, x_rec, config.feature_id)
        print("feat_x shape : ", feat_x.size())
        print("feat_x_rec : ", feat_x_rec.size())
        loss_content = torch.pow(feat_x - feat_x_rec, 2).mean()

        # color loss
        # gaussian blur image for discriminator_c
        fake_blur = gaussian_blur(y_fake, config.kernel_size, config.sigma, config.channels,
                                  config.height, config.width)
        real_blur = gaussian_blur(y_real, config.kernel_size, config.sigma, config.channels,
                                  config.height, config.width)
        print("fake blur image shape : ", fake_blur.size())
        print("real blur image shape : ", real_blur.size())
        logits_fake_blur = model.dis_c(y_fake)
        logits_real_blur = model.dis_c(y_real)
        loss_color = model.criterion(logits_fake_blur, true_labels)

        # texture loss
        # gray-scale image for discriminator_t
        fake_gray = gray_scale(y_fake, config.channels, config.height, config.width)
        real_gray = gray_scale(train_dslr, config.channels, config.height, config.width)
        print("fake grayscale image shape : ", fake_gray.size())
        print("real grayscale image shape : ", real_gray.size())
        logits_fake_gray = model.dis_t(y_fake)
        logits_real_gray = model.dis_t(y_real)
        loss_texture = model.criterion(logits_fake_gray, true_labels)

        # total variation loss

        # all loss sum
        loss = loss_content + config.lambda_color * loss_content + config.lambda_texture * loss_texture
        print("Iteration : ", str(idx + 1) + "/" + str(config.total_iters), "Loss : {0:.4f}".format(loss.data))
        print("Loss_content : {0:.4f}, Loss_color : {1:.4f}, Loss_texture : {2:.4f}".format(loss_content.data,
                                                                                            loss_color.data,
                                                                                            loss_texture.data))
        model.g_optimizer.zero_grad()
        model.f_optimizer.zero_grad()
        loss.backward()
        model.g_optimizer.step()
        model.f_optimizer.step()


if __name__ == '__main__':
    main()
