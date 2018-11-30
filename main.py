import torch.nn as nn
import torchvision.models as models

USE_CUDA = True
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
    model = get_feature_extractor()
    feat_x = get_feature(model, 'Put image tensor here', FEATURE_ID)
    feat_x_rec = get_feature(model, 'Put image tensor here', FEATURE_ID)


if __name__ == '__main__':
    main()
