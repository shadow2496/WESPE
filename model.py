import torch.nn as nn

from config import config

class Flatten(nn.Module):
    def forward(self, x):
        n = x.shape[0]
        return x.view(n, 128 * 7 * 7)


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, repeat_num=4):
        super(Generator, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(3, 64, kernel_size=9, padding=4, bias=True))
        layers.append(nn.ReLU())

        for _ in range(repeat_num):
            layers.append(ResidualBlock())

        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(64, 3, kernel_size=9, padding=4, bias=True))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(3, 48, kernel_size=11, padding=5, stride=4, bias=True))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(48, 128, kernel_size=5, padding=2, stride=2, bias=True))
        layers.append(nn.BatchNorm2d(128, momentum=0.1))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(192, momentum=0.1))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(192, momentum=0.1))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=2, bias=True))
        layers.append(nn.BatchNorm2d(128, momentum=0.1))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(Flatten())
        layers.append(nn.Linear(128 * 7 * 7, 1024))
        layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Linear(1024, 2))
        layers.append(nn.Softmax(dim=2))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class WESPE:

    def __init__(self, config, USE_CUDA=True, training=True):
        self.generator_g = Generator()
        self.generator_f = Generator()
        self.USE_CUDA    = USE_CUDA
        self.training    = training

        if self.USE_CUDA:
            self.generator_g.cuda()
            self.generator_f.cuda()

        if training:
            self.discriminator_c = Discriminator()
            self.discriminator_t = Discriminator()
